#!/usr/bin/env python3
"""Train FNO / Transolver / CNextU one-step operators on VAE latent trajectories."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class LatentTrajectoryDataset(Dataset):
    def __init__(
        self,
        latent_root: str | Path,
        split: str,
        *,
        history: int,
        norm_stats: dict | None = None,
        max_shards: int | None = None,
        max_samples_per_shard: int | None = None,
        full_rollout: bool = False,
        use_phys: bool = True,
    ):
        self.root = Path(latent_root).resolve()
        self.split = split
        self.split_dir = self.root / split
        self.manifest = torch.load(self.split_dir / "manifest.pt", map_location="cpu", weights_only=False)
        self.history = int(history)
        self.full_rollout = bool(full_rollout)
        self.use_phys = bool(use_phys)
        self.norm = norm_stats or torch.load(self.root / "norm_stats_latent.pt", map_location="cpu", weights_only=False)
        self.mean = torch.tensor(self.norm["mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(self.norm["std"], dtype=torch.float32).view(1, -1, 1, 1, 1)
        self._cache = {}
        shards = list(self.manifest["shards"])
        if max_shards is not None:
            shards = shards[:max_shards]
        self.shards = shards
        self.index = []
        for si, sh in enumerate(self.shards):
            d = self._load(si)
            n, _c, frames, _h, _w = d["z"].shape
            if max_samples_per_shard is not None:
                n = min(n, int(max_samples_per_shard))
            if full_rollout:
                self.index.extend((si, li, 0) for li in range(n))
            else:
                for li in range(n):
                    self.index.extend((si, li, t) for t in range(self.history, frames))

    def _load(self, si: int):
        if si not in self._cache:
            sh = self.shards[si]
            self._cache = {si: torch.load(self.split_dir / sh["path"], map_location="cpu", weights_only=False)}
        return self._cache[si]

    def __len__(self):
        return len(self.index)

    def _phys(self, regime, h: int, w: int) -> torch.Tensor:
        if not self.use_phys:
            return torch.empty(0, h, w)
        visc = float(regime.get("visc", 1e-3))
        k = float(regime.get("forcing_k", 1.0))
        amp = float(regime.get("forcing_amp", 0.1))
        vals = torch.tensor([math.log(1.0 / visc) - 8.0, k / 4.0, amp / 0.1], dtype=torch.float32)
        return vals[:, None, None].expand(-1, h, w)

    def __getitem__(self, idx):
        si, li, t = self.index[idx]
        d = self._load(si)
        z = d["z"][li:li + 1]
        z = ((z - self.mean) / (self.std + 1e-8))[0]
        c, frames, h, w = z.shape
        phys = self._phys(d.get("regime", {}), h, w)
        if self.full_rollout:
            return z, phys
        hist = z[:, t - self.history:t].permute(1, 0, 2, 3).reshape(self.history * c, h, w)
        if self.use_phys:
            hist = torch.cat([hist, phys], dim=0)
        target = z[:, t]
        return hist.contiguous(), target.contiguous()


def add_act_root(path: str | None):
    if path is None:
        path = "/media/home/lizhonghao/act/ACT-NeuralOP"
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"ACT-NeuralOP root not found: {p}")
    sys.path.insert(0, str(p))


def unwrap_model_output(y):
    return y[0] if isinstance(y, tuple) else y


def build_model(name: str, dim_in: int, dim_out: int, h: int, w: int, args):
    import benchmark.models as models

    if name == "fno":
        return models.ACTFNO(
            dim_in=dim_in,
            dim_out=dim_out,
            n_spatial_dims=2,
            spatial_resolution=(h, w),
            modes1=min(args.modes, h // 2),
            modes2=min(args.modes, w // 2),
            hidden_channels=args.hidden,
            gradient_checkpointing=args.grad_ckpt,
            use_act=False,
        )
    if name == "transolver":
        return models.TransolverPlus(
            dim_in=dim_in,
            dim_out=dim_out,
            n_spatial_dims=2,
            spatial_resolution=(h, w),
            n_layers=args.layers,
            n_hidden=args.hidden,
            n_head=args.heads,
            slice_num=args.slice_num,
            gradient_checkpointing=args.grad_ckpt,
            use_act=False,
        )
    if name == "cnextu":
        return models.UNetConvNext(
            dim_in=dim_in,
            dim_out=dim_out,
            n_spatial_dims=2,
            spatial_resolution=(h, w),
            stages=args.stages,
            blocks_per_stage=args.blocks_per_stage,
            blocks_at_neck=1,
            init_features=args.hidden,
            gradient_checkpointing=args.grad_ckpt,
        )
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def eval_one_step(model, loader, device, max_batches: int = 0) -> float:
    model.eval()
    total = 0.0
    n = 0
    for bi, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        pred = unwrap_model_output(model(x))
        rel = (pred - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)
        total += float(rel) * x.size(0)
        n += x.size(0)
        if max_batches and bi + 1 >= max_batches:
            break
    return total / max(n, 1)


@torch.no_grad()
def eval_rollout(model, loader, device, history: int, use_phys: bool, max_batches: int = 0) -> float:
    model.eval()
    total = 0.0
    n = 0
    for bi, (z, phys) in enumerate(loader):
        z = z.to(device)
        phys = phys.to(device)
        b, c, frames, h, w = z.shape
        hist = [z[:, :, i] for i in range(history)]
        preds = []
        for t in range(history, frames):
            x = torch.stack(hist[-history:], dim=1).reshape(b, history * c, h, w)
            if use_phys:
                x = torch.cat([x, phys], dim=1)
            y = unwrap_model_output(model(x))
            preds.append(y)
            hist.append(y)
        pred_seq = torch.stack(preds, dim=2)
        true_seq = z[:, :, history:]
        rel = (pred_seq - true_seq).square().mean().sqrt() / (true_seq.square().mean().sqrt() + 1e-8)
        total += float(rel) * b
        n += b
        if max_batches and bi + 1 >= max_batches:
            break
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-root", default="runs/latents")
    ap.add_argument("--act-root", default="/media/home/lizhonghao/act/ACT-NeuralOP")
    ap.add_argument("--models", nargs="+", default=["fno", "transolver", "cnextu"])
    ap.add_argument("--out-csv", default="runs/latent_operator/results.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--no-phys", dest="use_phys", action="store_false")
    ap.set_defaults(use_phys=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--slice-num", type=int, default=32)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--blocks-per-stage", type=int, default=1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=0)
    args = ap.parse_args()

    add_act_root(args.act_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = torch.load(Path(args.latent_root) / "norm_stats_latent.pt", map_location="cpu", weights_only=False)
    train_ds = LatentTrajectoryDataset(
        args.latent_root,
        "train",
        history=args.history,
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        use_phys=args.use_phys,
    )
    valid_ds = LatentTrajectoryDataset(
        args.latent_root,
        "valid",
        history=args.history,
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        use_phys=args.use_phys,
    )
    roll_ds = LatentTrajectoryDataset(
        args.latent_root,
        "valid",
        history=args.history,
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        full_rollout=True,
        use_phys=args.use_phys,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    roll_dl = DataLoader(roll_ds, batch_size=max(1, args.batch // 2), shuffle=False, num_workers=0)
    x0, y0 = train_ds[0]
    dim_in, h, w = x0.shape
    dim_out = y0.shape[0]

    rows = []
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        model = build_model(name, dim_in, dim_out, h, w, args).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best = float("inf")
        t_start = time.time()
        for ep in range(args.epochs):
            model.train()
            tsum = 0.0
            tn = 0
            for x, y in train_dl:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                pred = unwrap_model_output(model(x))
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tsum += float(loss.detach()) * x.size(0)
                tn += x.size(0)
            val = eval_one_step(model, valid_dl, device, args.eval_batches)
            best = min(best, val)
            print(f"{name} ep={ep:03d} train_mse={tsum / max(tn, 1):.4g} val_rel={val:.4g}", flush=True)
        rollout = eval_rollout(model, roll_dl, device, args.history, args.use_phys, args.eval_batches)
        row = {
            "model": name,
            "best_one_step_rel": best,
            "rollout_rel": rollout,
            "epochs": args.epochs,
            "history": args.history,
            "seconds": time.time() - t_start,
        }
        rows.append(row)
        torch.save({"model": model.state_dict(), "args": vars(args), "row": row}, out_csv.parent / f"{name}_latent_operator.pt")
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
