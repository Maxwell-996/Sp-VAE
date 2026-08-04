#!/usr/bin/env python3
"""Train FNO / Transolver / CNextU one-step operators on VAE latent trajectories."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfvae.learnable_moe import LearnableTokenMoE


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
            self._cache[si] = torch.load(
                self.split_dir / sh["path"],
                map_location="cpu",
                weights_only=False,
            )
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


class OperatorMoEAdapter(torch.nn.Module):
    """Zero-init routed correction on an operator's predicted latent field."""

    def __init__(self, base: torch.nn.Module, channels: int, args):
        super().__init__()
        self.base = base
        self.moe = LearnableTokenMoE(
            channels,
            hidden=getattr(args, "operator_moe_hidden", 64),
            num_experts=getattr(args, "operator_moe_experts", 4),
            top_k=getattr(args, "operator_moe_top_k", 2),
            global_context=getattr(args, "operator_moe_global", False),
            router_inhibition=getattr(args, "operator_moe_router_inhibition", False),
            local_context=getattr(args, "operator_moe_local_context", False),
            shared_experts=getattr(args, "operator_moe_shared_experts", 1),
            router_noise_std=getattr(args, "operator_moe_router_noise", 0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(unwrap_model_output(self.base(x)))


class OperatorInternalMoE(torch.nn.Module):
    """Inject zero-init token MoEs after an operator's hidden blocks."""

    def __init__(self, base: torch.nn.Module, name: str, args):
        super().__init__()
        self.base = base
        targets, channels = self._targets_and_channels(name, base, args)
        self.adapters = torch.nn.ModuleList(
            [
                LearnableTokenMoE(
                    channel_count,
                    hidden=getattr(args, "operator_moe_hidden", 64),
                    num_experts=getattr(args, "operator_moe_experts", 4),
                    top_k=getattr(args, "operator_moe_top_k", 2),
                    global_context=getattr(args, "operator_moe_global", False),
                    shared_experts=getattr(args, "operator_moe_shared_experts", 1),
                    router_noise_std=getattr(args, "operator_moe_router_noise", 0.0),
                )
                for channel_count in channels
            ]
        )
        self._hooks = [
            module.register_forward_hook(self._make_hook(adapter))
            for module, adapter in zip(targets, self.adapters)
        ]

    @staticmethod
    def _targets_and_channels(name: str, base: torch.nn.Module, args):
        if name == "fno":
            targets = list(base.model.blocks)
            channels = [args.hidden] * len(targets)
        elif name == "transolver":
            # The last block projects to the output channels, so keep MoE on
            # the hidden token states only.
            targets = list(base.blocks[:-1])
            channels = [args.hidden] * len(targets)
        elif name == "cnextu":
            scope = getattr(args, "operator_internal_moe_scope", "all")
            if scope == "decoder":
                targets = list(base.decoder)
            elif scope == "neck_decoder":
                targets = [base.neck] + list(base.decoder)
            elif scope == "all":
                targets = list(base.encoder) + [base.neck] + list(base.decoder)
            else:
                raise ValueError(f"unknown CNextU internal-MoE scope: {scope}")
            channels = []
            for module in targets:
                resample = getattr(module, "resample", None)
                out_channels = None
                if resample is not None:
                    for child in resample.modules():
                        if hasattr(child, "out_channels"):
                            out_channels = int(child.out_channels)
                if out_channels is None:
                    for child in module.modules():
                        if hasattr(child, "out_features"):
                            out_channels = int(child.out_features)
                if out_channels is None:
                    raise ValueError("unable to infer CNextU stage channels")
                channels.append(out_channels)
        else:
            raise ValueError(f"unknown internal-MoE operator: {name}")
        if not targets:
            raise ValueError(f"no internal-MoE targets found for {name}")
        return targets, channels

    @staticmethod
    def _make_hook(adapter: LearnableTokenMoE):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (adapter(output[0]), *output[1:])
            return adapter(output)

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unwrap_model_output(self.base(x))


def maybe_wrap_operator(
    model: torch.nn.Module, dim_out: int, name: str, args
) -> torch.nn.Module:
    if getattr(args, "operator_internal_moe", False):
        model = OperatorInternalMoE(model, name, args)
    if not getattr(args, "operator_moe", False):
        return model
    return OperatorMoEAdapter(model, dim_out, args)


def build_model(name: str, dim_in: int, dim_out: int, h: int, w: int, args):
    import benchmark.models as models

    if name == "fno":
        model = models.ACTFNO(
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
        return maybe_wrap_operator(model, dim_out, name, args)
    if name == "transolver":
        model = models.TransolverPlus(
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
        return maybe_wrap_operator(model, dim_out, name, args)
    if name == "cnextu":
        model = models.UNetConvNext(
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
        return maybe_wrap_operator(model, dim_out, name, args)
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def eval_one_step(model, loader, device, max_batches: int = 0, distributed: bool = False) -> float:
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
    if distributed:
        stats = torch.tensor([total, n], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, n = stats.tolist()
    return total / max(n, 1.0)


@torch.no_grad()
def eval_rollout(
    model,
    loader,
    device,
    history: int,
    use_phys: bool,
    max_batches: int = 0,
    distributed: bool = False,
) -> float:
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
    if distributed:
        stats = torch.tensor([total, n], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, n = stats.tolist()
    return total / max(n, 1.0)


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
    ap.add_argument("--operator-internal-moe", action="store_true")
    ap.add_argument(
        "--operator-internal-moe-scope",
        choices=["all", "decoder", "neck_decoder"],
        default="all",
    )
    ap.add_argument("--operator-moe", action="store_true")
    ap.add_argument("--operator-moe-hidden", type=int, default=64)
    ap.add_argument("--operator-moe-experts", type=int, default=4)
    ap.add_argument("--operator-moe-top-k", type=int, default=2)
    ap.add_argument("--operator-moe-global", action="store_true")
    ap.add_argument("--operator-moe-router-inhibition", action="store_true")
    ap.add_argument("--operator-moe-local-context", action="store_true")
    ap.add_argument("--operator-moe-shared-experts", type=int, default=1)
    ap.add_argument("--operator-moe-router-noise", type=float, default=0.0)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-GPU operator training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)

    add_act_root(args.act_root)
    seed_all(args.seed)
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
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
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    roll_sampler = (
        DistributedSampler(roll_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=train_sampler,
        shuffle=not distributed,
        num_workers=0,
        drop_last=True,
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=args.batch, sampler=valid_sampler, shuffle=False, num_workers=0
    )
    roll_dl = DataLoader(
        roll_ds,
        batch_size=max(1, args.batch // 2),
        sampler=roll_sampler,
        shuffle=False,
        num_workers=0,
    )
    test_ds = LatentTrajectoryDataset(
        args.latent_root,
        "test",
        history=args.history,
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        use_phys=args.use_phys,
    )
    test_roll_ds = LatentTrajectoryDataset(
        args.latent_root,
        "test",
        history=args.history,
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        full_rollout=True,
        use_phys=args.use_phys,
    )
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    test_roll_sampler = (
        DistributedSampler(test_roll_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch, sampler=test_sampler, shuffle=False, num_workers=0
    )
    test_roll_dl = DataLoader(
        test_roll_ds,
        batch_size=max(1, args.batch // 2),
        sampler=test_roll_sampler,
        shuffle=False,
        num_workers=0,
    )
    x0, y0 = train_ds[0]
    dim_in, h, w = x0.shape
    dim_out = y0.shape[0]

    rows = []
    out_csv = Path(args.out_csv).resolve()
    if rank == 0:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    for name in args.models:
        seed_all(args.seed)
        model = build_model(name, dim_in, dim_out, h, w, args).to(device)
        train_model: torch.nn.Module = model
        if distributed:
            train_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        opt = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=1e-4)
        best = float("inf")
        best_state = None
        t_start = time.time()
        for ep in range(args.epochs):
            if distributed:
                train_sampler.set_epoch(ep)
            train_model.train()
            tsum = 0.0
            tn = 0
            for x, y in train_dl:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                pred = unwrap_model_output(train_model(x))
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                opt.step()
                tsum += float(loss.detach()) * x.size(0)
                tn += x.size(0)
            if distributed:
                stats = torch.tensor([tsum, tn], device=device, dtype=torch.float64)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                tsum, tn = stats.tolist()
            val = eval_one_step(
                train_model, valid_dl, device, args.eval_batches, distributed
            )
            if val < best:
                best = val
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if rank == 0:
                print(
                    f"{name} ep={ep:03d} train_mse={tsum / max(tn, 1):.4g} "
                    f"val_rel={val:.4g}",
                    flush=True,
                )
        if best_state is not None:
            model.load_state_dict(best_state)
        test_one_step = eval_one_step(
            model, test_dl, device, args.eval_batches, distributed
        )
        rollout = eval_rollout(
            model, roll_dl, device, args.history, args.use_phys, args.eval_batches, distributed
        )
        test_rollout = eval_rollout(
            model,
            test_roll_dl,
            device,
            args.history,
            args.use_phys,
            args.eval_batches,
            distributed,
        )
        row = {
            "model": name,
            "best_one_step_rel": best,
            "test_one_step_rel": test_one_step,
            "rollout_rel": rollout,
            "test_rollout_rel": test_rollout,
            "epochs": args.epochs,
            "history": args.history,
            "seconds": time.time() - t_start,
        }
        if rank == 0:
            rows.append(row)
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "row": row},
                out_csv.parent / f"{name}_latent_operator.pt",
            )
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()
    if rank == 0:
        print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
