#!/usr/bin/env python3
"""Train a high-frequency robust Wan2.2 VAE on read-only NS shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfvae.losses import HFVAELossConfig, compute_vae_loss
from hfvae.ns_data import NSShardDataset, ShardContiguousSampler, resolve_data_root, resize_video
from hfvae.wan22_hf_vae import WanVAEHF


def build_model(args) -> WanVAEHF:
    return WanVAEHF(
        dim=args.dim,
        dec_dim=args.dim,
        z_dim=args.z_dim,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[True, True, False],
        patch_size=args.patch_size,
        anti_alias=args.anti_alias,
        latent_hfs=args.latent_hfs,
        output_hfs=args.output_hfs,
        decoder_spectral_filter=args.decoder_spectral_filter,
        filter_cutoff=args.filter_cutoff,
        hfs_cutoff=args.hfs_cutoff,
    )


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean().sqrt() / (target.square().mean().sqrt() + 1e-8)


@torch.no_grad()
def validate(model, loader, args, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x in loader:
        x = resize_video(x.to(device, non_blocking=True), args.out_hw, args.out_t)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            mu, _ = model.encode_dist(x, [0, 1])
            recon = model.decode(mu, [0, 1])
            score = rel_l2(recon, x)
        total += float(score) * x.size(0)
        count += x.size(0)
        if args.val_batches and count >= args.val_batches * args.batch:
            break
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None, help="Read-only root with train/valid shards.")
    ap.add_argument("--out-dir", default="runs/hf_vae", help="Local output directory in this workspace.")
    ap.add_argument("--tag", default="hfvae")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--z-dim", type=int, default=16)
    ap.add_argument("--patch-size", type=int, default=2, choices=[1, 2])
    ap.add_argument("--t-clip", type=int, default=49)
    ap.add_argument("--out-t", type=int, default=None)
    ap.add_argument("--out-hw", type=int, default=None)
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--init-from",
        default=None,
        help="Partially initialize from a compatible Wan VAE checkpoint.",
    )

    # Architecture knobs.
    ap.add_argument("--anti-alias", action="store_true")
    ap.add_argument("--latent-hfs", action="store_true")
    ap.add_argument("--output-hfs", action="store_true")
    ap.add_argument("--decoder-spectral-filter", action="store_true")
    ap.add_argument("--filter-cutoff", type=float, default=None)
    ap.add_argument("--hfs-cutoff", type=float, default=0.35)

    # Objective knobs.
    ap.add_argument("--beta-kl", type=float, default=1e-4)
    ap.add_argument("--lambda-lowpass", type=float, default=0.5)
    ap.add_argument("--lambda-hf", type=float, default=0.05)
    ap.add_argument("--lambda-spectral", type=float, default=0.0)
    ap.add_argument("--lambda-sobolev", type=float, default=0.05)
    ap.add_argument("--lambda-latent-consistency", type=float, default=0.05)
    ap.add_argument("--lambda-noisy-reconstruction", type=float, default=0.0)
    ap.add_argument("--lambda-temporal", type=float, default=0.0)
    ap.add_argument("--lowpass-cutoff", type=float, default=0.35)
    ap.add_argument("--highpass-cutoff", type=float, default=0.35)
    ap.add_argument("--transition", type=float, default=0.08)
    ap.add_argument("--sobolev-s", type=float, default=1.0)
    ap.add_argument("--consistency-noise", type=float, default=0.05)
    args = ap.parse_args()

    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NSShardDataset(
        data_root,
        "train",
        t_clip=args.t_clip,
        in_memory=not args.lazy,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
    )
    valid_ds = NSShardDataset(
        data_root,
        "valid",
        t_clip=args.t_clip,
        norm_stats=(train_ds.mean, train_ds.std),
        in_memory=False,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=ShardContiguousSampler(train_ds, shuffle=True) if args.lazy else None,
        shuffle=not args.lazy,
        num_workers=0 if args.lazy else 4,
        drop_last=True,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch,
        sampler=ShardContiguousSampler(valid_ds, shuffle=False),
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(args).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(
            f"initialized from {args.init_from}: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_cfg = HFVAELossConfig(
        beta_kl=args.beta_kl,
        lambda_lowpass=args.lambda_lowpass,
        lambda_hf=args.lambda_hf,
        lambda_spectral=args.lambda_spectral,
        lambda_sobolev=args.lambda_sobolev,
        lambda_latent_consistency=args.lambda_latent_consistency,
        lambda_noisy_reconstruction=args.lambda_noisy_reconstruction,
        lambda_temporal=args.lambda_temporal,
        lowpass_cutoff=args.lowpass_cutoff,
        highpass_cutoff=args.highpass_cutoff,
        transition=args.transition,
        sobolev_s=args.sobolev_s,
    )

    meta = {
        "args": vars(args),
        "data_root": str(data_root),
        "norm_mean": train_ds.mean,
        "norm_std": train_ds.std,
    }
    (out_dir / f"{args.tag}_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best = float("inf")
    print(f"data={data_root} train={len(train_ds)} valid={len(valid_ds)} device={device}")
    print(f"model params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        n = 0
        for x in train_dl:
            x = resize_video(x.to(device, non_blocking=True), args.out_hw, args.out_t)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, parts = compute_vae_loss(
                    model,
                    x,
                    loss_cfg,
                    latent_consistency_noise=args.consistency_noise,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * x.size(0)
            n += x.size(0)
        sched.step()
        val = validate(model, valid_dl, args, device)
        msg = " ".join(f"{k}={sums[k] / max(n, 1):.4g}" for k in sorted(sums))
        print(f"ep {ep:03d} {msg} val_rel_l2={val:.5f} time={time.time() - t0:.1f}s", flush=True)
        if val < best:
            best = val
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                "val": best,
            }
            torch.save(ckpt, out_dir / f"{args.tag}_best.pt")
    print(f"done best_val={best:.5f} ckpt={out_dir / f'{args.tag}_best.pt'}")


if __name__ == "__main__":
    main()
