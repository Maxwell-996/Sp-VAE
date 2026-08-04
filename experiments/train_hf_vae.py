#!/usr/bin/env python3
"""Train a high-frequency robust Wan2.2 VAE on read-only NS shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfvae.losses import HFVAELossConfig, compute_vae_loss
from hfvae.ns_data import NSShardDataset, ShardContiguousSampler, resolve_data_root, resize_video
from hfvae.wan22_hf_vae import WanVAEHF


class _LossModule(torch.nn.Module):
    """Expose the VAE loss through forward so DDP can synchronize gradients."""

    def __init__(self, model: WanVAEHF, loss_cfg: HFVAELossConfig, noise: float):
        super().__init__()
        self.model = model
        self.loss_cfg = loss_cfg
        self.noise = noise

    def forward(self, x: torch.Tensor):
        return compute_vae_loss(
            self.model,
            x,
            self.loss_cfg,
            latent_consistency_noise=self.noise,
        )


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
        decoder_filter_min_gain=args.decoder_filter_min_gain,
        hfs_cutoff=args.hfs_cutoff,
        latent_band_moe=args.latent_band_moe,
        moe_hidden=args.moe_hidden,
        moe_low_cutoff=args.moe_low_cutoff,
        moe_high_cutoff=args.moe_high_cutoff,
        moe_global_low=args.moe_global_low,
        moe_decomposition=args.moe_decomposition,
        moe_cross_band=args.moe_cross_band,
        moe_shared_expert=args.moe_shared_expert,
        moe_router_smooth=args.moe_router_smooth,
        latent_learnable_moe=args.latent_learnable_moe,
        output_learnable_moe=args.output_learnable_moe,
        learnable_moe_hidden=args.learnable_moe_hidden,
        learnable_moe_experts=args.learnable_moe_experts,
        learnable_moe_top_k=args.learnable_moe_top_k,
        learnable_moe_global=args.learnable_moe_global,
        learnable_moe_routing=args.learnable_moe_routing,
        learnable_moe_shared_experts=args.learnable_moe_shared_experts,
        learnable_moe_router_noise=args.learnable_moe_router_noise,
        output_band_moe=args.output_band_moe,
    )


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean().sqrt() / (target.square().mean().sqrt() + 1e-8)


@torch.no_grad()
def validate(model, loader, args, device, distributed: bool) -> float:
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
    if distributed:
        totals = torch.tensor([total, count], device=device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total, count = totals.tolist()
    return total / max(count, 1.0)


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
    ap.add_argument("--decoder-filter-min-gain", type=float, default=0.02)
    ap.add_argument("--hfs-cutoff", type=float, default=0.35)
    ap.add_argument("--latent-band-moe", action="store_true")
    ap.add_argument("--moe-hidden", type=int, default=32)
    ap.add_argument("--moe-low-cutoff", type=float, default=0.20)
    ap.add_argument("--moe-high-cutoff", type=float, default=0.50)
    ap.add_argument("--moe-global-low", action="store_true")
    ap.add_argument("--moe-decomposition", choices=["fft", "local"], default="fft")
    ap.add_argument("--moe-cross-band", action="store_true")
    ap.add_argument("--moe-shared-expert", action="store_true")
    ap.add_argument("--moe-router-smooth", action="store_true")
    ap.add_argument("--latent-learnable-moe", action="store_true")
    ap.add_argument("--output-learnable-moe", action="store_true")
    ap.add_argument("--learnable-moe-hidden", type=int, default=64)
    ap.add_argument("--learnable-moe-experts", type=int, default=4)
    ap.add_argument("--learnable-moe-top-k", type=int, default=2)
    ap.add_argument("--learnable-moe-global", action="store_true")
    ap.add_argument(
        "--learnable-moe-routing",
        choices=["token_choice", "expert_choice"],
        default="token_choice",
    )
    ap.add_argument("--learnable-moe-shared-experts", type=int, default=1)
    ap.add_argument("--learnable-moe-router-noise", type=float, default=0.0)
    ap.add_argument("--output-band-moe", action="store_true")

    # Objective knobs.
    ap.add_argument("--beta-kl", type=float, default=1e-4)
    ap.add_argument("--lambda-lowpass", type=float, default=0.5)
    ap.add_argument("--lambda-hf", type=float, default=0.05)
    ap.add_argument("--lambda-spectral", type=float, default=0.0)
    ap.add_argument("--lambda-wavelet", type=float, default=0.0)
    ap.add_argument("--lambda-sobolev", type=float, default=0.05)
    ap.add_argument("--lambda-latent-consistency", type=float, default=0.05)
    ap.add_argument("--lambda-noisy-reconstruction", type=float, default=0.0)
    ap.add_argument("--lambda-temporal", type=float, default=0.0)
    ap.add_argument("--lambda-vorticity", type=float, default=0.0)
    ap.add_argument("--lambda-moe-balance", type=float, default=0.0)
    ap.add_argument("--wavelet-levels", type=int, default=2)
    ap.add_argument("--lowpass-cutoff", type=float, default=0.35)
    ap.add_argument("--highpass-cutoff", type=float, default=0.35)
    ap.add_argument("--transition", type=float, default=0.08)
    ap.add_argument("--sobolev-s", type=float, default=1.0)
    ap.add_argument("--consistency-noise", type=float, default=0.05)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-GPU training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)

    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
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
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else (ShardContiguousSampler(train_ds, shuffle=True) if args.lazy else None)
    )
    valid_sampler = (
        DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else ShardContiguousSampler(valid_ds, shuffle=False)
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=train_sampler,
        shuffle=(not args.lazy and not distributed),
        num_workers=0 if args.lazy else 4,
        drop_last=True,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch,
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(args).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if rank == 0:
            print(
                f"initialized from {args.init_from}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    loss_cfg = HFVAELossConfig(
        beta_kl=args.beta_kl,
        lambda_lowpass=args.lambda_lowpass,
        lambda_hf=args.lambda_hf,
        lambda_spectral=args.lambda_spectral,
        lambda_wavelet=args.lambda_wavelet,
        lambda_sobolev=args.lambda_sobolev,
        lambda_latent_consistency=args.lambda_latent_consistency,
        lambda_noisy_reconstruction=args.lambda_noisy_reconstruction,
        lambda_temporal=args.lambda_temporal,
        lambda_vorticity=args.lambda_vorticity,
        lambda_moe_balance=args.lambda_moe_balance,
        wavelet_levels=args.wavelet_levels,
        lowpass_cutoff=args.lowpass_cutoff,
        highpass_cutoff=args.highpass_cutoff,
        transition=args.transition,
        sobolev_s=args.sobolev_s,
    )

    train_module: torch.nn.Module = _LossModule(model, loss_cfg, args.consistency_noise)
    if distributed:
        train_module = DDP(
            train_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    opt = torch.optim.AdamW(train_module.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    meta = {
        "args": vars(args),
        "data_root": str(data_root),
        "norm_mean": train_ds.mean,
        "norm_std": train_ds.std,
    }
    if rank == 0:
        (out_dir / f"{args.tag}_config.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    best = float("inf")
    if rank == 0:
        print(
            f"data={data_root} train={len(train_ds)} valid={len(valid_ds)} "
            f"device={device} world_size={world_size}"
        )
        print(f"model params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    for ep in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(ep)
        model.train()
        t0 = time.time()
        sums: dict[str, float] = {}
        n = 0
        for x in train_dl:
            x = resize_video(x.to(device, non_blocking=True), args.out_hw, args.out_t)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, parts = train_module(x)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * x.size(0)
            n += x.size(0)
        if distributed:
            total_n = torch.tensor(float(n), device=device, dtype=torch.float64)
            dist.all_reduce(total_n, op=dist.ReduceOp.SUM)
            n = int(total_n.item())
            for key in sums:
                value = torch.tensor(sums[key], device=device, dtype=torch.float64)
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                sums[key] = float(value.item())
        sched.step()
        val = validate(model, valid_dl, args, device, distributed)
        msg = " ".join(f"{k}={sums[k] / max(n, 1):.4g}" for k in sorted(sums))
        if rank == 0:
            print(
                f"ep {ep:03d} {msg} val_rel_l2={val:.5f} "
                f"time={time.time() - t0:.1f}s",
                flush=True,
            )
        if rank == 0 and val < best:
            best = val
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "norm_mean": train_ds.mean,
                "norm_std": train_ds.std,
                "val": best,
            }
            torch.save(ckpt, out_dir / f"{args.tag}_best.pt")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(f"done best_val={best:.5f} ckpt={out_dir / f'{args.tag}_best.pt'}")


if __name__ == "__main__":
    main()
