#!/usr/bin/env python3
"""Encode read-only NS shards into local VAE latent caches."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfvae.ns_data import resolve_data_root, resize_video
from experiments.eval_hf_vae import build_from_ckpt


@torch.no_grad()
def encode_video_batch(
    model,
    uvp,
    mean,
    std,
    device,
    batch: int,
    *,
    t_clip: int | None = None,
    out_t: int | None = None,
    out_hw: int | None = None,
    amp: bool = False,
):
    # uvp: (N,3,H,W,T) raw physical units.
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    zs = []
    for i in range(0, uvp.shape[0], batch):
        x = uvp[i:i + batch].to(device).float()
        if t_clip is not None:
            x = x[..., : int(t_clip)]
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = (x - mean_t) / (std_t + 1e-8)
        x = resize_video(x, out_hw, out_t)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            mu, _ = model.encode_dist(x, [0, 1])
        zs.append(mu.float().cpu())
    return torch.cat(zs, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-root", default="runs/latents")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--t-clip", type=int, default=None)
    ap.add_argument("--out-t", type=int, default=None)
    ap.add_argument("--out-hw", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun multi-GPU caching requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=local_rank)
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model, ck = build_from_ckpt(args.ckpt, device)
    mean = ck.get("norm_mean", [0.0, 0.0, 0.0])
    std = ck["norm_std"]
    data_root = resolve_data_root(args.data_root)
    split_dir = data_root / args.split
    src_manifest = torch.load(split_dir / "manifest.pt", map_location="cpu", weights_only=False)
    out_dir = Path(args.out_root).resolve() / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    csum = torch.zeros(model.z_dim, dtype=torch.float64)
    csq = torch.zeros(model.z_dim, dtype=torch.float64)
    count = 0
    manifest = []
    shards = list(src_manifest["shards"])
    if args.max_shards is not None:
        shards = shards[:args.max_shards]
    if distributed:
        shards = shards[rank::world_size]
    for sh in shards:
        src = torch.load(split_dir / sh["path"], map_location="cpu", weights_only=False)
        uvp = src["uvp"].float()
        if args.max_samples_per_shard is not None:
            uvp = uvp[: int(args.max_samples_per_shard)]
        z = encode_video_batch(
            model,
            uvp,
            mean,
            std,
            device,
            args.batch,
            t_clip=args.t_clip,
            out_t=args.out_t,
            out_hw=args.out_hw,
            amp=args.amp,
        )
        out_path = out_dir / sh["path"]
        torch.save(
            {
                "z": z,
                "labels": src.get("labels", {}),
                "regime": src.get("regime", {}),
                "t": src.get("t", None),
                "source": str(split_dir / sh["path"]),
            },
            out_path,
        )
        for c in range(z.shape[1]):
            ch = z[:, c].double()
            csum[c] += ch.sum()
            csq[c] += ch.square().sum()
        count += z.shape[0] * z.shape[2] * z.shape[3] * z.shape[4]
        manifest.append({"path": sh["path"], "n": z.shape[0], **src.get("regime", {})})
        print(f"rank={rank} {args.split}/{sh['path']}: z={tuple(z.shape)}", flush=True)

    if distributed:
        stats = torch.cat(
            [csum, csq, torch.tensor([float(count)], dtype=torch.float64)]
        ).to(device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        csum = stats[: model.z_dim].cpu()
        csq = stats[model.z_dim:2 * model.z_dim].cpu()
        count = int(stats[-1].item())
        torch.save(manifest, out_dir / f"manifest.rank{rank}.pt")
        dist.barrier()
        if rank == 0:
            merged = []
            for r in range(world_size):
                merged.extend(
                    torch.load(out_dir / f"manifest.rank{r}.pt", weights_only=False)
                )
            manifest = sorted(merged, key=lambda item: item["path"])
    if not distributed or rank == 0:
        torch.save(
            {
                "split": args.split,
                "shards": manifest,
                "total": sum(m["n"] for m in manifest),
                "vae_ckpt": str(Path(args.ckpt).resolve()),
                "z_dim": model.z_dim,
            },
            out_dir / "manifest.pt",
        )
    if args.split == "train" and (not distributed or rank == 0):
        mean_z = csum / max(count, 1)
        std_z = (csq / max(count, 1) - mean_z.square()).clamp_min(0).sqrt()
        torch.save(
            {"mean": mean_z.tolist(), "std": std_z.tolist(), "z_dim": model.z_dim},
            Path(args.out_root).resolve() / "norm_stats_latent.pt",
        )
    if distributed:
        dist.barrier()
        if rank == 0:
            for r in range(world_size):
                (out_dir / f"manifest.rank{r}.pt").unlink()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
