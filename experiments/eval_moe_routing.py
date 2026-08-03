#!/usr/bin/env python3
"""Inspect learned expert usage on a VAE checkpoint's valid split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hf_vae import build_from_ckpt
from hfvae.ns_data import NSShardDataset, ShardContiguousSampler, resolve_data_root, resize_video


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = build_from_ckpt(args.ckpt, device)
    moe = model.latent_learnable_moe
    if moe is None:
        raise ValueError("checkpoint does not enable latent_learnable_moe")
    data_root = resolve_data_root(args.data_root)
    ds = NSShardDataset(
        data_root,
        args.split,
        t_clip=49,
        norm_stats=(ck.get("norm_mean", [0.0, 0.0, 0.0]), ck["norm_std"]),
        in_memory=False,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=ShardContiguousSampler(ds, shuffle=False),
        num_workers=0,
    )
    importance = torch.zeros(moe.num_experts, dtype=torch.float64)
    load = torch.zeros_like(importance)
    count = 0
    model.eval()
    for bi, x in enumerate(dl):
        x = resize_video(x.to(device), 128, 49)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            model.encode_dist(x, [0, 1])
        stats = moe.routing_stats()
        importance += torch.tensor(stats["importance"], dtype=torch.float64)
        load += torch.tensor(stats["load"], dtype=torch.float64)
        count += 1
        if args.max_batches and bi + 1 >= args.max_batches:
            break
    result = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "split": args.split,
        "batches": count,
        "importance": (importance / max(count, 1)).tolist(),
        "load": (load / max(count, 1)).tolist(),
        "temperature": float(moe.log_temperature.exp().detach().cpu()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
