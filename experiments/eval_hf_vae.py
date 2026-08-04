#!/usr/bin/env python3
"""Evaluate VAE reconstruction and frequency-band errors on NS shards."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfvae.frequency import highpass_filter_2d, lowpass_filter_2d
from hfvae.ns_data import NSShardDataset, ShardContiguousSampler, resolve_data_root, resize_video
from hfvae.wan22_hf_vae import WanVAEHF


def build_from_ckpt(path: str, device: torch.device) -> tuple[WanVAEHF, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = WanVAEHF(
        dim=a.get("dim", 96),
        dec_dim=a.get("dim", 96),
        z_dim=a.get("z_dim", 16),
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[True, True, False],
        patch_size=a.get("patch_size", 2),
        anti_alias=a.get("anti_alias", False),
        latent_hfs=a.get("latent_hfs", False),
        output_hfs=a.get("output_hfs", False),
        decoder_spectral_filter=a.get("decoder_spectral_filter", False),
        filter_cutoff=a.get("filter_cutoff", None),
        decoder_filter_min_gain=a.get("decoder_filter_min_gain", 0.02),
        hfs_cutoff=a.get("hfs_cutoff", 0.35),
        latent_band_moe=a.get("latent_band_moe", False),
        moe_hidden=a.get("moe_hidden", 32),
        moe_low_cutoff=a.get("moe_low_cutoff", 0.20),
        moe_high_cutoff=a.get("moe_high_cutoff", 0.50),
        moe_global_low=a.get("moe_global_low", False),
        moe_decomposition=a.get("moe_decomposition", "fft"),
        moe_cross_band=a.get("moe_cross_band", False),
        moe_shared_expert=a.get("moe_shared_expert", False),
        moe_router_smooth=a.get("moe_router_smooth", False),
        latent_learnable_moe=a.get("latent_learnable_moe", False),
        output_learnable_moe=a.get("output_learnable_moe", False),
        learnable_moe_hidden=a.get("learnable_moe_hidden", 64),
        learnable_moe_experts=a.get("learnable_moe_experts", 4),
        learnable_moe_top_k=a.get("learnable_moe_top_k", 2),
        learnable_moe_global=a.get("learnable_moe_global", False),
        learnable_moe_routing=a.get("learnable_moe_routing", "token_choice"),
        learnable_moe_shared_experts=a.get("learnable_moe_shared_experts", 1),
        learnable_moe_router_noise=a.get("learnable_moe_router_noise", 0.0),
        output_band_moe=a.get("output_band_moe", False),
    ).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    allowed_missing = {
        "latent_learnable_moe.last_importance",
        "latent_learnable_moe.last_load",
        "output_learnable_moe.last_importance",
        "output_learnable_moe.last_load",
    }
    unexpected = set(unexpected)
    missing = set(missing)
    if unexpected or missing - allowed_missing:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    model.eval()
    return model, ck


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-8))


@torch.no_grad()
def evaluate(args) -> list[dict[str, float | int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = build_from_ckpt(args.ckpt, device)
    data_root = resolve_data_root(args.data_root)
    ds = NSShardDataset(
        data_root,
        args.split,
        t_clip=args.t_clip,
        norm_stats=(ck.get("norm_mean", [0.0, 0.0, 0.0]), ck["norm_std"]),
        in_memory=False,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=ShardContiguousSampler(ds, shuffle=False),
        num_workers=0,
        pin_memory=True,
    )
    rows = []
    for bi, x in enumerate(dl):
        x = resize_video(x.to(device), args.out_hw, args.out_t)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            mu, _ = model.encode_dist(x, [0, 1])
            rec = model.decode(mu, [0, 1])
        low_x = lowpass_filter_2d(x, args.low_cutoff, args.transition)
        low_r = lowpass_filter_2d(rec, args.low_cutoff, args.transition)
        high_x = highpass_filter_2d(x, args.high_cutoff, args.transition)
        high_r = highpass_filter_2d(rec, args.high_cutoff, args.transition)
        rows.append(
            {
                "batch": bi,
                "n": x.size(0),
                "rel_l2": rel_l2(rec, x),
                "low_rel_l2": rel_l2(low_r, low_x),
                "high_rel_l2": rel_l2(high_r, high_x),
                "pred_hf_energy": float(high_r.square().mean()),
                "target_hf_energy": float(high_x.square().mean()),
            }
        )
        print(rows[-1], flush=True)
        if args.max_batches and bi + 1 >= args.max_batches:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--t-clip", type=int, default=49)
    ap.add_argument("--out-t", type=int, default=None)
    ap.add_argument("--out-hw", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--low-cutoff", type=float, default=0.25)
    ap.add_argument("--high-cutoff", type=float, default=0.35)
    ap.add_argument("--transition", type=float, default=0.08)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    rows = evaluate(args)
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
