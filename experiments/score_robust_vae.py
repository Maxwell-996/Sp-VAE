#!/usr/bin/env python3
"""Run the fixed, side-effect-free micro benchmark for VAE experiments.

The benchmark trains a small model on a fixed NS subset, then evaluates both
clean reconstruction and reconstruction after injecting out-of-band noise.
All checkpoints and logs are placed in a temporary directory outside the repo.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hf_vae import build_from_ckpt
from hfvae.frequency import highpass_filter_2d
from hfvae.ns_data import NSShardDataset, ShardContiguousSampler, resize_video


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_variant(args: argparse.Namespace, out_dir: Path) -> Path:
    train_args = [
        "experiments/train_hf_vae.py",
        "--data-root",
        str(args.data_root),
        "--out-dir",
        str(out_dir),
        "--tag",
        "score",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--dim",
        str(args.dim),
        "--z-dim",
        str(args.z_dim),
        "--t-clip",
        str(args.t_clip),
        "--out-t",
        str(args.out_t),
        "--out-hw",
        str(args.out_hw),
        "--lazy",
        "--max-shards",
        str(args.max_shards),
    ]
    # Keep the benchmark small and deterministic; the explicit values are part
    # of the measurement contract and should not be changed during a run.
    train_args.extend(
        [
            "--max-samples-per-shard",
            str(args.train_samples),
            "--val-batches",
            str(args.valid_samples),
            "--beta-kl",
            "0.0001",
            "--transition",
            "0.04",
        ]
    )
    if args.variant == "hf":
        train_args.extend(
            [
                "--anti-alias",
                "--latent-hfs",
                "--output-hfs",
                "--decoder-spectral-filter",
                "--filter-cutoff",
                "0.15",
                "--hfs-cutoff",
                "0.15",
                "--lambda-lowpass",
                "0.5",
                "--lambda-hf",
                "0.10",
                "--lambda-latent-consistency",
                "0.05",
            "--lambda-noisy-reconstruction",
            "0.05",
            "--lambda-wavelet",
            str(args.lambda_wavelet),
            "--wavelet-levels",
            str(args.wavelet_levels),
            "--lowpass-cutoff",
            "0.10",
                "--highpass-cutoff",
                "0.15",
                "--consistency-noise",
                "0.10",
            ]
        )
    else:
        train_args.extend(
            [
                "--lambda-lowpass",
                "0",
                "--lambda-hf",
                "0",
                "--lambda-spectral",
                "0",
                "--lambda-sobolev",
                "0",
                "--lambda-latent-consistency",
                "0",
                "--lambda-noisy-reconstruction",
                "0",
            ]
        )

    previous_argv = sys.argv
    try:
        seed_all(args.seed)
        sys.argv = train_args
        runpy.run_path(
            str(PROJECT_ROOT / "experiments" / "train_hf_vae.py"),
            run_name="__main__",
        )
    finally:
        sys.argv = previous_argv
    return out_dir / "score_best.pt"


def per_sample_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, target.ndim))
    denom = target.square().mean(dim=dims).sqrt().clamp_min(1e-8)
    return (pred - target).square().mean(dim=dims).sqrt() / denom


def per_sample_rms_ratio(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, value.ndim))
    denom = reference.square().mean(dim=dims).sqrt().clamp_min(1e-8)
    return value.square().mean(dim=dims).sqrt() / denom


@torch.no_grad()
def evaluate(args: argparse.Namespace, checkpoint: Path) -> dict[str, float]:
    seed_all(args.seed + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = build_from_ckpt(str(checkpoint), device)
    dataset = NSShardDataset(
        args.data_root,
        "valid",
        t_clip=args.t_clip,
        norm_stats=(ck["norm_mean"], ck["norm_std"]),
        in_memory=False,
        max_shards=args.max_shards,
        max_samples_per_shard=args.valid_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=ShardContiguousSampler(dataset, shuffle=False),
        num_workers=0,
    )

    clean_values: list[float] = []
    noisy_values: list[float] = []
    leakage_values: list[float] = []
    latent_values: list[float] = []
    for x in loader:
        x = resize_video(x.to(device), args.out_hw, args.out_t)
        z = model.encode_dist(x, [0, 1])[0]
        clean = model.decode(z, [0, 1])

        noise = highpass_filter_2d(
            torch.randn_like(x), cutoff=args.noise_cutoff, transition=args.transition
        )
        x_rms = x.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
        noise_rms = noise.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
        noisy_input = x + args.noise_amplitude * noise * x_rms / (noise_rms + 1e-8)
        z_noisy = model.encode_dist(noisy_input, [0, 1])[0]
        noisy = model.decode(z_noisy, [0, 1])

        clean_values.extend(per_sample_rel_l2(clean, x).tolist())
        noisy_values.extend(per_sample_rel_l2(noisy, x).tolist())
        leakage = highpass_filter_2d(
            noisy - clean, cutoff=args.noise_cutoff, transition=args.transition
        )
        leakage_values.extend(per_sample_rms_ratio(leakage, x).tolist())
        latent_values.extend(per_sample_rel_l2(z_noisy, z).tolist())

    clean = sum(clean_values) / max(len(clean_values), 1)
    noisy = sum(noisy_values) / max(len(noisy_values), 1)
    result = {
        "clean_rel_l2": clean,
        "noisy_to_clean_rel_l2": noisy,
        "hfs_leakage": sum(leakage_values) / max(len(leakage_values), 1),
        "latent_shift": sum(latent_values) / max(len(latent_values), 1),
        "robust_val_rel_l2": 0.5 * (clean + noisy),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"non-finite benchmark result: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/media/home/lizhonghao/diff_pde/data")
    parser.add_argument("--variant", choices=["vanilla", "hf"], default="hf")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--t-clip", type=int, default=5)
    parser.add_argument("--out-t", type=int, default=5)
    parser.add_argument("--out-hw", type=int, default=32)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--valid-samples", type=int, default=4)
    parser.add_argument("--noise-cutoff", type=float, default=0.20)
    parser.add_argument("--noise-amplitude", type=float, default=0.10)
    parser.add_argument("--transition", type=float, default=0.04)
    parser.add_argument("--lambda-wavelet", type=float, default=0.5)
    parser.add_argument("--wavelet-levels", type=int, default=2)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hfvae-score-") as temporary:
        checkpoint = train_variant(args, Path(temporary))
        result = evaluate(args, checkpoint)
    result["variant"] = args.variant
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
