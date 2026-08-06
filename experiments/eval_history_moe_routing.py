#!/usr/bin/env python3
"""Report learned temporal-lag routing statistics for a history MoE operator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_latent_operator import (
    HistoryMixtureOperator,
    HistoryResidualOperator,
    LatentTrajectoryDataset,
    add_act_root,
    build_model,
)


def history_mixture_from(model: torch.nn.Module) -> HistoryMixtureOperator:
    while isinstance(model, HistoryResidualOperator):
        model = model.base
    if not isinstance(model, HistoryMixtureOperator):
        raise ValueError("operator checkpoint does not use --operator-history-moe")
    return model


def regime_key(physical: torch.Tensor) -> str:
    return ",".join(f"{float(value):.4g}" for value in physical.tolist())


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    add_act_root(args.act_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.operator_ckpt, map_location="cpu", weights_only=False)
    op_args = SimpleNamespace(**checkpoint["args"])
    if not getattr(op_args, "operator_history_moe", False):
        raise ValueError("operator checkpoint was not trained with --operator-history-moe")
    latent_root = Path(args.latent_root).resolve()
    norm = torch.load(
        latent_root / "norm_stats_latent.pt", map_location="cpu", weights_only=False
    )
    dataset = LatentTrajectoryDataset(
        latent_root,
        args.split,
        history=op_args.history,
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        rollout_horizon=op_args.train_rollout_horizon,
        use_phys=op_args.use_phys,
    )
    first_x, first_y = dataset[0]
    dim_out = first_y.shape[0] if first_y.ndim == 3 else first_y.shape[1]
    model = build_model(
        args.model,
        first_x.shape[0],
        dim_out,
        first_x.shape[-2],
        first_x.shape[-1],
        op_args,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    mixture = history_mixture_from(model)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    total_weight = torch.zeros(op_args.history, dtype=torch.float64)
    total_square = torch.zeros(op_args.history, dtype=torch.float64)
    total_entropy = 0.0
    total_count = 0
    top_counts = torch.zeros(op_args.history, dtype=torch.int64)
    by_regime: dict[str, dict[str, torch.Tensor | float | int]] = defaultdict(
        lambda: {
            "weight": torch.zeros(op_args.history, dtype=torch.float64),
            "count": 0,
        }
    )
    physical_start = op_args.history * dim_out
    for inputs, _targets in loader:
        inputs = inputs.to(device)
        weights = torch.softmax(mixture.router(inputs.mean(dim=(-2, -1))), dim=-1)
        cpu_weights = weights.cpu().double()
        total_weight += cpu_weights.sum(dim=0)
        total_square += cpu_weights.square().sum(dim=0)
        total_entropy += float(
            (-(cpu_weights * cpu_weights.clamp_min(1e-12).log()).sum(dim=1)).sum()
        )
        total_count += cpu_weights.shape[0]
        top_counts += torch.bincount(cpu_weights.argmax(dim=1), minlength=op_args.history)
        if op_args.use_phys:
            physical = inputs[:, physical_start:].mean(dim=(-2, -1)).cpu()
            for row, weights_row in zip(physical, cpu_weights):
                group = by_regime[regime_key(row)]
                group["weight"] += weights_row
                group["count"] += 1

    mean = total_weight / max(total_count, 1)
    variance = (total_square / max(total_count, 1) - mean.square()).clamp_min(0)
    result = {
        "model": args.model,
        "operator_ckpt": str(Path(args.operator_ckpt).resolve()),
        "split": args.split,
        "samples": total_count,
        "lag_order": [f"t-{op_args.history - index}" for index in range(op_args.history)],
        "mean_weight": mean.tolist(),
        "std_weight": variance.sqrt().tolist(),
        "top_lag_fraction": (top_counts.double() / max(total_count, 1)).tolist(),
        "mean_entropy": total_entropy / max(total_count, 1),
        "mean_effective_lags": float(torch.exp(torch.tensor(total_entropy / max(total_count, 1)))),
        "regime_mean_weight": {
            key: (value["weight"] / value["count"]).tolist()
            for key, value in sorted(by_regime.items())
            if value["count"]
        },
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-root", default="runs/latents_hf_ddp_global8")
    ap.add_argument("--operator-ckpt", required=True)
    ap.add_argument("--act-root", default="/media/home/lizhonghao/act/ACT-NeuralOP")
    ap.add_argument("--model", choices=["fno", "transolver", "cnextu"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = evaluate(args)
    print(json.dumps(result, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
