#!/usr/bin/env python3
"""Summarize learned routing statistics for a latent operator checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_latent_operator import (
    LatentTrajectoryDataset,
    add_act_root,
    build_model,
)
from hfvae.learnable_moe import LearnableTokenMoE


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.operator_ckpt, map_location="cpu", weights_only=False)
    op_args = SimpleNamespace(**checkpoint["args"])
    add_act_root(getattr(op_args, "act_root", args.act_root))

    root = Path(args.latent_root).resolve()
    norm = torch.load(root / "norm_stats_latent.pt", map_location="cpu", weights_only=False)
    dataset = LatentTrajectoryDataset(
        root,
        args.split,
        history=int(op_args.history),
        norm_stats=norm,
        max_shards=args.max_shards,
        max_samples_per_shard=args.max_samples_per_shard,
        use_phys=bool(op_args.use_phys),
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    first_x, first_y = dataset[0]
    model = build_model(
        args.model,
        first_x.shape[0],
        first_y.shape[0],
        first_x.shape[-2],
        first_x.shape[-1],
        op_args,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LearnableTokenMoE)
    ]
    if not modules:
        raise ValueError("operator checkpoint has no LearnableTokenMoE modules")
    totals = {
        name: {
            "importance": torch.zeros(module.num_experts, dtype=torch.float64),
            "load": torch.zeros(module.num_experts, dtype=torch.float64),
            "temperature": 0.0,
            "global_gate": 0.0,
            "local_scales": torch.zeros(
                3 if module.local_context_enabled else 0, dtype=torch.float64
            ),
            "local_mix": torch.zeros(
                3 if module.local_scale_mixer is not None else 0, dtype=torch.float64
            ),
            "local_mix_std": torch.zeros(
                3 if module.local_scale_mixer is not None else 0, dtype=torch.float64
            ),
            "expert_local_mix": torch.zeros(
                (module.num_experts, 3)
                if module.expert_local_mix_logits is not None
                else (0, 3),
                dtype=torch.float64,
            ),
            "kernel_scales": torch.zeros(module.num_experts, dtype=torch.float64),
            "kernel_scale_std": torch.zeros(module.num_experts, dtype=torch.float64),
            "kernel_scale_min": torch.zeros(module.num_experts, dtype=torch.float64),
            "kernel_scale_max": torch.zeros(module.num_experts, dtype=torch.float64),
            "dynamic_top_k_probability": torch.zeros(
                len(module.dynamic_top_k_values), dtype=torch.float64
            ),
            "dynamic_top_k_mean": 0.0,
            "dynamic_top_k_std": 0.0,
            "global_routing": torch.zeros(module.num_experts, dtype=torch.float64),
            "global_routing_std": torch.zeros(module.num_experts, dtype=torch.float64),
        }
        for name, module in modules
    }
    batches = 0
    for batch_index, (x, _y) in enumerate(loader):
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            model(x)
        for name, module in modules:
            stats = module.routing_stats()
            totals[name]["importance"] += torch.tensor(
                stats["importance"], dtype=torch.float64
            )
            totals[name]["load"] += torch.tensor(stats["load"], dtype=torch.float64)
            totals[name]["temperature"] += float(stats["temperature"][0])
            totals[name]["global_gate"] += float(stats["global_gate"][0])
            local_scales = stats.get("local_scales", [])
            if local_scales:
                totals[name]["local_scales"] += torch.tensor(
                    local_scales, dtype=torch.float64
                )
            local_mix = stats.get("local_mix", [])
            if local_mix:
                totals[name]["local_mix"] += torch.tensor(local_mix, dtype=torch.float64)
            local_mix_std = stats.get("local_mix_std", [])
            if local_mix_std:
                totals[name]["local_mix_std"] += torch.tensor(
                    local_mix_std, dtype=torch.float64
                )
            expert_local_mix = stats.get("expert_local_mix", [])
            if expert_local_mix:
                totals[name]["expert_local_mix"] += torch.tensor(
                    expert_local_mix, dtype=torch.float64
                )
            scales = stats.get("kernel_scales", [])
            if scales:
                totals[name]["kernel_scales"] += torch.tensor(
                    scales, dtype=torch.float64
                )
            scale_stats = stats.get("kernel_scale_stats", [])
            if scale_stats:
                scale_stats = torch.tensor(scale_stats, dtype=torch.float64)
                totals[name]["kernel_scale_std"] += scale_stats[:, 1]
                totals[name]["kernel_scale_min"] += scale_stats[:, 2]
                totals[name]["kernel_scale_max"] += scale_stats[:, 3]
            dynamic_probability = stats.get("dynamic_top_k_probability", [])
            if dynamic_probability:
                totals[name]["dynamic_top_k_probability"] += torch.tensor(
                    dynamic_probability, dtype=torch.float64
                )
                totals[name]["dynamic_top_k_mean"] += float(
                    stats["dynamic_top_k_mean"][0]
                )
                totals[name]["dynamic_top_k_std"] += float(
                    stats["dynamic_top_k_std"][0]
                )
            global_routing = stats.get("global_routing", [])
            if global_routing:
                totals[name]["global_routing"] += torch.tensor(
                    global_routing, dtype=torch.float64
                )
                totals[name]["global_routing_std"] += torch.tensor(
                    stats["global_routing_std"], dtype=torch.float64
                )
        batches += 1
        if args.max_batches and batch_index + 1 >= args.max_batches:
            break

    result = {
        "checkpoint": str(Path(args.operator_ckpt).resolve()),
        "model": args.model,
        "split": args.split,
        "batches": batches,
        "modules": {
            name: {
                "expert_kind": module.expert_kind,
                "importance": (values["importance"] / max(batches, 1)).tolist(),
                "load": (values["load"] / max(batches, 1)).tolist(),
                "temperature": values["temperature"] / max(batches, 1),
                "global_gate": values["global_gate"] / max(batches, 1),
                "local_scales": (
                    (values["local_scales"] / max(batches, 1)).tolist()
                    if module.local_context_enabled
                    else []
                ),
                "local_mix": (
                    (values["local_mix"] / max(batches, 1)).tolist()
                    if module.local_scale_mixer is not None
                    else []
                ),
                "local_mix_std": (
                    (values["local_mix_std"] / max(batches, 1)).tolist()
                    if module.local_scale_mixer is not None
                    else []
                ),
                "expert_local_mix": (
                    (values["expert_local_mix"] / max(batches, 1)).tolist()
                    if module.expert_local_mix_logits is not None
                    else []
                ),
                "kernel_scales": (
                    (values["kernel_scales"] / max(batches, 1)).tolist()
                    if module.expert_kind in {"kernel", "adaptive_kernel"}
                    else []
                ),
                "kernel_scale_std": (
                    (values["kernel_scale_std"] / max(batches, 1)).tolist()
                    if module.expert_kind in {"kernel", "adaptive_kernel"}
                    else []
                ),
                "kernel_scale_min": (
                    (values["kernel_scale_min"] / max(batches, 1)).tolist()
                    if module.expert_kind in {"kernel", "adaptive_kernel"}
                    else []
                ),
                "kernel_scale_max": (
                    (values["kernel_scale_max"] / max(batches, 1)).tolist()
                    if module.expert_kind in {"kernel", "adaptive_kernel"}
                    else []
                ),
                "dynamic_top_k_values": list(module.dynamic_top_k_values),
                "dynamic_top_k_probability": (
                    (values["dynamic_top_k_probability"] / max(batches, 1)).tolist()
                    if module.cardinality_router is not None
                    else []
                ),
                "dynamic_top_k_mean": (
                    values["dynamic_top_k_mean"] / max(batches, 1)
                    if module.cardinality_router is not None
                    else None
                ),
                "dynamic_top_k_std": (
                    values["dynamic_top_k_std"] / max(batches, 1)
                    if module.cardinality_router is not None
                    else None
                ),
                "hierarchical_global_routing": module.hierarchical_global_routing,
                "global_routing": (
                    (values["global_routing"] / max(batches, 1)).tolist()
                    if module.global_router is not None
                    else []
                ),
                "global_routing_std": (
                    (values["global_routing_std"] / max(batches, 1)).tolist()
                    if module.global_router is not None
                    else []
                ),
            }
            for name, module in modules
            for values in [totals[name]]
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--operator-ckpt", required=True)
    parser.add_argument("--model", choices=["fno", "transolver", "cnextu"], required=True)
    parser.add_argument("--act-root", default="/media/home/lizhonghao/act/ACT-NeuralOP")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
