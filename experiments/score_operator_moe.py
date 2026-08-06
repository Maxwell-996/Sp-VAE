#!/usr/bin/env python3
"""Train and score the fixed three-operator latent rollout protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--latent-root", default="runs/latents_hf_ddp_global8")
    ap.add_argument("--vae-ckpt", default="runs/hf_ddp_global8/hf_ddp_global8_best.pt")
    ap.add_argument("--models", nargs="+", default=["fno", "transolver", "cnextu"])
    ap.add_argument(
        "--operator-internal-moe-scope",
        choices=["all", "decoder", "neck_decoder"],
        default="all",
    )
    ap.add_argument(
        "--operator-moe-routing",
        choices=["token_choice", "expert_choice"],
        default="token_choice",
    )
    ap.add_argument(
        "--operator-moe-expert-kind",
        choices=["mlp", "kernel", "adaptive_kernel"],
        default="mlp",
    )
    ap.add_argument("--operator-moe-top-k", type=int, default=2)
    ap.add_argument("--operator-moe-dynamic-top-k-min", type=int, default=None)
    ap.add_argument("--operator-moe-dynamic-top-k-max", type=int, default=None)
    ap.add_argument(
        "--operator-moe-dynamic-top-k-scope",
        choices=["token", "sample"],
        default="token",
    )
    ap.add_argument("--operator-moe-hierarchical-global-routing", action="store_true")
    ap.add_argument("--operator-moe-shared-experts", type=int, default=2)
    ap.add_argument("--operator-moe-router-inhibition", action="store_true")
    ap.add_argument("--operator-moe-local-context", action="store_true")
    ap.add_argument("--operator-moe-adaptive-local-mix", action="store_true")
    ap.add_argument("--operator-moe-expert-local-mix", action="store_true")
    ap.add_argument("--train-rollout-horizon", type=int, default=1)
    ap.add_argument("--rollout-loss-gamma", type=float, default=1.0)
    ap.add_argument("--historical-rollout-ckpt", default=None)
    ap.add_argument("--historical-rollout-weight", type=float, default=0.0)
    ap.add_argument("--historical-rollout-margin", type=float, default=0.0)
    ap.add_argument("--historical-rollout-scale", type=float, default=0.01)
    ap.add_argument("--decoded-loss-weight", type=float, default=0.0)
    ap.add_argument("--decoded-loss-hf-weight", type=float, default=0.0)
    ap.add_argument("--master-port", type=int, default=29943)
    args = ap.parse_args()

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root is not None
        else Path(tempfile.mkdtemp(prefix="hfvae-operator-score-"))
    )
    if args.out_root is not None and out_root.exists():
        raise FileExistsError(f"score output already exists: {out_root}")
    if args.historical_rollout_ckpt is not None and len(args.models) != 1:
        raise ValueError("historical rollout scoring accepts exactly one model per run")
    operator_root = out_root / "operator"
    physical_root = out_root / "physical"
    operator_root.mkdir(parents=True)
    physical_root.mkdir(parents=True)

    train_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        f"--master_port={args.master_port}",
        "experiments/train_latent_operator.py",
        "--latent-root",
        args.latent_root,
        "--act-root",
        "/media/home/lizhonghao/act/ACT-NeuralOP",
        "--models",
        *args.models,
        "--out-csv",
        str(operator_root / "results.csv"),
        "--epochs",
        "30",
        "--batch",
        "16",
        "--lr",
        "5e-4",
        "--history",
        "4",
        "--train-rollout-horizon",
        str(args.train_rollout_horizon),
        "--rollout-loss-gamma",
        str(args.rollout_loss_gamma),
        "--hidden",
        "64",
        "--layers",
        "4",
        "--heads",
        "8",
        "--slice-num",
        "32",
        "--modes",
        "8",
        "--stages",
        "3",
        "--blocks-per-stage",
        "1",
        "--seed",
        "3407",
        "--operator-moe",
        "--operator-internal-moe",
        "--operator-internal-moe-scope",
        args.operator_internal_moe_scope,
        "--operator-moe-hidden",
        "64",
        "--operator-moe-experts",
        "8",
        "--operator-moe-top-k",
        str(args.operator_moe_top_k),
        "--operator-moe-global",
        "--operator-moe-routing",
        args.operator_moe_routing,
        "--operator-moe-expert-kind",
        args.operator_moe_expert_kind,
        "--operator-moe-shared-experts",
        str(args.operator_moe_shared_experts),
    ]
    if args.operator_moe_router_inhibition:
        train_cmd.append("--operator-moe-router-inhibition")
    if args.operator_moe_local_context:
        train_cmd.append("--operator-moe-local-context")
    if args.operator_moe_adaptive_local_mix:
        train_cmd.append("--operator-moe-adaptive-local-mix")
    if args.operator_moe_expert_local_mix:
        train_cmd.append("--operator-moe-expert-local-mix")
    if args.operator_moe_dynamic_top_k_min is not None:
        if args.operator_moe_dynamic_top_k_max is None:
            raise ValueError("dynamic top-k min and max must be set together")
        train_cmd.extend(
            [
                "--operator-moe-dynamic-top-k-min",
                str(args.operator_moe_dynamic_top_k_min),
                "--operator-moe-dynamic-top-k-max",
                str(args.operator_moe_dynamic_top_k_max),
                "--operator-moe-dynamic-top-k-scope",
                args.operator_moe_dynamic_top_k_scope,
            ]
        )
    elif args.operator_moe_dynamic_top_k_max is not None:
        raise ValueError("dynamic top-k min and max must be set together")
    if args.operator_moe_hierarchical_global_routing:
        train_cmd.append("--operator-moe-hierarchical-global-routing")
    if args.historical_rollout_ckpt is not None:
        train_cmd.extend(
            [
                "--historical-rollout-ckpt",
                args.historical_rollout_ckpt,
                "--historical-rollout-weight",
                str(args.historical_rollout_weight),
                "--historical-rollout-margin",
                str(args.historical_rollout_margin),
                "--historical-rollout-scale",
                str(args.historical_rollout_scale),
            ]
        )
    if args.decoded_loss_weight:
        train_cmd.extend(
            [
                "--decoded-loss-vae-ckpt",
                args.vae_ckpt,
                "--decoded-loss-weight",
                str(args.decoded_loss_weight),
                "--decoded-loss-hf-weight",
                str(args.decoded_loss_hf_weight),
            ]
        )
    subprocess.run(train_cmd, cwd=ROOT, check=True)

    jobs = []
    for index, model in enumerate(args.models):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        jobs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "experiments/eval_operator_physical.py",
                    "--latent-root",
                    args.latent_root,
                    "--operator-ckpt",
                    str(operator_root / f"{model}_latent_operator.pt"),
                    "--vae-ckpt",
                    args.vae_ckpt,
                    "--model",
                    model,
                    "--split",
                    "test",
                    "--batch",
                    "2",
                    "--amp",
                    "--out-json",
                    str(physical_root / f"{model}.json"),
                ],
                cwd=ROOT,
                env=env,
            )
        )
    for job in jobs:
        if job.wait() != 0:
            raise RuntimeError("physical operator scoring failed")

    values = {}
    for model in args.models:
        with (physical_root / f"{model}.json").open(encoding="utf-8") as handle:
            values[model] = json.load(handle)["decoded_rollout_to_data_rel"]
    mean_value = sum(values.values()) / len(values)
    print(json.dumps({"mean_physical_rel": mean_value, "physical": values}, sort_keys=True))


if __name__ == "__main__":
    main()
