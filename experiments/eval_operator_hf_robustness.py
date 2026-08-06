#!/usr/bin/env python3
"""Evaluate latent operators after high-frequency perturbations to the input history."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hf_vae import build_from_ckpt
from experiments.eval_operator_physical import physical_channels, per_sample_rel
from experiments.train_latent_operator import add_act_root, build_model, unwrap_model_output
from hfvae.frequency import highpass_filter_2d
from hfvae.ns_data import resize_video


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    add_act_root(args.act_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    operator_ckpt = torch.load(args.operator_ckpt, map_location="cpu", weights_only=False)
    operator_args = SimpleNamespace(**operator_ckpt["args"])
    latent_root = Path(args.latent_root).resolve()
    norm = torch.load(
        latent_root / "norm_stats_latent.pt", map_location="cpu", weights_only=False
    )
    manifest = torch.load(
        latent_root / args.split / "manifest.pt", map_location="cpu", weights_only=False
    )
    first_shard = torch.load(
        latent_root / args.split / manifest["shards"][0]["path"],
        map_location="cpu",
        weights_only=False,
    )
    _, channels, frames, height, width = first_shard["z"].shape
    history = int(operator_args.history)
    model = build_model(
        args.model,
        history * channels + (3 if operator_args.use_phys else 0),
        channels,
        height,
        width,
        operator_args,
    ).to(device)
    model.load_state_dict(operator_ckpt["model"])
    model.eval()

    vae, vae_ckpt = build_from_ckpt(args.vae_ckpt, device)
    vae.eval()
    vae_mean = torch.tensor(
        vae_ckpt.get("norm_mean", [0.0] * 3), device=device
    ).view(1, 3, 1, 1, 1)
    vae_std = torch.tensor(vae_ckpt["norm_std"], device=device).view(1, 3, 1, 1, 1)
    latent_mean = torch.tensor(norm["mean"], device=device).view(1, channels, 1, 1, 1)
    latent_std = torch.tensor(norm["std"], device=device).view(1, channels, 1, 1, 1)

    latent_total = 0.0
    physical_total = 0.0
    count = 0
    shards = list(manifest["shards"])
    if args.max_shards:
        shards = shards[: args.max_shards]

    for shard_meta in shards:
        cached = torch.load(
            latent_root / args.split / shard_meta["path"],
            map_location="cpu",
            weights_only=False,
        )
        source = cached.get("source")
        if not source:
            raise ValueError(f"cache shard has no source path: {shard_meta['path']}")
        source_data = torch.load(source, map_location="cpu", weights_only=False)
        z_raw = cached["z"].float()
        uvp = source_data["uvp"].float()
        if args.max_samples_per_shard:
            limit = int(args.max_samples_per_shard)
            z_raw = z_raw[:limit]
            uvp = uvp[:limit]
        target = uvp.permute(0, 1, 4, 2, 3).contiguous().to(device)
        target = (target - vae_mean) / (vae_std + 1e-8)
        target = resize_video(target, args.out_hw, args.out_t)
        regime = cached.get("regime", {})

        for start in range(0, z_raw.shape[0], args.batch):
            end = min(start + args.batch, z_raw.shape[0])
            z_batch = z_raw[start:end].to(device)
            target_batch = target[start:end]
            z_norm = (z_batch - latent_mean) / (latent_std + 1e-8)
            batch_size, _channels, _frames, h, w = z_norm.shape
            phys = physical_channels(regime, h, w, device).unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )

            clean_history = [z_norm[:, :, index] for index in range(history)]
            noise = torch.randn(
                batch_size,
                channels,
                history,
                h,
                w,
                device=device,
                dtype=z_norm.dtype,
            )
            noise = highpass_filter_2d(
                noise, args.noise_cutoff, args.noise_transition
            )
            noise_rms = noise.square().mean(dim=(-2, -1), keepdim=True).sqrt()
            noise = noise / noise_rms.clamp_min(1e-6) * args.noise_std
            noisy_history = [
                clean_history[index] + noise[:, :, index]
                for index in range(history)
            ]

            predictions = []
            history_values = noisy_history
            for _ in range(history, frames):
                op_input = torch.stack(history_values[-history:], dim=1).reshape(
                    batch_size, history * channels, h, w
                )
                if operator_args.use_phys:
                    op_input = torch.cat((op_input, phys), dim=1)
                predictions.append(unwrap_model_output(model(op_input)))
                history_values.append(predictions[-1])

            prediction_norm = torch.stack(predictions, dim=2)
            target_latent = z_norm[:, :, history:]
            latent_total += float(
                per_sample_rel(prediction_norm.float(), target_latent.float()).sum()
            )

            rollout_norm = torch.cat(
                (z_norm[:, :, :history], prediction_norm), dim=2
            )
            rollout_raw = rollout_norm * latent_std + latent_mean
            with torch.amp.autocast(
                "cuda", enabled=args.amp and device.type == "cuda"
            ):
                decoded = vae.decode(rollout_raw, [0, 1])
            physical_total += float(
                per_sample_rel(
                    decoded[:, :, history:].float(), target_batch[:, :, history:].float()
                ).sum()
            )
            count += batch_size

        del source_data, target

    return {
        "model": args.model,
        "latent_root": str(latent_root),
        "split": args.split,
        "samples": count,
        "noise_std": args.noise_std,
        "noise_cutoff": args.noise_cutoff,
        "noise_transition": args.noise_transition,
        "latent_forecast_rel": latent_total / max(count, 1),
        "decoded_physical_forecast_rel": physical_total / max(count, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-root", required=True)
    ap.add_argument("--operator-ckpt", required=True)
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--act-root", default="/media/home/lizhonghao/act/ACT-NeuralOP")
    ap.add_argument("--model", choices=["fno", "transolver", "cnextu"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out-hw", type=int, default=128)
    ap.add_argument("--out-t", type=int, default=49)
    ap.add_argument("--noise-std", type=float, default=0.1)
    ap.add_argument("--noise-cutoff", type=float, default=0.5)
    ap.add_argument("--noise-transition", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    if args.noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    result = evaluate(args)
    print(json.dumps(result, sort_keys=True))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
