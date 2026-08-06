#!/usr/bin/env python3
"""Decode latent operator rollouts and measure physical-field error."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_hf_vae import build_from_ckpt
from experiments.train_latent_operator import add_act_root, build_model, unwrap_model_output
from hfvae.frequency import highpass_filter_2d, lowpass_filter_2d
from hfvae.ns_data import resize_video


def physical_channels(regime: dict, h: int, w: int, device: torch.device) -> torch.Tensor:
    visc = float(regime.get("visc", 1e-3))
    forcing_k = float(regime.get("forcing_k", 1.0))
    forcing_amp = float(regime.get("forcing_amp", 0.1))
    values = torch.tensor(
        [math.log(1.0 / visc) - 8.0, forcing_k / 4.0, forcing_amp / 0.1],
        dtype=torch.float32,
        device=device,
    )
    return values[:, None, None].expand(-1, h, w)


def per_sample_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    return (pred - target).square().mean(dim=dims).sqrt() / (
        target.square().mean(dim=dims).sqrt() + 1e-8
    )


def per_sample_energy(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(dim=tuple(range(1, x.ndim)))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    add_act_root(args.act_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op_ckpt = torch.load(args.operator_ckpt, map_location="cpu", weights_only=False)
    op_args = SimpleNamespace(**op_ckpt["args"])
    root = Path(args.latent_root).resolve()
    norm = torch.load(root / "norm_stats_latent.pt", map_location="cpu", weights_only=False)
    first = torch.load(
        root / args.split / "manifest.pt", map_location="cpu", weights_only=False
    )
    first_shard = torch.load(
        root / args.split / first["shards"][0]["path"],
        map_location="cpu",
        weights_only=False,
    )
    _, channels, frames, height, width = first_shard["z"].shape
    history = int(op_args.history)
    model = build_model(
        args.model,
        history * channels + (3 if op_args.use_phys else 0),
        channels,
        height,
        width,
        op_args,
    ).to(device)
    model.load_state_dict(op_ckpt["model"])
    model.eval()
    operator_parameters = sum(parameter.numel() for parameter in model.parameters())

    vae, vae_ckpt = build_from_ckpt(args.vae_ckpt, device)
    vae.eval()
    vae_parameters = sum(parameter.numel() for parameter in vae.parameters())
    vae_mean = torch.tensor(vae_ckpt.get("norm_mean", [0.0] * 3), device=device).view(
        1, 3, 1, 1, 1
    )
    vae_std = torch.tensor(vae_ckpt["norm_std"], device=device).view(1, 3, 1, 1, 1)
    latent_mean = torch.tensor(norm["mean"], device=device).view(1, channels, 1, 1, 1)
    latent_std = torch.tensor(norm["std"], device=device).view(1, channels, 1, 1, 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    eval_start = time.perf_counter()

    total_data = 0.0
    total_decoded = 0.0
    total_low_data = 0.0
    total_high_error_energy = 0.0
    total_high_target_energy = 0.0
    total_count = 0
    operator_rollout_seconds = 0.0
    vae_decode_seconds = 0.0
    operator_forecast_frames = 0
    shards = list(first["shards"])
    if args.max_shards:
        shards = shards[: args.max_shards]
    for shard_meta in shards:
        cached = torch.load(
            root / args.split / shard_meta["path"],
            map_location="cpu",
            weights_only=False,
        )
        z_raw = cached["z"].float()
        source = cached.get("source")
        if not source:
            raise ValueError(f"cache shard has no source path: {shard_meta['path']}")
        source_data = torch.load(source, map_location="cpu", weights_only=False)
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
            b, _c, _frames, h, w = z_norm.shape
            phys = physical_channels(regime, h, w, device).unsqueeze(0).expand(b, -1, -1, -1)
            history_values = [z_norm[:, :, i] for i in range(history)]
            predictions = []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rollout_start = time.perf_counter()
            for _ in range(history, frames):
                op_input = torch.stack(history_values[-history:], dim=1).reshape(
                    b, history * channels, h, w
                )
                if op_args.use_phys:
                    op_input = torch.cat([op_input, phys], dim=1)
                prediction = unwrap_model_output(model(op_input))
                predictions.append(prediction)
                history_values.append(prediction)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            operator_rollout_seconds += time.perf_counter() - rollout_start
            operator_forecast_frames += b * (frames - history)
            rollout_norm = torch.cat(
                [z_norm[:, :, :history], torch.stack(predictions, dim=2)], dim=2
            )
            rollout_raw = rollout_norm * latent_std + latent_mean
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                decoded_rollout = vae.decode(rollout_raw, [0, 1])
                decoded_true = vae.decode(z_batch, [0, 1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            vae_decode_seconds += time.perf_counter() - decode_start
            total_data += float(per_sample_rel(decoded_rollout.float(), target_batch).sum())
            total_decoded += float(
                per_sample_rel(decoded_rollout.float(), decoded_true.float()).sum()
            )
            low_rollout = lowpass_filter_2d(
                decoded_rollout.float(), args.low_cutoff, args.transition
            )
            low_target = lowpass_filter_2d(
                target_batch.float(), args.low_cutoff, args.transition
            )
            high_error = highpass_filter_2d(
                decoded_rollout.float() - target_batch.float(),
                args.high_cutoff,
                args.transition,
            )
            high_target = highpass_filter_2d(
                target_batch.float(), args.high_cutoff, args.transition
            )
            total_low_data += float(per_sample_rel(low_rollout, low_target).sum())
            total_high_error_energy += float(per_sample_energy(high_error).sum())
            total_high_target_energy += float(per_sample_energy(high_target).sum())
            total_count += b
        del source_data, target

    eval_wall_seconds = time.perf_counter() - eval_start
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        peak_allocated_mib = None
        peak_reserved_mib = None

    return {
        "model": args.model,
        "latent_root": str(root),
        "split": args.split,
        "samples": total_count,
        "operator_parameters": operator_parameters,
        "vae_parameters": vae_parameters,
        "evaluation_device": str(device),
        "operator_precision": "fp32",
        "vae_decode_amp": bool(args.amp and device.type == "cuda"),
        "evaluation_wall_seconds": eval_wall_seconds,
        "operator_rollout_seconds": operator_rollout_seconds,
        "vae_decode_seconds": vae_decode_seconds,
        "operator_trajectories_per_second": total_count
        / max(operator_rollout_seconds, 1e-12),
        "operator_forecast_frames": operator_forecast_frames,
        "operator_forecast_frames_per_second": operator_forecast_frames
        / max(operator_rollout_seconds, 1e-12),
        "evaluation_peak_allocated_mib": peak_allocated_mib,
        "evaluation_peak_reserved_mib": peak_reserved_mib,
        "decoded_rollout_to_data_rel": total_data / max(total_count, 1),
        "decoded_rollout_vs_vae_rel": total_decoded / max(total_count, 1),
        "decoded_rollout_low_to_data_rel": total_low_data / max(total_count, 1),
        "decoded_rollout_high_error_energy": total_high_error_energy / max(total_count, 1),
        "decoded_target_high_energy": total_high_target_energy / max(total_count, 1),
        "decoded_rollout_high_error_to_target_energy": total_high_error_energy
        / max(total_high_target_energy, 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-root", required=True)
    ap.add_argument("--operator-ckpt", required=True)
    ap.add_argument("--vae-ckpt", required=True)
    ap.add_argument("--act-root", default="/media/home/lizhonghao/act/ACT-NeuralOP")
    ap.add_argument("--model", choices=["fno", "transolver", "cnextu"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-hw", type=int, default=128)
    ap.add_argument("--out-t", type=int, default=49)
    ap.add_argument("--low-cutoff", type=float, default=0.10)
    ap.add_argument("--high-cutoff", type=float, default=0.15)
    ap.add_argument("--transition", type=float, default=0.04)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    result = evaluate(args)
    print(json.dumps(result, sort_keys=True))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
