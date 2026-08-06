#!/usr/bin/env python3
"""Score grid operators from continuous, irregularly observed latent histories."""

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
from experiments.eval_operator_physical import (
    per_sample_energy,
    per_sample_rel,
    physical_channels,
)
from experiments.train_latent_operator import add_act_root, build_model, unwrap_model_output
from hfvae.frequency import highpass_filter_2d, lowpass_filter_2d
from hfvae.irregular import (
    default_rbf_length_scale,
    periodic_rbf_lift,
    sample_irregular_coordinates,
    sample_periodic_bilinear,
)
from hfvae.ns_data import resize_video


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    add_act_root(args.act_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    operator_ckpt = torch.load(args.operator_ckpt, map_location="cpu", weights_only=False)
    operator_args = SimpleNamespace(**operator_ckpt["args"])
    root = Path(args.latent_root).resolve()
    norm = torch.load(root / "norm_stats_latent.pt", map_location="cpu", weights_only=False)
    manifest = torch.load(root / args.split / "manifest.pt", map_location="cpu", weights_only=False)
    first_shard = torch.load(
        root / args.split / manifest["shards"][0]["path"],
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

    vae_mean = torch.tensor(vae_ckpt.get("norm_mean", [0.0] * 3), device=device).view(
        1, 3, 1, 1, 1
    )
    vae_std = torch.tensor(vae_ckpt["norm_std"], device=device).view(1, 3, 1, 1, 1)
    latent_mean = torch.tensor(norm["mean"], device=device).view(1, channels, 1, 1, 1)
    latent_std = torch.tensor(norm["std"], device=device).view(1, channels, 1, 1, 1)
    length_scale = args.length_scale or default_rbf_length_scale(args.n_obs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_start = time.perf_counter()

    total_latent = 0.0
    total_data = 0.0
    total_vae = 0.0
    total_low_data = 0.0
    total_high_error = 0.0
    total_high_target = 0.0
    total_count = 0
    lift_seconds = 0.0
    operator_seconds = 0.0
    decode_seconds = 0.0
    forecast_frames = 0
    timed_trajectories = 0
    batch_index = 0
    shards = list(manifest["shards"])
    if args.max_shards:
        shards = shards[: args.max_shards]
    for shard_index, shard_meta in enumerate(shards):
        cached = torch.load(
            root / args.split / shard_meta["path"], map_location="cpu", weights_only=False
        )
        z_raw = cached["z"].float()
        source = cached.get("source")
        if not source:
            raise ValueError(f"cache shard has no source path: {shard_meta['path']}")
        source_data = torch.load(source, map_location="cpu", weights_only=False)
        uvp = source_data["uvp"].float()
        if args.max_samples_per_shard:
            z_raw = z_raw[: args.max_samples_per_shard]
            uvp = uvp[: args.max_samples_per_shard]
        physical_target = uvp.permute(0, 1, 4, 2, 3).contiguous().to(device)
        physical_target = (physical_target - vae_mean) / (vae_std + 1e-8)
        physical_target = resize_video(physical_target, args.out_hw, args.out_t)
        regime = cached.get("regime", {})
        for start in range(0, z_raw.shape[0], args.batch):
            end = min(start + args.batch, z_raw.shape[0])
            z_batch = z_raw[start:end].to(device)
            physical_batch = physical_target[start:end]
            z_norm = (z_batch - latent_mean) / (latent_std + 1e-8)
            batch = z_norm.shape[0]
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + shard_index * 1_000_003 + start)
            coords = sample_irregular_coordinates(
                batch,
                args.n_obs,
                device=device,
                dtype=z_norm.dtype,
                generator=generator,
                pattern=args.pattern,
            )
            mask = torch.ones(batch, args.n_obs, device=device, dtype=torch.bool)
            packed_history = z_norm[:, :, :history].permute(0, 2, 1, 3, 4).reshape(
                batch, history * channels, height, width
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            lift_start = time.perf_counter()
            observed = sample_periodic_bilinear(packed_history, coords, mask)
            lifted_history, _coverage = periodic_rbf_lift(
                observed,
                coords,
                height=height,
                width=width,
                length_scale=length_scale,
                mask=mask,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            lift_seconds += time.perf_counter() - lift_start

            physical = physical_channels(regime, height, width, device).unsqueeze(0).expand(
                batch, -1, -1, -1
            )
            states = [
                lifted_history[:, index * channels : (index + 1) * channels]
                for index in range(history)
            ]
            predictions = []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            operator_start = time.perf_counter()
            for _ in range(history, frames):
                operator_input = torch.stack(states[-history:], dim=1).reshape(
                    batch, history * channels, height, width
                )
                if operator_args.use_phys:
                    operator_input = torch.cat([operator_input, physical], dim=1)
                prediction = unwrap_model_output(model(operator_input))
                predictions.append(prediction)
                states.append(prediction)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if batch_index >= args.warmup_batches:
                operator_seconds += time.perf_counter() - operator_start
                forecast_frames += batch * (frames - history)
                timed_trajectories += batch
            prediction_tensor = torch.stack(predictions, dim=2)
            target_latent = z_norm[:, :, history:]
            total_latent += float(per_sample_rel(prediction_tensor, target_latent).sum())

            prediction_raw = prediction_tensor * latent_std + latent_mean
            target_raw = z_batch[:, :, history:]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                decoded_prediction = vae.decode(prediction_raw, [0, 1])
                decoded_target = vae.decode(target_raw, [0, 1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if batch_index >= args.warmup_batches:
                decode_seconds += time.perf_counter() - decode_start
            future_data = physical_batch[:, :, -decoded_prediction.shape[2] :]
            total_data += float(per_sample_rel(decoded_prediction.float(), future_data.float()).sum())
            total_vae += float(
                per_sample_rel(decoded_prediction.float(), decoded_target.float()).sum()
            )
            low_prediction = lowpass_filter_2d(
                decoded_prediction.float(), args.low_cutoff, args.transition
            )
            low_data = lowpass_filter_2d(future_data.float(), args.low_cutoff, args.transition)
            high_error = highpass_filter_2d(
                decoded_prediction.float() - future_data.float(),
                args.high_cutoff,
                args.transition,
            )
            high_target = highpass_filter_2d(
                future_data.float(), args.high_cutoff, args.transition
            )
            total_low_data += float(per_sample_rel(low_prediction, low_data).sum())
            total_high_error += float(per_sample_energy(high_error).sum())
            total_high_target += float(per_sample_energy(high_target).sum())
            total_count += batch
            batch_index += 1
        del source_data, physical_target

    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
        "n_obs": args.n_obs,
        "observation_fraction": args.n_obs / float(height * width),
        "pattern": args.pattern,
        "rbf_length_scale": length_scale,
        "latent_forecast_rel": total_latent / max(total_count, 1),
        "decoded_future_to_data_rel": total_data / max(total_count, 1),
        "decoded_future_vs_vae_rel": total_vae / max(total_count, 1),
        "decoded_future_low_to_data_rel": total_low_data / max(total_count, 1),
        "decoded_future_high_error_energy": total_high_error / max(total_count, 1),
        "decoded_future_high_error_to_target_energy": total_high_error
        / max(total_high_target, 1e-12),
        "evaluation_wall_seconds": time.perf_counter() - evaluation_start,
        "lift_seconds": lift_seconds,
        "operator_rollout_seconds": operator_seconds,
        "vae_decode_seconds": decode_seconds,
        "operator_timed_trajectories": timed_trajectories,
        "operator_forecast_frames": forecast_frames,
        "operator_trajectories_per_second": timed_trajectories
        / max(operator_seconds, 1e-12),
        "operator_forecast_frames_per_second": forecast_frames
        / max(operator_seconds, 1e-12),
        "evaluation_peak_allocated_mib": peak_allocated_mib,
        "evaluation_peak_reserved_mib": peak_reserved_mib,
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
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--pattern", choices=["uniform", "clustered", "boundary"], default="uniform")
    ap.add_argument("--length-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out-hw", type=int, default=128)
    ap.add_argument("--out-t", type=int, default=49)
    ap.add_argument("--low-cutoff", type=float, default=0.10)
    ap.add_argument("--high-cutoff", type=float, default=0.15)
    ap.add_argument("--transition", type=float, default=0.04)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--warmup-batches", type=int, default=1)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples-per-shard", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    if args.n_obs < 1:
        raise ValueError("--n-obs must be positive")
    if args.length_scale is not None and args.length_scale <= 0:
        raise ValueError("--length-scale must be positive")
    result = evaluate(args)
    print(json.dumps(result, sort_keys=True))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
