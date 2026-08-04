#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/media/home/lizhonghao/diff_pde/data_128}"
INIT_CKPT="${INIT_CKPT:-runs/vae_hf_noisy0/hf_noisy0_best.pt}"

common_args=(
  --data-root "$DATA_ROOT"
  --epochs 1
  --init-from "$INIT_CKPT"
  --batch 1
  --lr 5e-5
  --dim 96
  --z-dim 16
  --patch-size 2
  --t-clip 49
  --out-t 49
  --out-hw 128
  --lazy
  --amp
  --anti-alias
  --latent-hfs
  --output-hfs
  --decoder-spectral-filter
  --filter-cutoff 0.15
  --decoder-filter-min-gain 0.02
  --hfs-cutoff 0.15
  --latent-learnable-moe
  --learnable-moe-hidden 64
  --lambda-lowpass 0.5
  --lambda-hf 0.1
  --lambda-spectral 0
  --lambda-wavelet 0
  --lambda-sobolev 0.05
  --lambda-latent-consistency 0.05
  --lambda-moe-balance 0.01
  --lambda-noisy-reconstruction 0
  --lambda-temporal 0
  --lambda-vorticity 0
  --lowpass-cutoff 0.1
  --highpass-cutoff 0.15
  --transition 0.04
  --sobolev-s 1.0
  --consistency-noise 0.1
)

run_variant() {
  local name="$1"
  local port="$2"
  shift 2
  local out="runs/$name"
  mkdir -p "$out"
  echo "starting $name on all visible GPUs"
  torchrun --standalone --nproc_per_node=4 --master_port="$port" \
    experiments/train_hf_vae.py \
    --out-dir "$out" --tag "$name" \
    "${common_args[@]}" "$@" > "$out/train.log" 2>&1
  python3 -u experiments/eval_hf_vae.py \
    --ckpt "$out/${name}_best.pt" --data-root "$DATA_ROOT" \
    --split valid --batch 2 --t-clip 49 --amp \
    --low-cutoff 0.10 --high-cutoff 0.15 --transition 0.04 \
    --out-csv "$out/eval_valid.csv" > "$out/eval.log" 2>&1
  python3 -u experiments/eval_moe_routing.py \
    --ckpt "$out/${name}_best.pt" --data-root "$DATA_ROOT" \
    --split valid --batch 2 --amp --out "$out/routing_valid.json" \
    > "$out/routing.log" 2>&1
  echo "completed $name"
}

# Keep the comparison protocol fixed: noisy top-k, fine-grained shared MoE,
# and dense soft routing are trained from the same Wan checkpoint.
run_variant hf_ddp_noise8 29631 \
  --learnable-moe-experts 8 --learnable-moe-top-k 2 \
  --learnable-moe-routing token_choice --learnable-moe-shared-experts 1 \
  --learnable-moe-router-noise 0.1
run_variant hf_ddp_fine16 29632 \
  --learnable-moe-experts 16 --learnable-moe-top-k 4 \
  --learnable-moe-routing token_choice --learnable-moe-shared-experts 2 \
  --learnable-moe-router-noise 0.0
run_variant hf_ddp_soft8 29633 \
  --learnable-moe-experts 8 --learnable-moe-top-k 8 \
  --learnable-moe-routing token_choice --learnable-moe-shared-experts 1 \
  --learnable-moe-router-noise 0.0
