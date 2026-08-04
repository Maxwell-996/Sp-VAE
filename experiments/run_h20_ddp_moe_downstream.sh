#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/media/home/lizhonghao/diff_pde/data_128}"
ACT_ROOT="${ACT_ROOT:-/media/home/lizhonghao/act/ACT-NeuralOP}"

run_variant() {
  local name="$1"
  local cache_port="$2"
  local op_port="$3"
  local vae="runs/$name/${name}_best.pt"
  local latent="runs/latents_$name"
  local op="runs/operator_$name"
  local physical="runs/physical_$name"
  mkdir -p "runs/$name" "$op" "$physical"

  for split in train valid test; do
    torchrun --standalone --nproc_per_node=4 --master_port="$cache_port" \
      experiments/cache_vae_latents.py \
      --ckpt "$vae" --data-root "$DATA_ROOT" --split "$split" \
      --out-root "$latent" --batch 1 --t-clip 49 --out-t 49 --out-hw 128 --amp \
      > "runs/$name/cache_${split}.log" 2>&1
    cache_port=$((cache_port + 1))
  done

  torchrun --standalone --nproc_per_node=4 --master_port="$op_port" \
    experiments/train_latent_operator.py \
    --latent-root "$latent" --act-root "$ACT_ROOT" \
    --models fno transolver cnextu --out-csv "$op/results.csv" \
    --epochs 30 --batch 16 --lr 5e-4 --history 4 --hidden 64 \
    --layers 4 --heads 8 --slice-num 32 --modes 8 --stages 3 \
    --blocks-per-stage 1 --seed 3407 > "runs/$name/operator_train.log" 2>&1

  # Physical decode is inference-only; three models use three cards concurrently.
  env CUDA_VISIBLE_DEVICES=0 python3 -u experiments/eval_operator_physical.py \
    --latent-root "$latent" --operator-ckpt "$op/fno_latent_operator.pt" \
    --vae-ckpt "$vae" --model fno --split test --batch 1 \
    --out-hw 128 --out-t 49 --amp --out-json "$physical/fno.json" \
    > "runs/$name/physical_fno.log" 2>&1 &
  local fno_pid=$!
  env CUDA_VISIBLE_DEVICES=1 python3 -u experiments/eval_operator_physical.py \
    --latent-root "$latent" --operator-ckpt "$op/transolver_latent_operator.pt" \
    --vae-ckpt "$vae" --model transolver --split test --batch 1 \
    --out-hw 128 --out-t 49 --amp --out-json "$physical/transolver.json" \
    > "runs/$name/physical_transolver.log" 2>&1 &
  local transolver_pid=$!
  env CUDA_VISIBLE_DEVICES=2 python3 -u experiments/eval_operator_physical.py \
    --latent-root "$latent" --operator-ckpt "$op/cnextu_latent_operator.pt" \
    --vae-ckpt "$vae" --model cnextu --split test --batch 1 \
    --out-hw 128 --out-t 49 --amp --out-json "$physical/cnextu.json" \
    > "runs/$name/physical_cnextu.log" 2>&1 &
  local cnextu_pid=$!
  wait "$fno_pid" "$transolver_pid" "$cnextu_pid"
}

# The VAE launcher runs these in order. Waiting makes this script safe to start
# immediately and keeps all subsequent experiments attached to real checkpoints.
for name in hf_ddp_noise8 hf_ddp_fine16 hf_ddp_soft8; do
  while [ ! -f "runs/$name/${name}_best.pt" ]; do sleep 60; done
  run_variant "$name" 29701 29801
done
