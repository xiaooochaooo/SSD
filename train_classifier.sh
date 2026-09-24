#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GPU_ID="${GPU_ID:-0}"
mkdir -p checkpoints/frozen_biview_transformer_300k_final logs

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u train.py \
  --train_path datasets/train.pt \
  --val_path datasets/val.pt \
  --test_path "" \
  --seq_ckpt checkpoints/C4/sequential_transformer_c4_300k.pt \
  --graph_ckpt checkpoints/C4/graph_rgcn_c4_300k.pt \
  --save_dir checkpoints/frozen_biview_transformer_300k_final \
  --batch_size 16 \
  --epochs 30 \
  --lr 1e-5 \
  --min_lr 1e-6 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-4 \
  --dropout 0.6 \
  --label_smoothing 0.05 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --no-save_every_epoch \
  --amp \
  --mmap_data \
  "$@" \
  2>&1 | tee logs/train_classifier_transformer_300k.log
