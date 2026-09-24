#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GPU_ID="${GPU_ID:-0}"

CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate.py \
  --data_path datasets/test.pt \
  --ckpt_path checkpoints/frozen_biview_transformer_300k_final/best_auc_model.pt \
  --save_pred_path "" \
  --batch_size 128 \
  --threshold 0.41 \
  --amp \
  --mmap_data \
  "$@"
