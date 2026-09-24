#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GPU_ID="${GPU_ID:-0}"
mkdir -p checkpoints/C4 logs

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u sequential.py \
  --data_path "datasets/C4/c4_train_300k_part*-of-03.pt" \
  --save_path checkpoints/C4/sequential_transformer_c4_300k.pt \
  --roberta models/roberta-base \
  --encoder_type transformer \
  --hidden_dim 768 \
  --num_layers 2 \
  --num_heads 8 \
  --ffn_dim 3072 \
  --output_dim 768 \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --mask_ratio 0.30 \
  --cos_weight 0.10 \
  --dropout 0.10 \
  --max_grad_norm 1.0 \
  --amp \
  --mmap_data \
  "$@" \
  2>&1 | tee logs/train_seq_transformer_c4_300k.log
