#!/usr/bin/env bash
# Stage 2: finetune_refiner
#
# Fine-tune the refiner model on stage-1 chunk→program pairs parquet.
# This is a thin wrapper around:
#   finetuning_refiner/run_finetune_refiner.sh
#
# Example (10 steps smoke run; 1 GPU):
#
#   CUDA_VISIBLE_DEVICES=0 \
#   TRAIN_PARQUET=/path/to/chunk_pairs.parquet \
#   OUT_DIR=/tmp/qwen05b_chunk_program_pairs_smoke \
#   bash finetune_refiner/main.sh \
#     --train.max_steps 10 \
#     --train.checkpoint.save_steps 10 \
#     --train.checkpoint.hf_save_steps 10 \
#     --train.wandb.enable false
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${ROOT}/finetuning_refiner/run_finetune_refiner.sh" "$@"

