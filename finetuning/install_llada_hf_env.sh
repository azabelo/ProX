#!/usr/bin/env bash
# Create conda env `prox-llada-hf` for Hugging Face Trainer finetuning (e.g. LLaDA / custom_code).
#
# Installs CUDA 12.4 PyTorch wheels first (see scripts/install_pytorch_cu124.sh rationale), then HF deps.
#
# Usage:
#   bash finetuning/install_llada_hf_env.sh
#   conda activate prox-llada-hf
#   python finetuning/finetune.py finetuning/example_finetune_qwen36_27b_dflash_hf.yaml

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-prox-llada-hf}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists; upgrading pip deps only."
else
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install torch==2.6.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
conda run -n "$ENV_NAME" pip install -r "$ROOT/finetuning/requirements_llada_hf.txt"

echo "Done. Activate with: conda activate $ENV_NAME"
