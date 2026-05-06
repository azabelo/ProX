#!/usr/bin/env bash
# Install PyTorch + torchvision + torchaudio built against CUDA 12.4.
#
# Why: wheels from PyPI named like `torch-2.11.0+cu130` need a *newer* driver than many
# Linux images that advertise "CUDA Version: 12.x" in nvidia-smi. PyTorch then disables CUDA
# (`torch.cuda.is_available()` is False) even when GPUs are visible.
#
# This wheel stack (cu124) matches typical 535+/550+/560+ drivers that support up through CUDA 12.6.
#
# Usage (recommended before `pip install -r requirements.txt` in a fresh env):
#   bash scripts/install_pytorch_cu124.sh
#   pip install -r requirements.txt
# If pip tries to upgrade torch back to a mismatched build, pin with:
#   pip install 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124
set -euo pipefail
exec pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
