#!/usr/bin/env bash
# Create or repair the ``pretrain`` conda env for continued pretraining with FlashAttention.
#
# Why this exists:
# - ``pip install flash-attn`` usually downloads a prebuilt GitHub wheel. Those wheels sometimes
#   mismatch your exact PyTorch C++ ABI and fail at import (`undefined symbol: c10::Error...`).
# - Mixing ``~/.local`` (``pip install --user``) with a conda env makes ``pip`` see the wrong torch
#   during FlashAttention builds.
#
# Recommended (known-good stack on this machine: Python 3.10 + torch 2.6 + CUDA 12.4
# + flash-attn 2.7.4.post1 official wheel):
#
#   bash continued_pretraining/setup_pretrain_env.sh
#
# On a cluster node, prefer:
#   export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

set -euo pipefail

ENV_NAME="${1:-pretrain}"
PY="${PY:-310}"

clone_from_refining() {
    local refine="${REFINING_ENV:-refining}"
    if conda env list | awk '{print $1}' | grep -qx "$refine"; then
        echo "[setup_pretrain_env] cloning conda env $refine -> $ENV_NAME"
        conda create -y -n "${ENV_NAME}" --clone "${refine}"
        return 0
    fi
    return 1
}

bootstrap_empty_env() {
    echo "[setup_pretrain_env] creating fresh conda env $ENV_NAME with Python 3.10 + nvcc support"
    conda create -y -n "${ENV_NAME}" python=3.10 pip
}

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup_pretrain_env] env ${ENV_NAME} already exists — will only install/fix pip packages inside it."
else
    clone_from_refining || bootstrap_empty_env
fi

PIP=(conda run -n "${ENV_NAME}" --no-capture-output python -m pip)
CONDA_PYTHON="$(conda run -n "${ENV_NAME}" --no-capture-output python -c "import sys; print(sys.executable)")"

echo "[setup_pretrain_env] using interpreter: ${CONDA_PYTHON}"

# Never mix user-site packages into FlashAttention builds.
export PYTHONNOUSERSITE=1

"${PIP[@]}" install -U pip setuptools wheel ninja packaging cmake einops huggingface-hub safetensors \
    "transformers==4.57.3" accelerate pyarrow pandas pydantic tqdm pyyaml datasets "fsspec[http]<=2024.6.1,>=2023.1.0" \
    aiohttp xxhash triton sympy wandb

"${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

# Align FlashAttention EXT fallback builds with nvcc bundled with CUDA toolkit on the machine.
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
fi
echo "[setup_pretrain_env] CUDA_HOME=${CUDA_HOME:-<unset>}"
if [[ ! -d "${CUDA_HOME:-/nonexistent}/include" ]]; then
    echo "[setup_pretrain_env][warn] CUDA_HOME does not look valid. Install CUDA toolkit / nvcc, then re-run." >&2
fi

# Use the older official torch-2.6 wheel: 2.8.3's torch-2.6 cp310 wheel currently
# imports with a C++ ABI mismatch here, while 2.7.4.post1 is verified.
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

"${PIP[@]}" uninstall -y flash-attn 2>/dev/null || true
if ! "${PIP[@]}" install --no-deps "${FLASH_ATTN_WHEEL}"; then
    echo "[setup_pretrain_env][warn] prebuilt flash-attn wheel failed; falling back to local H100-only source build." >&2
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    MAX_JOBS="${MAX_JOBS:-16}"
    export MAX_JOBS
    "${PIP[@]}" install "flash-attn==2.7.4.post1" --no-build-isolation --no-cache-dir
fi

conda run -n "${ENV_NAME}" --no-capture-output python - <<'PY'
import torch
import flash_attn
print("[setup_pretrain_env] OK torch", torch.__version__, "flash_attn", flash_attn.__version__, "cuda", torch.cuda.is_available())
from flash_attn.flash_attn_interface import flash_attn_func
if torch.cuda.is_available():
    q = torch.randn(2, 16, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
    print("[setup_pretrain_env] OK flash_attn_func", tuple(out.shape), out.dtype, torch.cuda.get_device_name(0))
PY

echo "[setup_pretrain_env] done. Activate with:  conda activate ${ENV_NAME}"
echo "[setup_pretrain_env] Run training with PYTHONNOUSERSITE=1 to avoid ~/.local shadowing torch (optional but safer):"
echo "  PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=1 continued_pretraining/continue_pretraining.py ..."
