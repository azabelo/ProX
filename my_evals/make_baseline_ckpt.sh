#!/bin/bash
# Materialise a HuggingFace model repo as a local ``hf_ckpt/`` directory so it
# can be passed to ``my_evals/eval_lm_eval.sh`` and compared with the trained
# checkpoints in ``my_evals/eval_compare.py`` without any special-casing.
#
# Why a separate script: ``eval_lm_eval.sh`` writes its results to
# ``<MODEL_PATH>/lm_eval_results/``. If you point it at a bare HF repo ID
# (e.g. ``unsloth/Llama-3.2-1B``) it would create a relative ``unsloth/...``
# directory next to your cwd, which won't match the layout the comparison
# script expects. Downloading once into the same ``checkpoints/<run>/`` tree
# means the baseline behaves like every other ``global_step_<N>`` entry.
#
# Usage:
#   ./my_evals/make_baseline_ckpt.sh
#       # downloads unsloth/Llama-3.2-1B to:
#       #   continued_pretraining_outputs/dclm_llama3p2_1b_fast/checkpoints/
#       #     baseline_unsloth_llama-3.2-1b/hf_ckpt/
#
#   ./my_evals/make_baseline_ckpt.sh <hf_repo>
#   ./my_evals/make_baseline_ckpt.sh <hf_repo> <output_parent_dir>
#       # output_parent_dir gets an ``hf_ckpt/`` subdir written inside it.
#
# Idempotent: if the target ``hf_ckpt`` already has a ``config.json`` we skip
# the download.
set -euo pipefail

REPO="${1:-unsloth/Llama-3.2-1B}"

REPO_SANITIZED="$(echo "$REPO" | tr '[:upper:]' '[:lower:]' | tr '/' '_')"
DEFAULT_PARENT="/home/ubuntu/ProX/continued_pretraining_outputs/dclm_llama3p2_1b_fast/checkpoints/baseline_${REPO_SANITIZED}"
OUTPUT_PARENT="${2:-$DEFAULT_PARENT}"
HF_DIR="${OUTPUT_PARENT%/}/hf_ckpt"

PRETRAIN_ENV="${PRETRAIN_ENV:-pretrain}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$PRETRAIN_ENV" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$PRETRAIN_ENV"
    else
        echo "[make_baseline_ckpt] conda not on PATH; current env is '${CONDA_DEFAULT_ENV:-<none>}'." >&2
        echo "[make_baseline_ckpt] activate the '$PRETRAIN_ENV' env first." >&2
        exit 1
    fi
fi

if [ -f "$HF_DIR/config.json" ] && [ -f "$HF_DIR/tokenizer.json" ]; then
    echo "[make_baseline_ckpt] $HF_DIR already populated; skipping download."
else
    mkdir -p "$HF_DIR"
    echo "[make_baseline_ckpt] downloading '$REPO' -> $HF_DIR"
    python - "$REPO" "$HF_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target_dir = sys.argv[1], sys.argv[2]
# allow_patterns: skip *.bin if .safetensors is also published, to avoid
# wasting ~2 GB on a duplicate format we never use. Most modern repos list
# both; HF will use safetensors automatically when both exist.
path = snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.model",
        "*.txt",
        "tokenizer*",
        "special_tokens_map*",
        "generation_config*",
    ],
)
print(f"[make_baseline_ckpt] snapshot_download done: {path}")
PY
fi

echo "[make_baseline_ckpt] checkpoint ready at: $HF_DIR"
echo "[make_baseline_ckpt] next steps:"
echo "  ./my_evals/eval_lm_eval.sh $HF_DIR"
echo "  # then add this line to my_evals/example_eval_compare.yaml under 'models:':"
echo "  #   baseline: $HF_DIR"
