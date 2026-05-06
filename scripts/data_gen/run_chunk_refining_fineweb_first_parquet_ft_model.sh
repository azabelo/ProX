#!/usr/bin/env bash
# Chunk-refine FineWeb (first parquet shard) using a **local Hugging Face model directory**
# (your fine-tuned chunk LM), same prompts/IO as run_chunk_refining_fineweb_first_parquet.sh.
#
# vLLM loads HF checkpoints only. VeOmni DCP shards (*.distcp) under checkpoints/global_step_*
# are NOT valid --model_path. You need an HF export:
#   .../checkpoints/global_step_<N>/hf_ckpt/
# produced when finetuning had train.checkpoint.save_hf_weights enabled (see qwen05b_refiner_chunk_pairs.yaml).
#
# Override MODEL explicitly if your export lives elsewhere; Hub IDs (org/name) also work.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

LIMIT="${LIMIT:--1}"
NGPU="${NGPU:-8}"
GPU_MEM="${GPU_MEM:-0.72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

FT_OUT="${FT_OUT:-${ROOT}/finetuning_refiner/outputs/qwen05b_chunk_program_pairs}"
CONFIG="${CONFIG:-data_gen/configs/apply_chunk_refining_fineweb_first_parquet_ft.yaml}"

# Resolve MODEL: env MODEL, else first existing hf_ckpt under FT_OUT, else fail with hint.
if [[ -z "${MODEL:-}" ]]; then
  for cand in \
    "${FT_OUT}/checkpoints/global_step_1000/hf_ckpt" \
    "${FT_OUT}/checkpoints/global_step_500/hf_ckpt" \
    "${FT_OUT}/checkpoints/global_step_0/hf_ckpt"
  do
    if [[ -d "${cand}" ]] && [[ -f "${cand}/config.json" || -f "${cand}/model.safetensors" ]]; then
      MODEL="${cand}"
      echo "Using HF export: ${MODEL}"
      break
    fi
  done
fi

if [[ -z "${MODEL:-}" ]]; then
  echo "ERROR: No MODEL set and no hf_ckpt found under ${FT_OUT}/checkpoints/*/hf_ckpt." >&2
  echo "Export HF weights from finetuning (train.checkpoint.save_hf_weights: true), then either:" >&2
  echo "  MODEL=/path/to/.../hf_ckpt bash $0" >&2
  echo "or re-run finetuning with HF export enabled so hf_ckpt appears." >&2
  exit 1
fi

# Local path sanity check (Hub IDs contain '/' too — require config.json for paths under ROOT or absolute).
if [[ "${MODEL}" == /* || "${MODEL}" == "${ROOT}"/* ]]; then
  if [[ ! -f "${MODEL}/config.json" ]]; then
    echo "ERROR: MODEL directory missing config.json: ${MODEL}" >&2
    echo "That path is not a Hugging Face model folder vLLM can load." >&2
    echo "VeOmni checkpoints under checkpoints/global_step_*/ are mostly *.distcp (DCP), not HF." >&2
    echo "Fix: re-run / resume finetuning with train.checkpoint.save_hf_weights: true (see qwen05b_refiner_chunk_pairs.yaml)." >&2
    echo "Then use .../global_step_<N>/hf_ckpt/ (contains config.json + model weights)." >&2
    exit 1
  fi
fi

SAVE_BASE="${SAVE_BASE:-${ROOT}/data/raw/chunk_refining_fineweb_first_parquet_ft}"
MERGED_OUT="${MERGED_OUT:-${SAVE_BASE}/fineweb_sample_10BT_chunk_refined_ft.parquet}"
SKIP_MERGE="${SKIP_MERGE:-0}"

mkdir -p "${SAVE_BASE}" logging

export NNODE="${NNODE:-1}"
export NGPU
export TOTAL_SPLIT=$((NNODE * NGPU))

echo "TOTAL_SPLIT=${TOTAL_SPLIT} LIMIT=${LIMIT} GPU_MEM=${GPU_MEM} MAX_NUM_SEQS=${MAX_NUM_SEQS} ENFORCE_EAGER=${ENFORCE_EAGER}"
echo "MODEL=${MODEL}"
echo "SAVE_BASE=${SAVE_BASE} MERGED_OUT=${MERGED_OUT} SKIP_MERGE=${SKIP_MERGE}"

CHUNK_EXTRA=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CHUNK_EXTRA+=(--enforce_eager)
fi

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  TOTAL_SPLIT="${TOTAL_SPLIT}" NODE_GPUS="${NGPU}" NODE_RANK=0 CUDA_VISIBLE_DEVICES="${i}" \
    conda run -s -n refining \
    python -u -m data_gen.tasks.apply_chunk_refining \
      --data_format parquet \
      --limit "${LIMIT}" \
      --gpu_memory_utilization "${GPU_MEM}" \
      --max_num_seqs "${MAX_NUM_SEQS}" \
      --model_path "${MODEL}" \
      --config_path "${CONFIG}" \
      "${CHUNK_EXTRA[@]}" \
    > "logging/apply_chunk_refining_fineweb_fp_ft_${i}.log" 2>&1 &
  pids+=("$!")
done
worker_ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    worker_ec=1
  fi
done
if [[ "${worker_ec}" != "0" ]]; then
  echo "ERROR: One or more chunk-refining workers exited non-zero. See logging/apply_chunk_refining_fineweb_fp_ft_*.log" >&2
  echo "Merge was skipped so old prox_*/ shards were not re-packaged into a misleading merged parquet." >&2
  exit "${worker_ec}"
fi

echo "Done. Parquet shards under: ${SAVE_BASE}/prox_*/"
echo "Per-rank stats: ${SAVE_BASE}/chunk_refining_stats_rank*.json"

if [[ "${SKIP_MERGE}" != "1" ]]; then
  conda run -s -n refining python -u my_Scripts/merge_chunk_refining_shards_to_one_parquet.py \
    --input-dir "${SAVE_BASE}" \
    --output "${MERGED_OUT}"
  echo "Single merged parquet: ${MERGED_OUT}"
  echo "Note: merged row count comes from prox_*/prox_*.parquet shards. Remove stale shards (e.g. rm -rf \"${SAVE_BASE}\"/prox_*) before a run if you want a clean output dir."
else
  echo "SKIP_MERGE=1 — merge skipped."
fi
