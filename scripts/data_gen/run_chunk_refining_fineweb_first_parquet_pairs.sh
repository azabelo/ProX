#!/usr/bin/env bash
# Chunk LM → per-chunk (text, target) parquet for FineWeb first shard (sample/10BT), multi-GPU.
# Each row: chunk text (input) and model output for that chunk (target); no merged refined doc.
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
MODEL="${MODEL:-gair-prox/web-chunk-refining-lm}"
CONFIG="${CONFIG:-data_gen/configs/apply_chunk_refining_fineweb_first_parquet_pairs.yaml}"

SAVE_BASE="${SAVE_BASE:-${ROOT}/data/raw/chunk_refining_fineweb_first_parquet_pairs}"
MERGED_OUT="${MERGED_OUT:-${SAVE_BASE}/fineweb_sample_10BT_chunk_pairs.parquet}"
SKIP_MERGE="${SKIP_MERGE:-0}"

mkdir -p "${SAVE_BASE}" logging

export NNODE="${NNODE:-1}"
export NGPU
export TOTAL_SPLIT=$((NNODE * NGPU))

echo "TOTAL_SPLIT=${TOTAL_SPLIT} LIMIT=${LIMIT} GPU_MEM=${GPU_MEM} MAX_NUM_SEQS=${MAX_NUM_SEQS} ENFORCE_EAGER=${ENFORCE_EAGER} MODEL=${MODEL}"
echo "SAVE_BASE=${SAVE_BASE} MERGED_OUT=${MERGED_OUT} SKIP_MERGE=${SKIP_MERGE}"

CHUNK_EXTRA=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CHUNK_EXTRA+=(--enforce_eager)
fi

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  TOTAL_SPLIT="${TOTAL_SPLIT}" NODE_GPUS="${NGPU}" NODE_RANK=0 CUDA_VISIBLE_DEVICES="${i}" \
    conda run -s -n refining \
    python -u -m data_gen.tasks.apply_chunk_refining_pair_export \
      --data_format parquet \
      --limit "${LIMIT}" \
      --gpu_memory_utilization "${GPU_MEM}" \
      --max_num_seqs "${MAX_NUM_SEQS}" \
      --model_path "${MODEL}" \
      --config_path "${CONFIG}" \
      "${CHUNK_EXTRA[@]}" \
    > "logging/apply_chunk_refining_fineweb_fp_pairs_${i}.log" 2>&1 &
  pids+=("$!")
done
worker_ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    worker_ec=1
  fi
done
if [[ "${worker_ec}" != "0" ]]; then
  echo "ERROR: One or more pair-export workers exited non-zero. See logging/apply_chunk_refining_fineweb_fp_pairs_*.log" >&2
  exit "${worker_ec}"
fi

echo "Done. Pair parquet shards under: ${SAVE_BASE}/prox_*/"
echo "Per-rank stats: ${SAVE_BASE}/chunk_pair_export_stats_rank*.json"

shopt -s nullglob
shard_files=("${SAVE_BASE}"/prox_*/prox_*.parquet)
shopt -u nullglob
if [[ "${#shard_files[@]}" -eq 0 ]]; then
  echo "ERROR: No shard files matching ${SAVE_BASE}/prox_*/prox_*.parquet — nothing to merge." >&2
  echo "Typical causes: all ranks were assigned zero parquet rows (check LIMIT vs row sharding + NGPU), or every batch had no pair rows, or workers crashed before the first write (see logs above)." >&2
  exit 1
fi

if [[ "${SKIP_MERGE}" != "1" ]]; then
  conda run -s -n refining python -u my_Scripts/merge_chunk_refining_shards_to_one_parquet.py \
    --input-dir "${SAVE_BASE}" \
    --output "${MERGED_OUT}"
  echo "Single merged parquet: ${MERGED_OUT}"
else
  echo "SKIP_MERGE=1 — merge skipped. Run later:"
  echo "  python my_Scripts/merge_chunk_refining_shards_to_one_parquet.py --input-dir ${SAVE_BASE} --output <path>.parquet"
fi
