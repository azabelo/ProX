#!/usr/bin/env bash
# Chunk-refine all rows in the first FineWeb parquet shard (sample/10BT), multi-GPU.
# After all ranks finish, aggregate stats:  python my_Scripts/aggregate_chunk_refining_stats.py <save_path>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
# So apply_chunk_refining progress lines (every 100 docs) flush to per-rank logs / terminal.
export PYTHONUNBUFFERED=1

LIMIT="${LIMIT:--1}"
# Row-based sharding splits one .parquet across ranks (see apply_chunk_refining). Each rank loads
# a full vLLM model — use NGPU<= available free GPUs, or NGPU=1 if others are busy.
NGPU="${NGPU:-8}"
# vLLM reserves a large KV pool at init; 0.9 often OOMs on 80GB when CUDA graphs reserve extra GB.
GPU_MEM="${GPU_MEM:-0.72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
# Set ENFORCE_EAGER=1 to disable CUDA graphs (less peak VRAM) if you still OOM at model init.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MODEL="${MODEL:-gair-prox/web-chunk-refining-lm}"
CONFIG="${CONFIG:-data_gen/configs/apply_chunk_refining_fineweb_first_parquet.yaml}"

SAVE_BASE="${SAVE_BASE:-${ROOT}/data/raw/chunk_refining_fineweb_first_parquet}"
MERGED_OUT="${MERGED_OUT:-${SAVE_BASE}/fineweb_sample_10BT_chunk_refined.parquet}"
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
    > "logging/apply_chunk_refining_fineweb_fp_${i}.log" 2>&1 &
done
wait

echo "Done. Parquet shards under: ${SAVE_BASE}/prox_*/"
echo "Per-rank stats: ${SAVE_BASE}/chunk_refining_stats_rank*.json"
echo "Aggregate: python my_Scripts/aggregate_chunk_refining_stats.py ${SAVE_BASE}"

if [[ "${SKIP_MERGE}" != "1" ]]; then
  conda run -s -n refining python -u my_Scripts/merge_chunk_refining_shards_to_one_parquet.py \
    --input-dir "${SAVE_BASE}" \
    --output "${MERGED_OUT}"
  echo "Single merged parquet: ${MERGED_OUT}"
else
  echo "SKIP_MERGE=1 — merge skipped. Run later:"
  echo "  python my_Scripts/merge_chunk_refining_shards_to_one_parquet.py --input-dir ${SAVE_BASE} --output <path>.parquet"
fi
