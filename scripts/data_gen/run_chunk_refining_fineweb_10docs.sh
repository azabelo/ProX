#!/usr/bin/env bash
# Run web-chunk-refining-lm on the first N FineWeb parquet documents (raw shards; no doc LM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

LIMIT="${LIMIT:-10}"
NGPU="${NGPU:-1}"
MODEL="${MODEL:-gair-prox/web-chunk-refining-lm}"
CONFIG="${CONFIG:-data_gen/configs/apply_chunk_refining_fineweb_direct.yaml}"

mkdir -p "${ROOT}/data/raw/chunk_refining_fineweb_10docs" logging

export NNODE="${NNODE:-1}"
export NGPU
export TOTAL_SPLIT=$((NNODE * NGPU))

echo "TOTAL_SPLIT=$TOTAL_SPLIT LIMIT=$LIMIT MODEL=$MODEL"

for i in $(seq 0 $((NGPU - 1))); do
  TOTAL_SPLIT="$TOTAL_SPLIT" NODE_GPUS="$NGPU" NODE_RANK=0 CUDA_VISIBLE_DEVICES="$i" \
    conda run -n refining \
    python -m data_gen.tasks.apply_chunk_refining \
      --data_format parquet \
      --limit "$LIMIT" \
      --model_path "$MODEL" \
      --config_path "$CONFIG" \
    > "logging/apply_chunk_refining_fw_${i}.log" 2>&1 &
done
wait
echo "Done. Logs: logging/apply_chunk_refining_fw_*.log Output: data/raw/chunk_refining_fineweb_10docs/"
