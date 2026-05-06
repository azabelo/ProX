#!/usr/bin/env bash
# Single-node, 8-GPU doc-level refining (matches example_doc_refining.sh layout).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
mkdir -p logging "${ROOT}/data/raw/doc_refining_fineweb_8gpu"

NNODE="${NNODE:-1}"
NGPU="${NGPU:-8}"
export NNODE NGPU
export TOTAL_SPLIT=$((NNODE * NGPU))

LIMIT="${LIMIT:-2}"   # per-GPU cap (datatrove ParquetReader); total docs <= LIMIT * NGPU
MODEL_PATH="${MODEL_PATH:-gair-prox/web-doc-refining-lm}"
CONFIG="${CONFIG:-data_gen/configs/apply_doc_refining_fineweb_local.yaml}"

echo "TOTAL_SPLIT=$TOTAL_SPLIT NGPU=$NGPU LIMIT=$LIMIT"

for i in $(seq 0 $((NGPU - 1))); do
  TOTAL_SPLIT="$TOTAL_SPLIT" NODE_GPUS="$NGPU" NODE_RANK=0 CUDA_VISIBLE_DEVICES="$i" \
    conda run -n refining \
    python -m data_gen.tasks.apply_doc_refining \
      --data_format parquet \
      --limit "$LIMIT" \
      --model_path "$MODEL_PATH" \
      --config_path "$CONFIG" \
    > "logging/apply_doc_refining_${i}.log" 2>&1 &
done
wait
echo "All workers finished. Logs under logging/apply_doc_refining_*.log"
