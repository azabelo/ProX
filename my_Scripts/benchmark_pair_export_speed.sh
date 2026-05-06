#!/usr/bin/env bash
# Benchmark apply_chunk_refining_pair_export variants on LIMIT docs (default 1000), 1 GPU.
# Compares wall_seconds_after_model_load from each run's stats JSON (excludes one-time model load).
#
# Usage (from ProX repo root, refining env recommended):
#   LIMIT=1000 bash my_Scripts/benchmark_pair_export_speed.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export TOTAL_SPLIT=1
export NODE_RANK=0
export NODE_GPUS=1

LIMIT="${LIMIT:-1000}"
MODEL="${MODEL:-gair-prox/web-chunk-refining-lm}"
GPU_MEM="${GPU_MEM:-0.72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
CONDA_ENV="${CONDA_ENV:-refining}"
CONFIG="${CONFIG:-data_gen/configs/apply_chunk_refining_fineweb_first_parquet_pairs.yaml}"
OUT_BASE="${BENCH_OUT:-${ROOT}/data/raw/pair_export_bench_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_BASE"

echo "OUT_BASE=$OUT_BASE  LIMIT=$LIMIT  MODEL=$MODEL  CONDA_ENV=$CONDA_ENV"
echo "Each variant uses a fresh save subdir; stats written for wall_seconds_after_model_load."
echo ""

run_one() {
  local tag="$1"
  shift
  local sdir="${OUT_BASE}/${tag}"
  mkdir -p "$sdir"
  local st="${sdir}/stats.json"
  local log="${sdir}/run.log"
  set +e
  /usr/bin/time -f "%e" -o "${sdir}/usr_time.txt" -- \
    conda run -s -n "${CONDA_ENV}" \
    python -u -m data_gen.tasks.apply_chunk_refining_pair_export \
      --data_format parquet \
      --config_path "${CONFIG}" \
      --model_path "${MODEL}" \
      --limit "${LIMIT}" \
      --gpu_memory_utilization "${GPU_MEM}" \
      --max_num_seqs "${MAX_NUM_SEQS}" \
      --save_path "${sdir}/out" \
      --stats_json "${st}" \
      $([[ "${ENFORCE_EAGER}" == "1" ]] && echo --enforce_eager) \
      "$@" \
      >"${log}" 2>&1
  local ec=$?
  set -e
  echo "$ec" >"${sdir}/exit.txt"
  if [[ "$ec" != "0" ]]; then
    echo "  [${tag}] FAIL exit=$ec  (see ${log})"
    return 0
  fi
  ST_PATH="${st}" TAG="${tag}" python3 - <<'PY'
import json, os
p = os.environ["ST_PATH"]
tag = os.environ["TAG"]
if not os.path.isfile(p):
    print(f"  [{tag}] no stats")
else:
    s = json.load(open(p))
    print(
        f"  [{tag}] wall_total_s={s.get('wall_seconds_total', 0):.2f}  "
        f"after_model_s={s.get('wall_seconds_after_model_load', 0):.2f}  "
        f"pairs={s.get('pairs', 0)}  docs={s.get('documents', 0)}"
    )
PY
}

echo "--- baseline (tqdm on, default max_model_len from model) ---"
run_one "01_baseline"

echo "--- disable tqdm + fast batch path (implicit; LIMIT<=save_interval) ---"
run_one "02_no_tqdm" --disable_tqdm

echo "--- no tqdm + higher max_model_len (fewer scheduler truncation warnings) ---"
run_one "03_no_tqdm_m4096" --disable_tqdm --max_model_len 4096

echo "--- no tqdm + stream_window 500 => two flushes for LIMIT=1000 (usually slower overhead) ---"
run_one "04_no_tqdm_sw500" --disable_tqdm --stream_window 500

echo ""
echo "Done. Full logs under: $OUT_BASE"
echo "Faster = lower wall_seconds_after_model_load (in stats) for the same pair count."
echo "If a run failed, check its run.log and exit.txt."
