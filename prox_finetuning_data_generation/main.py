"""
Stage 1: prox_finetuning_data_generation

Generates chunk → program training pairs (written to parquet) by running inference
with an existing chunk model. This is a thin wrapper around:
  `python -m data_gen.tasks.apply_chunk_refining_pair_export`

Example (single GPU, 10 docs):

  # Input: FineWeb parquet shard directory (or any parquet directory the config points to)
  # Output: shard parquets under SAVE_BASE/prox_0/
  #
  # TOTAL_SPLIT/ CUDA_VISIBLE_DEVICES are required by the underlying runner.
  TOTAL_SPLIT=1 CUDA_VISIBLE_DEVICES=0 \
    conda run -n refining \
    python -u prox_finetuning_data_generation/main.py \
      --config_path data_gen/configs/apply_chunk_refining_fineweb_first_parquet_pairs.yaml \
      --model_path gair-prox/web-chunk-refining-lm \
      --data_format parquet \
      --limit 10
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    # These stage entrypoints are often run via `conda run` without PYTHONPATH set.
    # Ensure the repo root is importable so `data_gen.*` resolves.
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    runpy.run_module("data_gen.tasks.apply_chunk_refining_pair_export", run_name="__main__")


if __name__ == "__main__":
    main()

