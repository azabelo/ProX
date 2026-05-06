"""
Stage 3: refiner_inference

Runs the (fine-tuned) chunk refiner model to generate programs per chunk, executes
those programs, and writes cleaned text to parquet.

Thin wrapper around:
  `python -m data_gen.tasks.apply_chunk_refining`

Example (single GPU, 10 docs):

  TOTAL_SPLIT=1 CUDA_VISIBLE_DEVICES=0 \
    conda run -n refining \
    python -u refiner_inference/main.py \
      --config_path data_gen/configs/apply_chunk_refining_fineweb_first_parquet_ft.yaml \
      --model_path /path/to/finetuned/hf_ckpt \
      --data_format parquet \
      --limit 10
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    # Ensure repo root is importable when run via `conda run` without PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    runpy.run_module("data_gen.tasks.apply_chunk_refining", run_name="__main__")


if __name__ == "__main__":
    main()

