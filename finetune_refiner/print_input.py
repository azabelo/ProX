"""
Stage 2 helper: print part of the *training input* parquet (chunk→program pairs).

Reuses:
  `my_Scripts/print_ft_parquet_example.py`

Example:

  python finetune_refiner/print_input.py --parquet /path/to/chunk_pairs.parquet --index 0
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=os.environ.get("FT_PARQUET", ""))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--one-based", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "my_Scripts" / "print_ft_parquet_example.py"

    argv = [str(script)]
    if args.parquet:
        argv += ["--parquet", os.path.expanduser(args.parquet)]
    argv += ["--index", str(args.index)]
    if args.one_based:
        argv += ["--one-based"]
    if args.json:
        argv += ["--json"]

    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

