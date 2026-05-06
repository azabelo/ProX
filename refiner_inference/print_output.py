"""
Stage 3 helper: print part of the *output* refined parquet (from chunk refining).

Reuses:
  `my_Scripts/print_chunk_refining_parquet.py`

Example:

  conda run -n refining python refiner_inference/print_output.py \
    data/raw/chunk_refining_fineweb_first_parquet_ft/prox_0/prox_000001.parquet --row 0
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=os.environ.get("CHUNK_REFINING_PARQUET", ""))
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "my_Scripts" / "print_chunk_refining_parquet.py"

    argv = [str(script)]
    if args.parquet:
        argv.append(os.path.expanduser(args.parquet))
    if args.row is not None:
        argv += ["--row", str(args.row)]
    if args.json:
        argv += ["--json"]

    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

