"""
Stage 3 helper: print part of the *input* dataset (FineWeb) by global row index.

Reuses:
  `my_Scripts/print_fineweb_chunk_row_compare.py`

Examples:
  conda run -n refining python refiner_inference/print_input.py --row 0
  conda run -n refining python refiner_inference/print_input.py --row 42 --text-only
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--original", type=str, default="")
    ap.add_argument("--refined", type=str, default="")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--max-chars", type=int, default=8000)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "my_Scripts" / "print_fineweb_chunk_row_compare.py"

    argv = [str(script), "--row", str(args.row)]
    if args.original:
        argv += ["--original", os.path.expanduser(args.original)]
    if args.refined:
        argv += ["--refined", os.path.expanduser(args.refined)]
    if args.text_only:
        argv += ["--text-only"]
    argv += ["--max-chars", str(args.max_chars)]

    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

