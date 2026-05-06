#!/usr/bin/env python3
"""
Download a single data shard from Hugging Face `mlfoundations/dclm-baseline-1.0`.

This dataset does **not** use Parquet: each slice is a Zstandard-compressed JSONL file
(`*_processed.jsonl.zst`) under paths such as:

  global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst

So "one shard" here means one `.jsonl.zst` file (same role as a single Parquet part in other corpora).

Examples:
  python3 download_dclm_baseline_one_shard.py
  python3 download_dclm_baseline_one_shard.py \\
    --filename global-shard_01_of_10/local-shard_0_of_10/shard_00000016_processed.jsonl.zst
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "mlfoundations/dclm-baseline-1.0"
DEFAULT_FILE = (
    "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"
)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(os.path.join(here, "..", "data", "raw", "DCLM-baseline-one-shard"))

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--filename", default=DEFAULT_FILE, help="One path inside the repo")
    p.add_argument(
        "--local-dir",
        default=os.environ.get("DCLM_BASELINE_DOWNLOAD_DIR", default_out),
        help="Root directory for mirrored paths",
    )
    args = p.parse_args()

    path = hf_hub_download(
        repo_id=args.repo,
        repo_type="dataset",
        filename=args.filename,
        local_dir=args.local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
