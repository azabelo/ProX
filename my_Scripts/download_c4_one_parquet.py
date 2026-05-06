#!/usr/bin/env python3
"""
Download exactly one shard file for C4-style corpora on Hugging Face.

Important: the official English C4 dataset `allenai/c4` under `en/` is stored as
`*.json.gz` WebText-style shards, not Parquet (see https://huggingface.co/datasets/allenai/c4/tree/main/en).

This script defaults to **one Parquet shard** from `gair-prox/c4-pro` (processed C4 used elsewhere
in this repo), which is smaller than a full HF `en/c4-train.*.json.gz` shard.

Examples:
  # One parquet (~172 MB default shard)
  python3 download_c4_one_parquet.py

  # Official HF English C4: one JSON gzip shard (~300 MB)
  python3 download_c4_one_parquet.py \\
    --repo allenai/c4 \\
    --filename en/c4-train.00000-of-01024.json.gz

  # Another parquet from c4-pro
  python3 download_c4_one_parquet.py --filename data/000_1_7.parquet
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "gair-prox/c4-pro"
# Smallest `data/*.parquet` shard in that repo at time of writing (~172 MiB).
DEFAULT_PARQUET = "data/000_7_7.parquet"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(os.path.join(here, "..", "data", "raw", "C4-one-shard"))

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument(
        "--filename",
        default=DEFAULT_PARQUET,
        help="Single path inside the repo (parquet under gair-prox/c4-pro, or en/*.json.gz for allenai/c4)",
    )
    p.add_argument(
        "--local-dir",
        default=os.environ.get("C4_DOWNLOAD_DIR", default_out),
        help="Directory root where the file tree will be mirrored",
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
