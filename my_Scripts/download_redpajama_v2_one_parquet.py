#!/usr/bin/env python3
"""
Download exactly one parquet file from Hugging Face `togethercomputer/RedPajama-Data-V2`.

Note: RedPajama-V2 *document* shards on HF are `sample/documents/.../*.json.gz` (same idea as
Together's `auto_download_redpajama.sh` URLs). Parquet on this repo is auxiliary data
(minhash / duplicates). Default file is a small minhash shard (~4 MB).

Examples:
  python3 download_redpajama_v2_one_parquet.py
  python3 download_redpajama_v2_one_parquet.py \\
    --filename sample/duplicates/2023-06/0001/it_head.duplicates.parquet
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "togethercomputer/RedPajama-Data-V2"
# Small minhash shard (not full text; minhash signatures for dedup pipeline)
DEFAULT_FILE = "sample/minhash/2023-06/0005/it_head.minhash.parquet"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(os.path.join(here, "..", "data", "raw", "RedPajama-V2-one-parquet"))

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--filename", default=DEFAULT_FILE, help="Path inside the repo to one .parquet file")
    p.add_argument(
        "--local-dir",
        default=os.environ.get("REDPAJAMA_V2_DOWNLOAD_DIR", default_out),
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
