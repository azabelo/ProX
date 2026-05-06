#!/usr/bin/env python3
"""
Download exactly one Parquet file that holds one Common Crawl snapshot slice of FineWeb.

The Hugging Face dataset `HuggingFaceFW/fineweb` exposes each crawl as `data/<CC-SNAPSHOT>/*.parquet`.
Shard sizes are usually ~2 GiB; the **last** shard of older snapshots can be much smaller.

Defaults:
  - Snapshot: CC-MAIN-2013-20
  - File:     last shard `004_00004.parquet` (~0.6 MiB) so the download is quick for smoke tests.

Override with --filename or --snapshot plus --shard (e.g. first large shard `000_00000.parquet`).

Examples:
  python3 download_common_crawl_one_parquet.py
  python3 download_common_crawl_one_parquet.py --snapshot CC-MAIN-2024-51 --shard 000_00000.parquet
  python3 download_common_crawl_one_parquet.py \\
    --filename data/CC-MAIN-2013-20/004_00004.parquet
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "HuggingFaceFW/fineweb"
DEFAULT_SNAPSHOT = "CC-MAIN-2013-20"
# Small tail shard (not representative of full crawl size).
DEFAULT_SHARD = "004_00004.parquet"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(
        os.path.join(here, "..", "data", "raw", "CommonCrawl-one-parquet")
    )

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset id (default: FineWeb CC slices)")
    p.add_argument(
        "--filename",
        default=None,
        help="Full path inside repo, e.g. data/CC-MAIN-2013-20/004_00004.parquet (overrides --snapshot/--shard)",
    )
    p.add_argument(
        "--snapshot",
        default=DEFAULT_SNAPSHOT,
        help="Common Crawl snapshot id under data/, e.g. CC-MAIN-2024-51",
    )
    p.add_argument(
        "--shard",
        default=DEFAULT_SHARD,
        help="Parquet filename only, e.g. 000_00000.parquet",
    )
    p.add_argument(
        "--local-dir",
        default=os.environ.get("COMMON_CRAWL_DOWNLOAD_DIR", default_out),
        help="Root directory for mirrored paths",
    )
    args = p.parse_args()

    if args.filename:
        filename = args.filename
    else:
        filename = f"data/{args.snapshot}/{args.shard}"

    path = hf_hub_download(
        repo_id=args.repo,
        repo_type="dataset",
        filename=filename,
        local_dir=args.local_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
