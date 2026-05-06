#!/usr/bin/env python3
"""
Print one document from local FineWeb parquet shards by global position.

Shards are read in sorted filename order (000_00000.parquet, 001_..., …).
The n-th document is 1-based: 1 = first row of the first shard.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pyarrow.parquet as pq


def _default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "data", "raw", "HuggingFaceFW", "fineweb", "sample", "10BT")
    )


def _get_nth_row(data_dir: str, n_one_based: int, columns: list[str]) -> tuple[dict, str, int]:
    """Return (row dict, shard basename, row index within shard). n_one_based >= 1."""
    if n_one_based < 1:
        raise ValueError("n must be >= 1 (1 = first document).")

    target = n_one_based - 1
    paths = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {data_dir!r}")

    offset = target
    for path in paths:
        pf = pq.ParquetFile(path)
        nrows = pf.metadata.num_rows
        if offset >= nrows:
            offset -= nrows
            continue

        row_in_shard = offset
        scanned = 0
        for rg in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            if row_in_shard >= scanned + rg_rows:
                scanned += rg_rows
                continue
            local = row_in_shard - scanned
            table = pf.read_row_group(rg, columns=columns)
            row_table = table.slice(local, 1)
            row = {c: row_table.column(c)[0].as_py() for c in columns}
            return row, os.path.basename(path), row_in_shard

    total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    raise IndexError(
        f"n={n_one_based} is past end of dataset (total documents ≈ {total})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the n-th document (1-based) from FineWeb parquet shards."
    )
    parser.add_argument(
        "n",
        type=int,
        help="Document number to print (1-based: 1 = first document across sorted shards).",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("FINEWEB_DATA_DIR", _default_data_dir()),
        help="Directory containing FineWeb *.parquet shards.",
    )
    parser.add_argument(
        "--text-chars",
        type=int,
        default=0,
        help="Truncate printed text to this many characters (0 = full text, default: 0).",
    )
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    cols = [
        "text",
        "id",
        "dump",
        "url",
        "date",
        "token_count",
        "language",
        "language_score",
    ]

    try:
        row, shard, idx_in_shard = _get_nth_row(data_dir, args.n, cols)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except IndexError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    text = row["text"] or ""
    if args.text_chars and len(text) > args.text_chars:
        text = text[: args.text_chars] + " …"

    print("=" * 72)
    print(f"n (1-based):    {args.n}")
    print(f"shard:          {shard}")
    print(f"row in shard:   {idx_in_shard}")
    print(f"id:             {row['id']}")
    print(f"dump:           {row['dump']}")
    print(f"url:            {row['url']}")
    print(f"date:           {row['date']}")
    print(f"token_count:    {row['token_count']}")
    print(f"language:       {row['language']} ({row['language_score']:.4f})")
    print("-" * 72)
    print(text)
    print("=" * 72)


if __name__ == "__main__":
    main()
