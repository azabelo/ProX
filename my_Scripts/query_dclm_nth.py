#!/usr/bin/env python3
"""
Print the n-th document (1-based) from local DCLM data.

Supports:
  - Baseline JSONL + Zstd (`*_processed.jsonl.zst`), e.g. after
    `download_dclm_baseline_one_shard.py` → paths like
    `global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst`.
  - Parquet (e.g. `mlfoundations/dclm-baseline-1.0-parquet`): sorted `*.parquet` under
    --data-dir, same global row index as `query_c4_nth.py`.

Set DCLM_DATA_DIR or pass --data-dir to your corpus root.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq
import zstandard as zstd


def _default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "data", "raw", "DCLM-baseline-one-shard")
    )


def _collect_parquet_paths(data_dir: str) -> list[str]:
    flat = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if flat:
        return flat
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))


def _collect_jsonl_zst_paths(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl.zst"), recursive=True))


def _get_nth_parquet(data_dir: str, n_one_based: int) -> tuple[dict, str, int]:
    if n_one_based < 1:
        raise ValueError("n must be >= 1 (1 = first document).")

    paths = _collect_parquet_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"No parquet files under {data_dir!r}")

    target = n_one_based - 1
    offset = target
    for path in paths:
        pf = pq.ParquetFile(path)
        nrows = pf.metadata.num_rows
        if offset >= nrows:
            offset -= nrows
            continue

        row_in_shard = offset
        columns = pf.schema_arrow.names
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
            return row, os.path.relpath(path, data_dir), row_in_shard

    total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    raise IndexError(f"n={n_one_based} is past end of corpus (total rows ≈ {total}).")


def _get_nth_jsonl_zst(data_dir: str, n_one_based: int) -> tuple[dict, str, int]:
    if n_one_based < 1:
        raise ValueError("n must be >= 1 (1 = first document).")

    paths = _collect_jsonl_zst_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"No *.jsonl.zst under {data_dir!r}")

    want = n_one_based - 1
    seen = 0
    for path in paths:
        rel = os.path.relpath(path, data_dir)
        doc_idx_in_file = 0
        with zstd.open(open(path, "rb"), "rt", encoding="utf-8", errors="replace") as f:
            for line_i, line in enumerate(f):
                if not line.strip():
                    continue
                if seen == want:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {path!r} line {line_i}: {e}") from e
                    if not isinstance(obj, dict):
                        obj = {"_value": obj}
                    return obj, rel, doc_idx_in_file
                seen += 1
                doc_idx_in_file += 1

    raise IndexError(f"n={n_one_based} is past end of corpus (total documents ≈ {seen}).")


def _main_text(record: dict) -> str | None:
    for key in ("text", "content", "raw_content"):
        v = record.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "n",
        type=int,
        help="Document number (1-based) across sorted shards / files.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DCLM_DATA_DIR", _default_data_dir()),
        help="Corpus root (nested *.jsonl.zst or *.parquet).",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "parquet", "jsonl.zst"),
        default="auto",
        help="auto: parquet if any *.parquet exists under data-dir, else jsonl.zst",
    )
    parser.add_argument(
        "--text-chars",
        type=int,
        default=0,
        help="Truncate printed text (0 = full text).",
    )
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    if fmt == "auto":
        fmt = "parquet" if _collect_parquet_paths(data_dir) else "jsonl.zst"

    try:
        if fmt == "parquet":
            record, shard, idx_in_shard = _get_nth_parquet(data_dir, args.n)
        else:
            record, shard, idx_in_shard = _get_nth_jsonl_zst(data_dir, args.n)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except IndexError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not isinstance(record, dict):
        record = {"_value": record}

    text = _main_text(record)
    if text is None:
        text = json.dumps(record, indent=2, ensure_ascii=False, default=str)

    if args.text_chars and len(text) > args.text_chars:
        text = text[: args.text_chars] + " …"

    print("=" * 72)
    print(f"n (1-based):    {args.n}")
    print(f"format:         {fmt}")
    print(f"shard:          {shard}")
    print(f"row in shard:   {idx_in_shard}")
    for k, v in sorted(record.items()):
        if k in ("text", "content", "raw_content"):
            continue
        s = v if isinstance(v, (str, int, float, bool)) or v is None else repr(v)
        if isinstance(s, str) and len(s) > 200:
            s = s[:200] + " …"
        print(f"{k}: {s}")
    print("-" * 72)
    print(text)
    print("=" * 72)


if __name__ == "__main__":
    main()
