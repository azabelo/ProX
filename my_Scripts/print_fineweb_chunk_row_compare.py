#!/usr/bin/env python3
"""
Print global row index ``n`` from the original FineWeb shard parquet and the same
index from the merged chunk-refined parquet (merge order: rank 0 shards, then rank 1, …).

Examples::

  conda run -n refining python my_Scripts/print_fineweb_chunk_row_compare.py -n 0
  conda run -n refining python my_Scripts/print_fineweb_chunk_row_compare.py -n 42 --text-only
  conda run -n refining python my_Scripts/print_fineweb_chunk_row_compare.py --row 42 --max-chars 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO_DEFAULT_ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "data/raw/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
)
REPO_DEFAULT_REFINED = (
    Path(__file__).resolve().parents[1]
    / "data/raw/chunk_refining_fineweb_first_parquet/fineweb_sample_10BT_chunk_refined.parquet"
)


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path.as_posix()).metadata.num_rows


def read_one_row_dict(path: Path, index: int) -> dict:
    if index < 0:
        raise IndexError(index)
    pf = pq.ParquetFile(path.as_posix())
    remain = index
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if remain < n:
            tbl = pf.read_row_group(rg)
            return tbl.slice(remain, 1).to_pylist()[0]
        remain -= n
    raise IndexError(f"row {index} out of range (file has {pf.metadata.num_rows} rows)")


def _truncate(s: str, max_chars: int | None) -> str:
    if max_chars is None or len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n… [{len(s) - max_chars} more chars truncated]"


def _pick_text(row: dict) -> str:
    if "text" in row and row["text"] is not None:
        return str(row["text"])
    return ""


def _format_val(v, max_chars: int | None) -> str:
    if v is None:
        return "(null)"
    if isinstance(v, (dict, list)):
        try:
            s = json.dumps(v, indent=2, ensure_ascii=False)
        except TypeError:
            s = repr(v)
        return _truncate(s, max_chars)
    s = str(v)
    return _truncate(s, max_chars)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-n",
        "--row",
        type=int,
        required=True,
        help="Global row index (0-based). Same index in original shard and merged refined file.",
    )
    ap.add_argument(
        "--original",
        type=Path,
        default=REPO_DEFAULT_ORIGINAL,
        help="Original FineWeb parquet (default: sample/10BT first shard).",
    )
    ap.add_argument(
        "--refined",
        type=Path,
        default=REPO_DEFAULT_REFINED,
        help="Merged chunk-refined parquet (default: repo chunk_refining_fineweb path).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Max characters per long string field (default 8000); use 0 for no limit.",
    )
    ap.add_argument(
        "-t",
        "--text-only",
        action="store_true",
        help="Print only original `text` then refined `text` (two blocks, no paths/columns).",
    )
    args = ap.parse_args()
    max_chars: int | None = None if args.max_chars == 0 else args.max_chars

    orig_path = args.original.expanduser().resolve()
    ref_path = args.refined.expanduser().resolve()
    if not orig_path.is_file():
        raise SystemExit(f"Missing original parquet: {orig_path}")
    if not ref_path.is_file():
        raise SystemExit(f"Missing refined parquet: {ref_path}")

    n_orig = parquet_row_count(orig_path)
    n_ref = parquet_row_count(ref_path)
    if args.row >= n_orig:
        raise SystemExit(f"--row {args.row} out of range for original (rows={n_orig})")
    if args.row >= n_ref:
        raise SystemExit(
            f"--row {args.row} out of range for refined (rows={n_ref}). "
            f"Original has {n_orig} rows — partial refine or merge?"
        )

    orig = read_one_row_dict(orig_path, args.row)
    ref = read_one_row_dict(ref_path, args.row)

    orig_text = _pick_text(orig)
    ref_text = ref.get("text")
    ref_text = "" if ref_text is None else str(ref_text)
    ref_raw = ref.get("raw_content")
    ref_raw = None if ref_raw is None else str(ref_raw)

    if args.text_only:
        print(_truncate(orig_text, max_chars), flush=True)
        print(flush=True)
        print(_truncate(ref_text, max_chars), flush=True)
        return

    print(f"=== Original row {args.row} ===\n{orig_path}", flush=True)
    for k in sorted(orig.keys()):
        print(f"\n--- {k} ---", flush=True)
        print(_format_val(orig[k], max_chars), flush=True)

    print(f"\n\n=== Refined row {args.row} ===\n{ref_path}", flush=True)
    for k in sorted(ref.keys()):
        print(f"\n--- {k} ---", flush=True)
        print(_format_val(ref[k], max_chars), flush=True)

    if ref_raw is not None and ref_raw != orig_text:
        print(
            "\n\nNote: refined `raw_content` differs from original `text` for this row "
            "(schema / normalization); compare original `text` to refined `text` above.",
            flush=True,
        )


if __name__ == "__main__":
    main()
