#!/usr/bin/env python3
"""
Print the n-th row from a chunk-refining finetuning parquet (pair or doc-level).

Pair export (apply_chunk_refining_pair_export): columns ``text`` (chunk) and ``target`` (program).

Doc-level (apply_chunk_refining): uses ``doc_content`` / ``raw_content`` as input context and
``metadata.chunk_program`` as the model program when ``target`` is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _iter_rank_dirs(base: Path) -> list[tuple[int, Path]]:
    dirs: list[tuple[int, Path]] = []
    for p in base.iterdir():
        m = re.fullmatch(r"prox_(\d+)", p.name)
        if p.is_dir() and m:
            dirs.append((int(m.group(1)), p))
    return sorted(dirs, key=lambda x: x[0])


def _shard_files_under_dir(base: Path) -> list[Path]:
    files: list[Path] = []
    for _, d in _iter_rank_dirs(base):
        files.extend(sorted(d.glob("prox_*.parquet")))
    return files


def _resolve_parquet_sources(parquet: str | None, input_dir: str | None) -> list[Path]:
    if parquet:
        p = Path(parquet).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(
                f"Not a file: {p}\n"
                "Use a real path to your merged .parquet (e.g. from the pair-export run: "
                "<repo>/data/raw/chunk_refining_fineweb_first_parquet_pairs/fineweb_sample_10BT_chunk_pairs.parquet), "
                "or pass --input-dir <save_path> with prox_*/prox_*.parquet shards."
            )
        return [p]
    if input_dir:
        base = Path(input_dir).expanduser().resolve()
        if not base.is_dir():
            raise SystemExit(f"Not a directory: {base}")
        files = _shard_files_under_dir(base)
        if not files:
            raise SystemExit(
                f"No prox_*/prox_*.parquet under {base}. "
                "Pass --parquet to a single .parquet file instead."
            )
        return files
    raise SystemExit("Provide --parquet <file.parquet> or --input-dir <save_path>.")


def _nth_row(files: list[Path], index: int) -> dict:
    """0-based global row index across files, streaming batches."""
    if index < 0:
        raise IndexError("index must be non-negative")
    remaining = index
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches():
            n = batch.num_rows
            if n <= remaining:
                remaining -= n
                continue
            tbl = pa.Table.from_batches([batch])
            return tbl.slice(remaining, 1).to_pylist()[0]
    raise IndexError(f"row index {index} out of range")


def _program_and_chunk_text(row: dict) -> tuple[str, str]:
    """Return (chunk_or_input_text, program_string)."""
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}

    if "target" in row:
        chunk = (row.get("text") or "").strip()
        prog = (row.get("target") or "").strip()
        return chunk, prog

    chunk = (
        row.get("doc_content")
        or row.get("raw_content")
        or row.get("text")
        or ""
    ).strip()
    prog = (meta.get("chunk_program") or "").strip()
    return chunk, prog


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_merged = root / "data/raw/chunk_refining_fineweb_first_parquet_pairs/fineweb_sample_10BT_chunk_pairs.parquet"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        default=os.environ.get("FT_PARQUET"),
        help="Single .parquet file (pair-export merged or one shard).",
    )
    ap.add_argument(
        "--input-dir",
        default=None,
        help="Chunk-refining save_path with prox_*/prox_*.parquet shards (same order as merge script).",
    )
    ap.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index (0-based by default; use --one-based for 1-based).",
    )
    ap.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret --index as 1-based (first row is 1).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print the full row as JSON (still prints chunk + program first).",
    )
    args = ap.parse_args()

    pq_arg = args.parquet
    if not pq_arg and not args.input_dir and default_merged.is_file():
        pq_arg = str(default_merged)

    files = _resolve_parquet_sources(pq_arg, args.input_dir)
    idx = args.index - 1 if args.one_based else args.index

    try:
        row = _nth_row(files, idx)
    except IndexError as e:
        raise SystemExit(str(e)) from e

    chunk, program = _program_and_chunk_text(row)
    print(f"=== Row {idx} (0-based) | source: {files[0] if len(files) == 1 else f'{len(files)} shard files'} ===\n")
    print("--- Text chunk (input) ---")
    print(chunk if chunk else "(empty)")
    print("\n--- Program (target / chunk_program) ---")
    print(program if program else "(empty)")

    extra = []
    if "chunk_index" in row and row["chunk_index"] is not None:
        extra.append(f"chunk_index={row['chunk_index']}")
    dp = row.get("doc_program")
    if dp:
        extra.append("doc_program present")
    if extra:
        print("\n--- Meta ---")
        print(", ".join(extra))

    if args.json:
        print("\n--- Full row (JSON) ---")
        print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
