#!/usr/bin/env python3
"""
Print chunk-refining Parquet rows as plain text: input (before), chunk program, output (after).

Expected columns: raw_content, doc_content, text, metadata.doc_program, metadata.chunk_program
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq


def _default_path() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "raw",
            "chunk_refining_fineweb_10docs",
            "prox_0",
            "prox_1_1.parquet",
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "parquet",
        nargs="?",
        default=os.environ.get("CHUNK_REFINING_PARQUET", _default_path()),
        help="Path to chunk-refining output .parquet",
    )
    p.add_argument(
        "--row",
        type=int,
        default=None,
        help="Print only this row index (0-based). Default: all rows.",
    )
    p.add_argument("--json", action="store_true", help="One JSON object per line")
    args = p.parse_args()

    path = os.path.expanduser(args.parquet)
    if not os.path.isfile(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = pq.read_table(path).to_pylist()
    if args.row is not None:
        rows = [rows[args.row]] if 0 <= args.row < len(rows) else []
        if not rows:
            print(f"Invalid --row {args.row} (table has {len(pq.read_table(path))} rows)", file=sys.stderr)
            sys.exit(2)

    for i, row in enumerate(rows):
        idx = args.row if args.row is not None else i
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        doc_prog = meta.get("doc_program", "") or ""
        chunk_prog = meta.get("chunk_program", "") or ""
        before = row.get("doc_content") or row.get("raw_content") or ""
        after = row.get("text") or ""

        if args.json:
            print(
                json.dumps(
                    {
                        "index": idx,
                        "before": before,
                        "after": after,
                        "doc_program": doc_prog,
                        "chunk_program": chunk_prog,
                    },
                    ensure_ascii=False,
                )
            )
            continue

        sep = "=" * 72
        print(sep)
        print(f"ROW {idx}")
        print(sep)
        print("--- chunk_program (model output for this doc) ---")
        print(chunk_prog or "(empty)")
        print()
        print("--- doc_program (from doc-level pass; empty if raw FineWeb) ---")
        print(doc_prog or "(empty)")
        print()
        print("--- BEFORE (doc_content / input text) ---")
        print(before or "(empty)")
        print()
        print("--- AFTER (refined text column) ---")
        print(after or "(empty)")
        print()


if __name__ == "__main__":
    main()
