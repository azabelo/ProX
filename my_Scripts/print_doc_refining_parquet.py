#!/usr/bin/env python3
"""
Print rows from a doc-refining output Parquet (e.g. prox_*/*.parquet) as plain text.

Columns (from apply_doc_refining): raw_content, text, metadata.doc_program
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
            "doc_refining_fineweb_8gpu",
            "prox_0",
            "prox_1_1.parquet",
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "parquet",
        nargs="?",
        default=os.environ.get("DOC_REFINING_PARQUET", _default_path()),
        help="Path to refining output .parquet",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per line instead of human-oriented blocks",
    )
    args = p.parse_args()

    path = os.path.expanduser(args.parquet)
    if not os.path.isfile(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    table = pq.read_table(path)
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        meta = row.get("metadata") or {}
        if isinstance(meta, dict):
            prog = meta.get("doc_program", "")
        else:
            prog = str(meta)

        if args.json:
            rec = {
                "index": i,
                "raw_content": row.get("raw_content", ""),
                "text": row.get("text", ""),
                "doc_program": prog,
            }
            print(json.dumps(rec, ensure_ascii=False))
            continue

        sep = "=" * 72
        print(sep)
        print(f"ROW {i}")
        print(sep)
        print("--- doc_program (model output) ---")
        print(prog or "(empty)")
        print()
        print("--- refined text (text column) ---")
        print(row.get("text", "") or "(empty)")
        print()
        print("--- raw_content (input) ---")
        print(row.get("raw_content", "") or "(empty)")
        print()


if __name__ == "__main__":
    main()
