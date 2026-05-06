#!/usr/bin/env python3
"""
Export chunk-refining Parquet to a directory of .txt files (same layout idea as refining_fineweb_demo):

  doc_{i:03d}_input.txt          document text before chunk ops (doc_content)
  doc_{i:03d}_model_output.txt   chunk LM program (metadata.chunk_program)
  doc_{i:03d}_refined.txt        text after execute_meta_operations
  doc_{i:03d}_doc_program.txt    doc-level program if present (else empty file)

Plus manifest.json with paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq


def _default_parquet() -> str:
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


def _default_out_dir() -> str:
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "chunk_refining_fineweb_10docs_txt",
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "--parquet",
        default=os.environ.get("CHUNK_REFINING_PARQUET", _default_parquet()),
        help="Input chunk-refining .parquet",
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("CHUNK_REFINING_TXT_DIR", _default_out_dir()),
        help="Directory to create txt files in",
    )
    args = p.parse_args()

    pq_path = os.path.expanduser(args.parquet)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    if not os.path.isfile(pq_path):
        print(f"Not found: {pq_path}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    rows = pq.read_table(pq_path).to_pylist()
    manifest = []

    for i, row in enumerate(rows):
        stem = f"doc_{i:03d}"
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        chunk_prog = meta.get("chunk_program", "") or ""
        doc_prog = meta.get("doc_program", "") or ""
        before = row.get("doc_content") or row.get("raw_content") or ""
        after = row.get("text") or ""

        paths = {
            "input": os.path.join(out_dir, f"{stem}_input.txt"),
            "model_output": os.path.join(out_dir, f"{stem}_model_output.txt"),
            "refined": os.path.join(out_dir, f"{stem}_refined.txt"),
            "doc_program": os.path.join(out_dir, f"{stem}_doc_program.txt"),
        }
        with open(paths["input"], "w", encoding="utf-8") as f:
            f.write(before)
        with open(paths["model_output"], "w", encoding="utf-8") as f:
            f.write(chunk_prog)
        with open(paths["refined"], "w", encoding="utf-8") as f:
            f.write(after)
        with open(paths["doc_program"], "w", encoding="utf-8") as f:
            f.write(doc_prog)

        manifest.append({"index": i, "files": paths})

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"source_parquet": pq_path, "n_rows": len(rows), "entries": manifest},
            f,
            indent=2,
        )
    print(f"Wrote {len(rows)} document(s) under {out_dir}")


if __name__ == "__main__":
    main()
