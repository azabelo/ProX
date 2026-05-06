#!/usr/bin/env python3
"""
Print the n-th document (1-based) from local RedPajama data.

Supports:
  - Parquet shards (e.g. gair-prox/RedPajama-pro `data/*.parquet`): same ordering idea as
    `query_fineweb_place.py` — sorted shard filenames, global row index.
  - RedPajama-V2-style `*.json.gz` (JSON Lines, one object per line), e.g. Together/HF
    `sample/documents/**/*.json.gz` — uses column `raw_content` when present (see
    `train/data_tokenize/prepare_web.py`), else `text` / `content`.

Set REDPAJAMA_DATA_DIR or pass --data-dir to your corpus root or a single subfolder of shards.

Minhash parquet (RedPajama-V2 `*.minhash.parquet`) has no text; if `sample/documents/**` exists
under the same HF checkout (or you pass --documents-root), this script loads the matching
`raw_content` from the referenced `*.json.gz` shard.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys

import pyarrow.parquet as pq


def _default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "data", "raw", "redpj_400B"))


def _collect_parquet_paths(data_dir: str) -> list[str]:
    flat = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if flat:
        return flat
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))


def _collect_json_gz_paths(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.json.gz"), recursive=True))


def _get_nth_parquet(data_dir: str, n_one_based: int) -> tuple[dict, str, int, str]:
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
            return row, os.path.basename(path), row_in_shard, path

    total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    raise IndexError(f"n={n_one_based} is past end of corpus (total rows ≈ {total}).")


def _get_nth_json_gz(data_dir: str, n_one_based: int) -> tuple[dict, str, int, None]:
    if n_one_based < 1:
        raise ValueError("n must be >= 1 (1 = first document).")

    paths = _collect_json_gz_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"No *.json.gz files under {data_dir!r}")

    want = n_one_based - 1
    seen = 0
    for path in paths:
        rel = os.path.relpath(path, data_dir)
        doc_idx_in_file = 0
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
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
                    return obj, rel, doc_idx_in_file, None
                seen += 1
                doc_idx_in_file += 1

    raise IndexError(f"n={n_one_based} is past end of corpus (total documents ≈ {seen}).")


def _main_text(record: dict) -> str | None:
    for key in ("raw_content", "text", "content", "body"):
        v = record.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _is_minhash_row(record: dict) -> bool:
    if not isinstance(record, dict) or "shard_id" not in record:
        return False
    return any(k.startswith("signature_sim") for k in record)


def _doc_line_index_from_minhash_id(id_str: str | None, fallback: int) -> int:
    """Parse trailing /{int} from minhash id, e.g. .../it_head.json.gz/0 -> 0."""
    if isinstance(id_str, str):
        m = re.search(r"/(\d+)$", id_str)
        if m:
            return int(m.group(1))
    return fallback


def _find_document_json_gz(shard_id: str, parquet_path: str, documents_root: str | None) -> str | None:
    """Return path to sample/documents/{shard_id} if it exists on disk."""
    if documents_root:
        dr = os.path.normpath(os.path.expanduser(documents_root))
        candidates = [
            os.path.join(dr, shard_id),
            os.path.join(dr, "sample", "documents", shard_id),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
        return None

    rel = os.path.join("sample", "documents", shard_id)
    p = os.path.abspath(os.path.dirname(parquet_path))
    for _ in range(16):
        cand = os.path.join(p, rel)
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


def _read_json_gz_line_by_doc_index(json_gz_path: str, doc_index: int) -> dict:
    """Nth non-empty JSON line (0-based doc_index)."""
    seen = 0
    with gzip.open(json_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if seen == doc_index:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return {"_value": obj}
                return obj
            seen += 1
    raise IndexError(
        f"Document index {doc_index} not found in {json_gz_path!r} (only {seen} documents)."
    )


def _resolve_minhash_to_document(
    record: dict, parquet_path: str, idx_in_shard: int, documents_root: str | None
) -> tuple[dict | None, str | None]:
    """If minhash row, load matching document JSON; else (None, None)."""
    if not _is_minhash_row(record):
        return None, None
    shard_id = record.get("shard_id")
    if not isinstance(shard_id, str):
        return None, None
    doc_idx = _doc_line_index_from_minhash_id(record.get("id"), idx_in_shard)
    jpath = _find_document_json_gz(shard_id, parquet_path, documents_root)
    if not jpath:
        return None, None
    doc = _read_json_gz_line_by_doc_index(jpath, doc_idx)
    return doc, jpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "n",
        type=int,
        help="Document number (1-based: 1 = first document across sorted shards).",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("REDPAJAMA_DATA_DIR", _default_data_dir()),
        help="Root folder containing parquet shards or nested *.json.gz (default: REDPAJAMA_DATA_DIR or ./data/raw/redpj_400B).",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "parquet", "json.gz"),
        default="auto",
        help="auto: use parquet if any *.parquet exists under data-dir, else json.gz",
    )
    parser.add_argument(
        "--text-chars",
        type=int,
        default=0,
        help="Truncate main text field to this many characters (0 = no truncation).",
    )
    parser.add_argument(
        "--documents-root",
        default=os.environ.get("REDPAJAMA_DOCUMENTS_ROOT"),
        help="Folder that contains `sample/documents/` tree (HF RedPajama-Data-V2 checkout), "
        "or the `sample/documents` directory itself. Used to resolve minhash rows to raw text.",
    )
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    if fmt == "auto":
        has_pq = bool(_collect_parquet_paths(data_dir))
        fmt = "parquet" if has_pq else "json.gz"

    try:
        if fmt == "parquet":
            record, shard, idx_in_shard, parquet_path = _get_nth_parquet(data_dir, args.n)
        else:
            record, shard, idx_in_shard, parquet_path = _get_nth_json_gz(data_dir, args.n)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except IndexError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    resolved_from: str | None = None
    display_record = record
    if fmt == "parquet" and isinstance(record, dict) and parquet_path:
        doc, jpath = _resolve_minhash_to_document(
            record, parquet_path, idx_in_shard, args.documents_root
        )
        if doc is not None:
            display_record = doc
            resolved_from = jpath

    text = _main_text(display_record) if isinstance(display_record, dict) else None
    if text is None and isinstance(display_record, dict):
        text = json.dumps(display_record, indent=2, ensure_ascii=False, default=str)
    elif text is None:
        text = str(display_record)

    if args.text_chars and len(text) > args.text_chars:
        text = text[: args.text_chars] + " …"

    print("=" * 72)
    print(f"n (1-based):    {args.n}")
    print(f"format:         {fmt}")
    print(f"shard:          {shard}")
    print(f"index in shard: {idx_in_shard}")
    if resolved_from:
        print(f"text source:    {resolved_from}")
    elif fmt == "parquet" and isinstance(record, dict) and _is_minhash_row(record):
        print(
            "text source:    (minhash only — download matching `sample/documents/**` from "
            "togethercomputer/RedPajama-Data-V2 or set --documents-root / REDPAJAMA_DOCUMENTS_ROOT)",
            file=sys.stderr,
        )
    if isinstance(display_record, dict):
        for k, v in display_record.items():
            if k in ("raw_content", "text", "content", "body"):
                continue
            if isinstance(v, (list, tuple)) and len(v) > 8:
                s = f"<{type(v).__name__} len={len(v)}>"
            elif isinstance(v, (bytes, bytearray)):
                s = f"<{len(v)} bytes>"
            else:
                s = v if isinstance(v, (str, int, float, bool)) or v is None else repr(v)
            if isinstance(s, str) and len(s) > 200:
                s = s[:200] + " …"
            print(f"{k}: {s}")
    print("-" * 72)
    print(text)
    print("=" * 72)


if __name__ == "__main__":
    main()
