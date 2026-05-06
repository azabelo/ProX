#!/usr/bin/env python3
"""
Print row ``n`` from the input corpus and the matching row from the OpenRouter output,
using the same paths and shard order as ``run_openrouter_chunked.py``.

``n`` is a **global** 0-based document index across the selected input shards (sorted
like the runner: ``sorted_shard_paths``, then ``max_parquets`` cap). Row ``i`` in
``<stem>_openrouter.parquet`` matches row ``i`` of that input shard.

Usage (from repo root, or any cwd — the script ``chdir``s to repo root like the runner)::

  python openrouter_data_generation/print_nth_input_output.py \\
    openrouter_data_generation/example_openrouter_rewrite.yaml 0

  python openrouter_data_generation/print_nth_input_output.py config.yaml 3 --one-based
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_MOD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MOD_DIR.parent


def _load_run_module():
    name = "run_openrouter_chunked"
    path = _MOD_DIR / "run_openrouter_chunked.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _shard_num_rows(path: Path, roc) -> int:
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path.as_posix()).metadata.num_rows
    if roc._is_jsonl_zst(path):
        return roc._count_jsonl_zst_lines(path)
    raise SystemExit(f"Unsupported input shard: {path}")


def _nth_row_parquet(path: Path, index: int) -> dict:
    if index < 0:
        raise IndexError("index must be non-negative")
    pf = pq.ParquetFile(path.as_posix())
    seen = 0
    for batch in pf.iter_batches():
        n = batch.num_rows
        if seen + n <= index:
            seen += n
            continue
        j = index - seen
        names = batch.schema.names
        return {name: batch.column(names.index(name))[j].as_py() for name in names}
    raise IndexError(f"row {index} out of range (fewer than {seen + n} rows)")


def _nth_row_shard(path: Path, index: int, roc) -> dict:
    if path.suffix.lower() == ".parquet":
        return _nth_row_parquet(path, index)
    for i, row in enumerate(roc.iter_shard_rows(path)):
        if i == index:
            return row
    raise IndexError(f"row {index} out of range in {path}")


def _resolve_shard_and_local(shard_paths: list[Path], global_idx: int, roc) -> tuple[Path, int]:
    rem = global_idx
    for sp in shard_paths:
        nr = _shard_num_rows(sp, roc)
        if rem < nr:
            return sp, rem
        rem -= nr
    total = sum(_shard_num_rows(p, roc) for p in shard_paths)
    raise IndexError(f"global row {global_idx} out of range (total rows ≈ {total})")


def _pick_output_text(row: dict) -> str:
    for k in ("text", "programs_delimited", "raw_content"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    progs = row.get("programs")
    if isinstance(progs, list):
        return "\n\n".join(str(p) for p in progs)
    return json.dumps(row, ensure_ascii=False, indent=2)


def load_input_output_for_yaml_row(
    yaml_config: Path, n: int, *, one_based: bool
) -> tuple[str, str, dict[str, Any]]:
    """
    Resolve the same (input text, output text) pair as ``main()`` would print.

    Returns ``(source_text, output_text, meta)`` where ``meta`` includes paths and indices.
    """
    roc = _load_run_module()
    os.chdir(_REPO_ROOT)
    cfg_path = yaml_config.expanduser().resolve()
    raw = roc.load_yaml_config(cfg_path)
    vp = roc.parse_view_paths(raw)

    all_shards = roc.sorted_shard_paths(vp.data_parquet_dir)
    if not all_shards:
        raise SystemExit(f"No .parquet or *.jsonl.zst under {vp.data_parquet_dir}")
    if vp.max_parquets >= 0:
        shard_paths = all_shards[: vp.max_parquets]
    else:
        shard_paths = all_shards

    idx = n - 1 if one_based else n
    shard_path, local_i = _resolve_shard_and_local(shard_paths, idx, roc)

    out_path = roc.resolve_view_output_parquet(vp.data_output_dir, shard_path.stem)

    row_text_fn = roc.get_row_text_fn(vp.data_parquet_type)
    in_row = _nth_row_shard(shard_path, local_i, roc)
    out_row = _nth_row_parquet(out_path, local_i)
    source_text = row_text_fn(in_row)
    out_text = _pick_output_text(out_row)
    meta: dict[str, Any] = {
        "global_idx": idx,
        "local_row": local_i,
        "shard_path": shard_path,
        "output_parquet": out_path,
        "yaml_config": cfg_path,
    }
    return source_text, out_text, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml_config", type=Path, help="Path to run_openrouter_chunked YAML.")
    ap.add_argument("n", type=int, help="Global row index (0-based unless --one-based).")
    ap.add_argument(
        "--one-based",
        action="store_true",
        help="Interpret n as 1-based (first document is 1).",
    )
    args = ap.parse_args()

    try:
        source_text, out_text, meta = load_input_output_for_yaml_row(
            args.yaml_config, args.n, one_based=args.one_based
        )
    except IndexError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    idx = meta["global_idx"]
    shard_path = meta["shard_path"]
    local_i = meta["local_row"]
    out_path = meta["output_parquet"]
    sep = "\n" * 8
    print(f"=== INPUT global_row={idx} local_row={local_i} ===\n{shard_path}\n")
    print(source_text)
    print(sep)
    print(f"=== OUTPUT global_row={idx} local_row={local_i} ===\n{out_path}\n")
    print(out_text)


if __name__ == "__main__":
    main()
