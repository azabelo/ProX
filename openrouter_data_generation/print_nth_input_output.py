#!/usr/bin/env python3
"""
Print row ``n``: optional raw corpus, OpenRouter input row (middle), and matching OpenRouter output,
using the same paths and shard order as ``run_openrouter_chunked.py``.

``n`` is a **global** 0-based index across **paired** rows per shard: only indices that exist in
both the input shard and the matching output parquet (``min(input_rows, output_rows)`` per shard).
If the pipeline stopped early, extra input-only rows are skipped with a stderr warning when scanning.

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
from typing import Any, Iterator

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


def _output_parquet_num_rows(path: Path) -> int:
    return pq.ParquetFile(path.as_posix()).metadata.num_rows


def _paired_shard_row_count(vp: Any, roc: Any, shard_path: Path) -> int:
    """Rows comparable for input vs output (same local index): ``min(input_rows, output_rows)``."""
    out_path = roc.resolve_view_output_parquet(vp.data_output_dir, shard_path.stem)
    n_in = _shard_num_rows(shard_path, roc)
    n_out = _output_parquet_num_rows(out_path)
    return min(n_in, n_out)


def _resolve_shard_and_local_paired(
    shard_paths: list[Path],
    global_idx: int,
    vp: Any,
    roc: Any,
) -> tuple[Path, int]:
    """Map global index using **paired** row counts per shard (aligned input/output rows only)."""
    if global_idx < 0:
        raise IndexError("global row index must be non-negative")
    rem = global_idx
    for sp in shard_paths:
        nr = _paired_shard_row_count(vp, roc, sp)
        if rem < nr:
            return sp, rem
        rem -= nr
    total = sum(_paired_shard_row_count(vp, roc, p) for p in shard_paths)
    raise IndexError(
        f"global row {global_idx} out of range "
        f"(≈{total} paired input/output rows across {len(shard_paths)} shard(s))"
    )


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


def _row_triplet_for_shard_local(
    vp: Any,
    roc: Any,
    shard_path: Path,
    local_i: int,
) -> tuple[str, str, str]:
    """``(raw_text, middle_text, openrouter_text)`` for one shard row."""
    row_text_fn = roc.get_row_text_fn(vp.data_parquet_type)
    in_row = _nth_row_shard(shard_path, local_i, roc)
    middle_text = row_text_fn(in_row)
    if vp.raw_parquet_dir is not None:
        raw_shard = vp.raw_parquet_dir / shard_path.name
        if raw_shard.is_file():
            raw_row = _nth_row_shard(raw_shard, local_i, roc)
            raw_text = row_text_fn(raw_row)
        else:
            raw_text = middle_text
    else:
        raw_text = middle_text

    out_path = roc.resolve_view_output_parquet(vp.data_output_dir, shard_path.stem)
    out_row = _nth_row_parquet(out_path, local_i)
    openrouter_text = _pick_output_text(out_row)
    return raw_text, middle_text, openrouter_text


def iter_view_yaml_rows(yaml_config: Path) -> Iterator[tuple[int, str, str, str, dict[str, Any]]]:
    """
    Yield ``(global_idx, raw_text, middle_text, openrouter_text, meta)`` for every row in
    YAML-resolved shard order (same as ``load_input_output_for_yaml_row``).
    """
    roc = _load_run_module()
    os.chdir(_REPO_ROOT)
    cfg_path = yaml_config.expanduser().resolve()
    raw_cfg = roc.load_yaml_config(cfg_path)
    vp = roc.parse_view_paths(raw_cfg)

    all_shards = roc.sorted_shard_paths(vp.data_parquet_dir)
    if not all_shards:
        raise SystemExit(f"No .parquet or *.jsonl.zst under {vp.data_parquet_dir}")
    shard_paths = (
        all_shards[: vp.max_parquets] if vp.max_parquets >= 0 else all_shards
    )

    global_idx = 0
    for shard_path in shard_paths:
        out_path = roc.resolve_view_output_parquet(vp.data_output_dir, shard_path.stem)
        n_in = _shard_num_rows(shard_path, roc)
        n_out = _output_parquet_num_rows(out_path)
        n_rows = min(n_in, n_out)
        if n_rows < n_in:
            print(
                f"[warn] {shard_path.name}: input has {n_in} rows but {out_path.name} has {n_out}; "
                f"using first {n_rows} paired row(s) only.",
                file=sys.stderr,
            )
        for local_i in range(n_rows):
            raw_text, middle_text, openrouter_text = _row_triplet_for_shard_local(
                vp, roc, shard_path, local_i
            )
            meta: dict[str, Any] = {
                "global_idx": global_idx,
                "local_row": local_i,
                "shard_path": shard_path,
                "output_parquet": out_path,
                "yaml_config": cfg_path,
                "raw_parquet_dir": vp.raw_parquet_dir,
            }
            yield global_idx, raw_text, middle_text, openrouter_text, meta
            global_idx += 1


def load_input_output_for_yaml_row(
    yaml_config: Path, n: int, *, one_based: bool
) -> tuple[str, str, str, dict[str, Any]]:
    """
    Resolve ``(raw_text, middle_text, openrouter_text, meta)`` for the global row.

    - ``middle_text``: row text from ``data_parquet_dir`` / ``input_parquet_dir`` (OpenRouter input).
    - ``openrouter_text``: matching row from the output parquet.
    - ``raw_text``: optional earlier corpus when YAML sets ``diff_raw_parquet_dir`` /
      ``raw_input_parquet_dir`` (same shard basename + row index); otherwise ``raw_text == middle_text``.

    ``meta`` includes paths and indices.
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
    shard_path, local_i = _resolve_shard_and_local_paired(shard_paths, idx, vp, roc)

    out_path = roc.resolve_view_output_parquet(vp.data_output_dir, shard_path.stem)
    raw_text, middle_text, openrouter_text = _row_triplet_for_shard_local(
        vp, roc, shard_path, local_i
    )
    meta: dict[str, Any] = {
        "global_idx": idx,
        "local_row": local_i,
        "shard_path": shard_path,
        "output_parquet": out_path,
        "yaml_config": cfg_path,
        "raw_parquet_dir": vp.raw_parquet_dir,
    }
    return raw_text, middle_text, openrouter_text, meta


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
        raw_text, middle_text, out_text, meta = load_input_output_for_yaml_row(
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
    print(f"=== MIDDLE (OpenRouter input) global_row={idx} local_row={local_i} ===\n{shard_path}\n")
    print(middle_text)
    print(sep)
    if raw_text != middle_text:
        print(
            f"=== RAW (pre-middle) global_row={idx} local_row={local_i} ===\n"
            f"{meta.get('raw_parquet_dir')}\n"
        )
        print(raw_text)
        print(sep)
    print(f"=== OPENROUTER OUTPUT global_row={idx} local_row={local_i} ===\n{out_path}\n")
    print(out_text)


if __name__ == "__main__":
    main()
