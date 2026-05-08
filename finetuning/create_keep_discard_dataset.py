#!/usr/bin/env python3
"""
Build per-character keep/discard masks aligned to a **reference** corpus (raw or pipeline input).

For each document row, compare ``reference_text`` (from ``raw_input_parquet_dir`` when set, else
``data_parquet_dir``) to the OpenRouter pipeline output on the same shard row. The diff uses the
same ``difflib.SequenceMatcher(..., autojunk=False).get_opcodes()`` rule as
``openrouter_data_generation/view_nth_input_output_diff.py``: only ``equal`` spans count as
retained (1); ``delete`` / ``replace`` regions on the reference side are 0; insertions in the
revised text are ignored for the mask (they do not add entries—mask length is always
``len(reference_text)``).

OpenRouter chunk-merge markers (``PROGRAM_CHUNK_SEPARATOR``) are stripped from the revised string
before diffing, matching the viewer's OpenRouter tab.

YAML (all configuration via a single file; run from repo root or any cwd — paths resolve
relative to the YAML file's directory when not absolute)::

  python finetuning/create_keep_discard_dataset.py finetuning/example_create_keep_discard_dataset.yaml

Typical keys (merge with or without a full OpenRouter run YAML):

  - ``merge_from_openrouter_yaml``: optional path; loaded first, then the main YAML overwrites.
  - ``output_masks_dir``: directory to write ``{{shard_stem}}_keep_discard.parquet`` files (required).
  - ``data_parquet_dir`` / ``input_parquet_dir``: shard directory (required unless merged in).
  - ``data_output_dir`` / ``output_data_dir``: OpenRouter output directory (required unless merged in).
  - ``data_parquet_type`` / ``dataset_type``: fineweb | dclm | … (default fineweb).
  - ``raw_input_parquet_dir`` / ``diff_raw_parquet_dir``: optional; when set, masks index this text;
    when omitted, reference text is the pipeline input row (same as middle in the viewer).
  - ``max_parquets``: cap input shards (-1 = all).
  - ``is_code``: if true, read merged output from ``programs_delimited`` (see OpenRouter code mode).
  - ``strip_chunk_program_separators``: default true; strip ``PROGRAM_CHUNK_SEPARATOR`` before diff.
  - ``mask_column_name``: default ``char_keep_mask`` (string of ``'0'`` / ``'1'``, length = len(reference)).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openrouter_data_generation.run_openrouter_chunked import (  # noqa: E402
    PROGRAM_CHUNK_SEPARATOR,
    get_row_text_fn,
    iter_shard_rows,
    load_yaml_config,
    parse_view_paths,
    resolve_view_output_parquet,
    sorted_shard_paths,
)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def strip_openrouter_chunk_delimiters(text: str) -> str:
    if not text:
        return text
    return text.replace(PROGRAM_CHUNK_SEPARATOR, "")


def pick_openrouter_output_text(row: dict[str, Any], *, is_code: bool) -> str:
    if is_code:
        v = row.get("programs_delimited")
        if isinstance(v, str) and v.strip():
            return v
        progs = row.get("programs")
        if isinstance(progs, list):
            return PROGRAM_CHUNK_SEPARATOR.join(str(p) for p in progs)
    for k in ("text", "programs_delimited", "raw_content"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    progs = row.get("programs")
    if isinstance(progs, list):
        return "\n\n".join(str(p) for p in progs)
    return json.dumps(row, ensure_ascii=False)


def character_keep_mask(reference: str, revised: str) -> str:
    """
    Return a string of ``'0'``/``'1'`` with length ``len(reference)``.
    ``'1'`` iff that reference codepoint lies in a SequenceMatcher ``equal`` opcode vs ``revised``.
    """
    n = len(reference)
    if n == 0:
        return ""
    sm = SequenceMatcher(None, reference, revised, autojunk=False)
    parts: list[str] = ["0"] * n
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal" and i2 > i1:
            width = i2 - i1
            parts[i1:i2] = ["1"] * width
    return "".join(parts)


def _shard_num_rows(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path.as_posix()).metadata.num_rows
    # jsonl.zst: fall back to counting via iter (rare for this script)
    n = 0
    for _ in iter_shard_rows(path):
        n += 1
    return n


def _output_parquet_num_rows(path: Path) -> int:
    return pq.ParquetFile(path.as_posix()).metadata.num_rows


def _nth_row_parquet(path: Path, index: int) -> dict[str, Any]:
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
    raise IndexError(f"row {index} out of range")


def _nth_row_shard(path: Path, index: int) -> dict[str, Any]:
    if path.suffix.lower() == ".parquet":
        return _nth_row_parquet(path, index)
    for i, row in enumerate(iter_shard_rows(path)):
        if i == index:
            return row
    raise IndexError(f"row {index} out of range in {path}")


@dataclass(frozen=True)
class KeepDiscardJobConfig:
    output_masks_dir: Path
    data_parquet_dir: Path
    data_output_dir: Path
    data_parquet_type: str
    max_parquets: int
    raw_parquet_dir: Path | None
    is_code: bool
    strip_chunk_separators: bool
    mask_column_name: str

    @staticmethod
    def from_merged_yaml(raw: dict[str, Any], yaml_path: Path) -> KeepDiscardJobConfig:
        ym = yaml_path.parent

        def P(key: str, default: Any = None) -> Any:
            return raw[key] if key in raw else default

        merge_path = P("merge_from_openrouter_yaml")
        if merge_path:
            mp = Path(str(merge_path).strip())
            if not mp.is_absolute():
                mp = (ym / mp).resolve()
            base = load_yaml_config(mp)
            raw = _deep_merge(base, raw)

        outp = P("output_masks_dir") or P("output_mask_dir")
        if not outp:
            raise SystemExit("YAML must set output_masks_dir (directory for *_keep_discard.parquet).")

        out_dir = Path(os.path.expanduser(str(outp)))
        if not out_dir.is_absolute():
            out_dir = (ym / out_dir).resolve()
        else:
            out_dir = out_dir.resolve()

        vp = parse_view_paths(raw)
        strip_sep = P("strip_chunk_program_separators")
        if strip_sep is None:
            strip_sep = True
        mask_col = str(P("mask_column_name") or "char_keep_mask")

        raw_is_code = raw.get("is_code")
        if raw_is_code is None:
            raw_is_code = raw.get("output_is_code", False)

        def _bool(v: Any, default: bool = False) -> bool:
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            return s in ("1", "true", "yes", "y", "on")

        return KeepDiscardJobConfig(
            output_masks_dir=out_dir,
            data_parquet_dir=vp.data_parquet_dir,
            data_output_dir=vp.data_output_dir,
            data_parquet_type=vp.data_parquet_type,
            max_parquets=vp.max_parquets,
            raw_parquet_dir=vp.raw_parquet_dir,
            is_code=_bool(raw_is_code, False),
            strip_chunk_separators=_bool(strip_sep, True),
            mask_column_name=mask_col,
        )


def iter_paired_shard_rows(
    cfg: KeepDiscardJobConfig,
) -> Iterator[tuple[Path, Path, Path | None, int, dict[str, Any], dict[str, Any]]]:
    """
    Yield ``(shard_path, out_path, raw_shard_or_none, local_i, reference_row, openrouter_row)``.
    Only rows present in both input shard and output parquet (``min(n_in, n_out)`` per shard).
    """
    all_shards = sorted_shard_paths(cfg.data_parquet_dir)
    if not all_shards:
        raise SystemExit(f"No shards under {cfg.data_parquet_dir}")
    shards = (
        all_shards[: cfg.max_parquets] if cfg.max_parquets >= 0 else all_shards
    )

    for shard_path in shards:
        out_path = resolve_view_output_parquet(cfg.data_output_dir, shard_path.stem)
        n_in = _shard_num_rows(shard_path)
        n_out = _output_parquet_num_rows(out_path)
        n_rows = min(n_in, n_out)
        if n_rows < n_in:
            print(
                f"[warn] {shard_path.name}: input {n_in} rows vs output {n_out}; using {n_rows} paired.",
                file=sys.stderr,
            )

        raw_shard: Path | None = None
        if cfg.raw_parquet_dir is not None:
            raw_shard_path = cfg.raw_parquet_dir / shard_path.name
            if raw_shard_path.is_file():
                raw_shard = raw_shard_path
            else:
                print(
                    f"[warn] raw_input shard missing for {shard_path.name!r}; "
                    f"fell back to pipeline input rows as reference.",
                    file=sys.stderr,
                )

        for local_i in range(n_rows):
            in_row = _nth_row_shard(shard_path, local_i)
            ref_row = (
                _nth_row_shard(raw_shard, local_i)
                if raw_shard is not None
                else in_row
            )
            out_row = _nth_row_parquet(out_path, local_i)
            yield shard_path, out_path, raw_shard, local_i, ref_row, out_row


def write_shard_masks(
    cfg: KeepDiscardJobConfig,
    shard_path: Path,
    *,
    rows_data: list[dict[str, Any]],
) -> Path:
    cfg.output_masks_dir.mkdir(parents=True, exist_ok=True)
    stem = shard_path.stem
    out_path = cfg.output_masks_dir / f"{stem}_keep_discard.parquet"
    table = pa.Table.from_pylist(rows_data)
    pq.write_table(table, out_path.as_posix(), compression="zstd")
    return out_path


def run(cfg_yaml: Path) -> None:
    os.chdir(_REPO_ROOT)
    cfg_yaml = cfg_yaml.expanduser().resolve()
    root = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise SystemExit("YAML root must be a mapping.")
    cfg = KeepDiscardJobConfig.from_merged_yaml(root, cfg_yaml)
    cfg.output_masks_dir.mkdir(parents=True, exist_ok=True)

    row_text_fn = get_row_text_fn(cfg.data_parquet_type)
    cur_shard: Path | None = None
    bucket: list[dict[str, Any]] = []

    def flush_shard() -> None:
        nonlocal bucket, cur_shard
        if cur_shard is None or not bucket:
            bucket = []
            return
        op = write_shard_masks(cfg, cur_shard, rows_data=bucket)
        print(f"[write] {op} ({len(bucket)} rows)", flush=True)
        bucket = []

    for shard_path, _out_used, raw_shard, local_i, ref_row, out_row in iter_paired_shard_rows(
        cfg
    ):
        if cur_shard != shard_path:
            flush_shard()
            cur_shard = shard_path

        reference_text = row_text_fn(ref_row)
        if not isinstance(reference_text, str):
            reference_text = ""

        revised = pick_openrouter_output_text(out_row, is_code=cfg.is_code)
        if cfg.strip_chunk_separators:
            revised = strip_openrouter_chunk_delimiters(revised)

        mask_s = character_keep_mask(reference_text, revised)
        if len(mask_s) != len(reference_text):
            raise RuntimeError("internal: mask length != reference length")

        record: dict[str, Any] = {
            cfg.mask_column_name: mask_s,
            "reference_char_len": len(reference_text),
            "source_parquet": out_row.get("source_parquet", shard_path.name),
            "source_row_index": int(
                out_row.get("source_row_index", local_i)
                if out_row.get("source_row_index") is not None
                else local_i
            ),
            "input_shard_path": shard_path.as_posix(),
            "reference_shard_used": (
                raw_shard.as_posix() if raw_shard is not None else shard_path.as_posix()
            ),
        }
        bucket.append(record)

    flush_shard()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "yaml_config",
        type=Path,
        help="YAML with merged OpenRouter paths + output_masks_dir and options.",
    )
    args = ap.parse_args()
    run(args.yaml_config)


if __name__ == "__main__":
    main()
