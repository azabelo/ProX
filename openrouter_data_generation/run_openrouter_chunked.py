#!/usr/bin/env python3
"""
Chunk documents from parquet or DCLM-style ``*.jsonl.zst`` shards, send (prompt + chunk) to OpenRouter concurrently,
and write output parquet(s).

YAML keys (required unless noted):
  - model, num_concurrent, prompt_path, data_parquet_dir, data_parquet_type,
    chunk_size, max_chars, max_docs, max_parquets, data_output_dir,
    allow_output_overwrite (or typo allow_outpute_overwrite), output_is_code (true|false, or 0|1),
    continue (optional, default false): when true, if ``data_output_dir`` already contains
    the copied YAML from a prior run and all **output-affecting** fields match (see
    ``output_effect_fingerprint``), reuse existing output rows and only process remaining
    documents; ``allow_output_overwrite`` may stay false. ``max_docs`` / ``max_chars`` are
    not part of that fingerprint so you can **raise** those caps and still resume. If
    ``continue`` is true but the fingerprint differs, the run exits with an error instead
    of overwriting.
  - Optional: metrics_interval_sec, http_referer, x_title, openrouter: { ... },
    max_tokens (see below)
  - Any other top-level keys are forwarded to the OpenRouter chat/completions JSON
    (e.g. temperature, top_p). Put OpenRouter ``max_tokens`` (completion limit) under
    ``openrouter:`` so it is forwarded; **top-level** ``max_tokens`` is reserved for
    metrics only: denominator ``min(max_tokens, estimated tokens in selected parquets)``.

Limits: processing stops when max_docs, max_chars, or end of selected input shards is
reached — whichever comes first (``max_parquets`` caps how many input files are read:
``.parquet`` shards, or ``*.jsonl.zst`` DCLM-style shards, discovered under ``data_parquet_dir``).
``max_docs`` applies to started rows as well as completed ones: a row with chunk HTTP work
in flight counts toward the cap so extra rows are not queued while earlier rows finish.

The resolved YAML config path is copied into ``data_output_dir`` (same basename) at run
start so each dataset folder records how it was produced.

Concurrency: one shared pool of up to ``num_concurrent`` **chunk** HTTP calls. Chunk jobs
for a row are submitted in order (within-doc), then the next row’s chunks are queued as rows
are read (across docs). When all chunks for a row finish, the row is merged and written.
Output rows are sorted by ``source_row_index`` before writing each parquet.

Environment:
  OPENROUTER_API_KEY — required.

Usage:
  export OPENROUTER_API_KEY=...
  python openrouter_data_generation/run_openrouter_chunked.py path/to/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

import pyarrow as pa
import pyarrow.parquet as pq

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_zstandard_mod: Any = None


def _get_zstandard():
    """Lazy import: only ``*.jsonl.zst`` shards need ``zstandard`` (not plain ``*.parquet``)."""
    global _zstandard_mod
    if _zstandard_mod is None:
        try:
            import zstandard as zstd  # noqa: PLC0415

            _zstandard_mod = zstd
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "zstandard is required for *.jsonl.zst shards (not for *.parquet): pip install zstandard"
            ) from e
    return _zstandard_mod

# Delimiter between per-chunk model outputs (rewrite ``text`` and code ``programs_delimited``).
PROGRAM_CHUNK_SEPARATOR = "\n\n### OPENROUTER_CHUNK_PROGRAM ###\n\n"

# Keys consumed by this runner (not forwarded to OpenRouter JSON body)
RESERVED_KEYS = frozenset(
    {
        "model",
        "num_concurrent",
        "prompt_path",
        "data_parquet_dir",
        "data_parquet_type",
        "chunk_size",
        "max_chars",
        "max_docs",
        "max_parquets",
        "data_output_dir",
        "allow_output_overwrite",
        "allow_outpute_overwrite",  # common typo
        "output_is_code",
        "metrics_interval_sec",
        "http_referer",
        "x_title",
        "continue",
        # Progress/metrics cap: min(max_tokens, parquet token estimate). Not sent to API;
        # use openrouter.max_tokens for the OpenRouter completion budget.
        "max_tokens",
        # allow nested extras instead of flat
        "openrouter",
    }
)


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def _output_is_code_bool(v: Any) -> bool:
    """YAML-friendly: ``true``/``false`` (or legacy ``0``/``1``)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v in (0, 1):
            return bool(v)
        raise SystemExit(f"output_is_code int must be 0 or 1; got {v}")
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("0", "false", "no", "off", "n"):
            return False
        if s in ("1", "true", "yes", "on", "y"):
            return True
        raise SystemExit(f"output_is_code string must be true/false or 0/1; got {v!r}")
    raise SystemExit(f"output_is_code must be boolean or 0/1; got {type(v).__name__}: {v!r}")


def _fmt_eta_sec(sec: float) -> str:
    if not math.isfinite(sec) or sec <= 0:
        return "~0s"
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.2f}h"


def _text_fineweb(row: dict[str, Any]) -> str:
    t = row.get("text")
    return t if isinstance(t, str) else ""


def _text_redpajama(row: dict[str, Any]) -> str:
    for k in ("raw_content", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _text_passthrough(row: dict[str, Any]) -> str:
    t = row.get("text")
    if isinstance(t, str):
        return t
    return json.dumps(row, ensure_ascii=False)[:1_000_000]


ADAPTERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "fineweb": _text_fineweb,
    "fine_web": _text_fineweb,
    "dclm": _text_fineweb,
    "redpajama": _text_redpajama,
    "redpajama-v2": _text_redpajama,
    "passthrough": _text_passthrough,
}


def get_row_text_fn(data_parquet_type: str) -> Callable[[dict[str, Any]], str]:
    key = (data_parquet_type or "passthrough").strip().lower()
    if key not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise SystemExit(f"Unknown data_parquet_type={data_parquet_type!r}. Use one of: {known}")
    return ADAPTERS[key]


def sorted_shard_paths(data_dir: Path) -> list[Path]:
    """Prefer ``*.parquet`` (recursive); else ``*.jsonl.zst`` (e.g. DCLM baseline shards)."""
    parq = sorted({p for p in data_dir.rglob("*.parquet") if p.is_file()})
    if parq:
        return parq
    return sorted({p for p in data_dir.rglob("*.jsonl.zst") if p.is_file()})


def _is_jsonl_zst(path: Path) -> bool:
    return path.name.endswith(".jsonl.zst")


def _count_jsonl_zst_lines(path: Path) -> int:
    zstd = _get_zstandard()
    dctx = zstd.ZstdDecompressor()
    n = 0
    buf = b""
    with path.open("rb") as f, dctx.stream_reader(f) as reader:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                n += 1
                buf = buf[i + 1 :]
    if buf.strip():
        n += 1
    return n


def iter_jsonl_zst_rows(path: Path) -> Iterator[dict[str, Any]]:
    zstd = _get_zstandard()
    dctx = zstd.ZstdDecompressor()
    buf = b""
    with path.open("rb") as f, dctx.stream_reader(f) as reader:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                line = buf[:i]
                buf = buf[i + 1 :]
                if line.strip():
                    yield json.loads(line.decode("utf-8"))
        if buf.strip():
            yield json.loads(buf.decode("utf-8"))


def iter_shard_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        yield from iter_parquet_rows(path)
    elif _is_jsonl_zst(path):
        yield from iter_jsonl_zst_rows(path)
    else:
        raise ValueError(f"Unsupported input shard type: {path}")


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split into chunks of at most chunk_size characters, preferring newline breaks."""
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            nl = window.rfind("\n", max(0, len(window) - 500))
            if nl > 0:
                end = start + nl + 1
        piece = text[start:end]
        chunks.append(piece)
        start = end
    return chunks


def _merge_row_from_chunk_outputs(
    cfg: RunConfig,
    *,
    pq_name: str,
    row_idx: int,
    full_text: str,
    parts: list[str],
) -> dict[str, Any]:
    """Build one output row dict from ordered chunk model strings.

    Chunk strings are joined with ``PROGRAM_CHUNK_SEPARATOR`` for both rewrite (``text``)
    and code (``programs_delimited``) modes.
    """
    programs = [p if isinstance(p, str) else "" for p in parts]
    if not cfg.output_is_code:
        merged = PROGRAM_CHUNK_SEPARATOR.join(programs)
        return {
            "text": merged,
            "raw_content": full_text,
            "source_parquet": pq_name,
            "source_row_index": row_idx,
            "num_chunks": len(parts),
        }
    return {
        "raw_content": full_text,
        "programs": programs,
        "programs_delimited": PROGRAM_CHUNK_SEPARATOR.join(programs),
        "source_parquet": pq_name,
        "source_row_index": row_idx,
        "num_chunks": len(parts),
    }


class RowChunkTracker:
    """Thread-safe collector for parallel chunk results belonging to one document row."""

    __slots__ = (
        "row_idx",
        "full_text",
        "n_chunks",
        "pq_name",
        "doc_len",
        "parts",
        "_remaining",
        "trunc_chunks",
        "_lock",
    )

    def __init__(
        self,
        *,
        row_idx: int,
        full_text: str,
        n_chunks: int,
        pq_name: str,
        doc_len: int,
    ) -> None:
        self.row_idx = row_idx
        self.full_text = full_text
        self.n_chunks = n_chunks
        self.pq_name = pq_name
        self.doc_len = doc_len
        self.parts: list[str | None] = [None] * n_chunks
        self._remaining = n_chunks
        self.trunc_chunks = 0
        self._lock = threading.Lock()

    def record_chunk(
        self,
        chunk_idx: int,
        text: str,
        pt: int,
        ct: int,
        fr: str,
        metrics: Metrics,
    ) -> bool:
        """Store one chunk result; returns True when the whole row is complete."""
        metrics.add_usage(pt, ct)
        with self._lock:
            if _finish_reason_hit_max_tokens(fr):
                self.trunc_chunks += 1
            self.parts[chunk_idx] = text if isinstance(text, str) else ""
            self._remaining -= 1
            return self._remaining == 0


def _openrouter_one_chunk(
    prompt_prefix: str,
    chunk_body: str,
    cfg: RunConfig,
    api_key: str,
    *,
    global_doc_index: int | None = None,
    source_parquet_name: str | None = None,
    source_row_index: int | None = None,
    chunk_index0: int | None = None,
    n_chunks: int | None = None,
) -> tuple[str, int, int, str]:
    user_content = f"{prompt_prefix}\n\n{chunk_body}"
    return openrouter_chat(
        api_key=api_key,
        model=cfg.model,
        user_content=user_content,
        body_extras=cfg.body_extras,
        http_referer=cfg.http_referer,
        x_title=cfg.x_title,
        warn_global_doc_index=global_doc_index,
        warn_source_parquet_name=source_parquet_name,
        warn_source_row_index=source_row_index,
        warn_chunk_index0=chunk_index0,
        warn_n_chunks=n_chunks,
    )


@dataclass
class Metrics:
    lock: Any = field(default_factory=lambda: __import__("threading").Lock())
    completed_chunks: int = 0
    completed_docs: int = 0
    completed_parquets: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks_truncated_max_tokens: int = 0
    total_parquets: int = 0
    # Denominators (set after parquet precount)
    doc_denom: int = -1
    char_denom: int = -1
    api_tok_denom: int = -1

    def add_usage(self, pt: int, ct: int) -> None:
        with self.lock:
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.completed_chunks += 1

    def add_doc(self) -> None:
        with self.lock:
            self.completed_docs += 1

    def add_parquet_done(self) -> None:
        with self.lock:
            self.completed_parquets += 1

    def add_truncated_chunks(self, n: int) -> None:
        with self.lock:
            self.chunks_truncated_max_tokens += n

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "completed_chunks": self.completed_chunks,
                "completed_docs": self.completed_docs,
                "completed_parquets": self.completed_parquets,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "chunks_truncated_max_tokens": self.chunks_truncated_max_tokens,
            }


def _finish_reason_hit_max_tokens(finish_reason: Any) -> bool:
    """True when the model stopped because of the completion token cap (OpenAI-style API)."""
    if not isinstance(finish_reason, str):
        return False
    return finish_reason.strip().lower() in ("length", "max_tokens")


def _extract_assistant_text(message: dict[str, Any]) -> str:
    """
    Normalize ``choices[0].message.content`` from OpenRouter / OpenAI-style JSON.

    ``content`` may be a string, null, or a list of content parts (e.g. ``[{type,text}, ...]``).
    """
    raw = message.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("type")
                if t == "text" or "text" in block:
                    parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(raw)


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    user_content: str,
    body_extras: dict[str, Any],
    http_referer: str,
    x_title: str,
    timeout_sec: float = 300.0,
    warn_global_doc_index: int | None = None,
    warn_source_parquet_name: str | None = None,
    warn_source_row_index: int | None = None,
    warn_chunk_index0: int | None = None,
    warn_n_chunks: int | None = None,
) -> tuple[str, int, int, str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
    }
    for k, v in body_extras.items():
        if v is None or k in ("model", "messages"):
            continue
        payload[k] = v

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer,
            "X-Title": x_title,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_body}") from e

    try:
        choice0 = data["choices"][0]
        msg = choice0["message"]
        if not isinstance(msg, dict):
            raise TypeError("message is not a dict")
        content = _extract_assistant_text(msg)
        finish_reason = choice0.get("finish_reason") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response: {data!r}") from e

    pt = ct = 0
    usage = data.get("usage") or {}
    if isinstance(usage, dict):
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)

    if not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)

    if content == "":
        loc_parts: list[str] = []
        if warn_global_doc_index is not None:
            loc_parts.append(
                f"global_doc_index={warn_global_doc_index} "
                "(0-based ``n`` for ``view_nth_input_output_diff.py`` / ``print_nth_input_output.py``)"
            )
        if warn_source_parquet_name is not None and warn_source_row_index is not None:
            loc_parts.append(
                f"source_parquet={warn_source_parquet_name!r} source_row_index={warn_source_row_index}"
            )
        if (
            warn_chunk_index0 is not None
            and warn_n_chunks is not None
            and warn_n_chunks > 0
        ):
            loc_parts.append(f"chunk={warn_chunk_index0 + 1}/{warn_n_chunks}")
        loc_s = (" " + " ".join(loc_parts)) if loc_parts else ""
        print(
            "[warn] OpenRouter returned empty assistant ``content`` for this chunk "
            f"(finish_reason={finish_reason!r}, completion_tokens={ct}, prompt_tokens={pt}).{loc_s} "
            "With a very small ``openrouter.max_tokens``, some models spend the whole budget "
            "on internal reasoning and never emit visible text—try raising max_tokens, "
            "using a non-reasoning model, or shrinking ``chunk_size``.",
            flush=True,
        )

    return content, pt, ct, finish_reason


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SystemExit("YAML root must be a mapping (dict).")
    return raw


def build_body_extras(raw: dict[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    nested = raw.get("openrouter")
    if isinstance(nested, dict):
        extras.update(nested)
    for k, v in raw.items():
        if k in RESERVED_KEYS:
            continue
        extras[k] = v
    return extras


def output_effect_fingerprint(raw: dict[str, Any], *, shard_paths: list[Path]) -> str:
    """
    Stable digest of everything that changes per-chunk OpenRouter requests or merge shape.

    Includes model, prompt file bytes, data paths/type, chunk size, output_is_code, full
    API body extras, and the ordered list of resolved input shard paths (so ``max_parquets``
    / corpus layout changes invalidate resume).

    ``max_docs`` and ``max_chars`` are intentionally **excluded** so increasing those limits
    does not break ``continue: true`` (the run still enforces the **new** limits after resume).
    """
    pp = Path(os.path.expanduser(str(raw["prompt_path"])))
    if pp.is_file():
        ph = hashlib.sha256(pp.read_bytes()).hexdigest()
    else:
        ph = f"missing:{pp}"
    extras = build_body_extras(raw)
    dp = Path(os.path.expanduser(str(raw["data_parquet_dir"]))).resolve()
    blob: dict[str, Any] = {
        "model": raw["model"],
        "prompt_sha256": ph,
        "data_parquet_dir": str(dp),
        "data_parquet_type": str(raw.get("data_parquet_type", "fineweb")),
        "chunk_size": int(raw["chunk_size"]),
        "output_is_code": raw.get("output_is_code"),
        "body_extras": {k: extras[k] for k in sorted(extras.keys())},
        "input_shards": [str(p.resolve()) for p in shard_paths],
    }
    return json.dumps(blob, sort_keys=True, default=str)


def load_existing_output_rows(out_path: Path) -> list[dict[str, Any]]:
    """Load prior output rows; must have contiguous ``source_row_index`` starting at 0."""
    if not out_path.is_file():
        return []
    rows: list[dict[str, Any]] = pq.read_table(out_path).to_pylist()
    rows.sort(key=lambda r: int(r["source_row_index"]) if r.get("source_row_index") is not None else -1)
    for i, r in enumerate(rows):
        si = r.get("source_row_index")
        if si is None or int(si) != i:
            raise SystemExit(
                f"Cannot resume from {out_path}: need contiguous source_row_index 0..N-1; "
                f"position {i} has {si!r}."
            )
    return rows


@dataclass
class RunConfig:
    model: str
    num_concurrent: int
    prompt_path: Path
    data_parquet_dir: Path
    data_parquet_type: str
    chunk_size: int
    max_chars: int
    max_docs: int
    max_parquets: int
    max_tokens: int  # -1: metrics api_tok denom = parquet est only; else min(cap, est)
    data_output_dir: Path
    allow_output_overwrite: bool
    output_is_code: bool
    body_extras: dict[str, Any]
    metrics_interval_sec: float
    http_referer: str
    x_title: str
    continue_run: bool


def parse_config(raw: dict[str, Any]) -> RunConfig:
    try:
        model = str(raw["model"])
        num_concurrent = int(raw["num_concurrent"])
        prompt_path = Path(os.path.expanduser(str(raw["prompt_path"])))
        data_parquet_dir = Path(os.path.expanduser(str(raw["data_parquet_dir"])))
        data_parquet_type = str(raw.get("data_parquet_type", "fineweb"))
        chunk_size = int(raw["chunk_size"])
        max_chars = int(raw.get("max_chars", -1))
        max_docs = int(raw.get("max_docs", -1))
        max_parquets = int(raw.get("max_parquets", -1))
        max_tokens = int(raw.get("max_tokens", -1))
        data_output_dir = Path(os.path.expanduser(str(raw["data_output_dir"])))
        allow = _boolish(
            raw.get(
                "allow_output_overwrite",
                raw.get("allow_outpute_overwrite", False),
            )
        )
        output_is_code = _output_is_code_bool(raw.get("output_is_code", False))
        metrics_interval_sec = float(raw.get("metrics_interval_sec", 15.0))
        http_referer = str(
            raw.get("http_referer", "https://github.com/GAIR-NLP/ProX"),
        )
        x_title = str(raw.get("x_title", "ProX-openrouter-chunked"))
        continue_run = _boolish(raw.get("continue", False))
    except KeyError as e:
        raise SystemExit(f"Missing required config key: {e}") from e

    if num_concurrent < 1:
        raise SystemExit("num_concurrent must be >= 1")
    body_extras = build_body_extras(raw)

    return RunConfig(
        model=model,
        num_concurrent=num_concurrent,
        prompt_path=prompt_path,
        data_parquet_dir=data_parquet_dir,
        data_parquet_type=data_parquet_type,
        chunk_size=chunk_size,
        max_chars=max_chars,
        max_docs=max_docs,
        max_parquets=max_parquets,
        max_tokens=max_tokens,
        data_output_dir=data_output_dir,
        allow_output_overwrite=allow,
        output_is_code=output_is_code,
        body_extras=body_extras,
        metrics_interval_sec=metrics_interval_sec,
        http_referer=http_referer,
        x_title=x_title,
        continue_run=continue_run,
    )


@dataclass(frozen=True)
class ViewPaths:
    """Subset of config for viewers (no OpenRouter API / concurrency keys)."""

    data_parquet_dir: Path
    data_parquet_type: str
    data_output_dir: Path
    max_parquets: int


def parse_view_paths(raw: dict[str, Any]) -> ViewPaths:
    """
    Fields needed to map a global row index to an input shard row and the matching output parquet.

    Accepts OpenRouter names (``data_parquet_dir``, ``data_output_dir``, …) and refiner/denoise
    aliases (``input_parquet_dir``, ``output_data_dir``, ``dataset_type``).
    """
    dp = raw.get("data_parquet_dir") or raw.get("input_parquet_dir")
    if dp is None:
        raise SystemExit(
            "YAML must include data_parquet_dir or input_parquet_dir (input shard directory)."
        )
    out = raw.get("data_output_dir") or raw.get("output_data_dir")
    if out is None:
        raise SystemExit(
            "YAML must include data_output_dir or output_data_dir (output directory)."
        )
    dtype = raw.get("data_parquet_type") or raw.get("dataset_type") or "fineweb"
    max_parquets = int(raw.get("max_parquets", -1))
    return ViewPaths(
        data_parquet_dir=Path(os.path.expanduser(str(dp))).resolve(),
        data_parquet_type=str(dtype),
        data_output_dir=Path(os.path.expanduser(str(out))).resolve(),
        max_parquets=max_parquets,
    )


def resolve_view_output_parquet(output_dir: Path, shard_stem: str) -> Path:
    """
    Output shard file written by ``run_openrouter_chunked`` or ``refiner_data_generation/denoise_dataset``.

    Prefers ``{{stem}}_openrouter.parquet``, then ``{{stem}}_denoised.parquet``.
    """
    candidates = [
        output_dir / f"{shard_stem}_openrouter.parquet",
        output_dir / f"{shard_stem}_denoised.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Output parquet not found (run the pipeline first). Tried: {tried}")


def precount_doc_rows(shard_paths: list[Path]) -> int:
    """Row counts from parquet footers (fast) or newline count for ``*.jsonl.zst``."""
    total = 0
    for p in shard_paths:
        if p.suffix.lower() == ".parquet":
            total += pq.ParquetFile(p).metadata.num_rows
        elif _is_jsonl_zst(p):
            total += _count_jsonl_zst_lines(p)
        else:
            raise ValueError(f"Unsupported shard for row count: {p}")
    return total


def precount_chars_and_token_est(
    parquet_paths: list[Path],
    row_text_fn: Callable[[dict[str, Any]], str],
) -> tuple[int, int]:
    """Sum document characters and a rough token estimate (ceil(len/4)) per row."""
    tot_chars = 0
    tot_tok_est = 0
    for path in parquet_paths:
        for row in iter_shard_rows(path):
            s = row_text_fn(row)
            n = len(s)
            tot_chars += n
            tot_tok_est += max(1, (n + 3) // 4)
    return tot_chars, tot_tok_est


def _doc_denom(max_docs: int, row_total: int) -> int:
    if row_total < 0:
        return max_docs if max_docs >= 0 else -1
    if max_docs < 0:
        return row_total
    return min(max_docs, row_total)


def _char_denom(max_chars: int, parquet_chars: int) -> int:
    if parquet_chars < 0:
        return max_chars if max_chars >= 0 else -1
    if max_chars < 0:
        return parquet_chars
    return min(max_chars, parquet_chars)


def _api_tok_denom(max_tokens: int, parquet_tok_est: int) -> int:
    if parquet_tok_est < 0:
        return max_tokens if max_tokens >= 0 else -1
    if max_tokens < 0:
        return parquet_tok_est
    return min(max_tokens, parquet_tok_est)


def iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches():
        col_names = batch.schema.names
        for i in range(batch.num_rows):
            yield {name: batch.column(j)[i].as_py() for j, name in enumerate(col_names)}


def run(cfg_path: Path) -> None:
    raw = load_yaml_config(cfg_path)
    cfg = parse_config(raw)
    metrics = Metrics()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment.")

    if not cfg.prompt_path.is_file():
        raise SystemExit(f"prompt_path not found: {cfg.prompt_path}")
    prompt_prefix = cfg.prompt_path.read_text(encoding="utf-8").strip()
    if not cfg.data_parquet_dir.is_dir():
        raise SystemExit(f"data_parquet_dir is not a directory: {cfg.data_parquet_dir}")

    row_text_fn = get_row_text_fn(cfg.data_parquet_type)
    all_shards = sorted_shard_paths(cfg.data_parquet_dir)
    if not all_shards:
        raise SystemExit(
            f"No .parquet or *.jsonl.zst shards under {cfg.data_parquet_dir} (recursive search)."
        )

    if cfg.max_parquets >= 0:
        parquet_paths = all_shards[: cfg.max_parquets]
    else:
        parquet_paths = all_shards

    fp_new = output_effect_fingerprint(raw, shard_paths=parquet_paths)
    cfg_src = cfg_path.expanduser().resolve()
    if not cfg_src.is_file():
        raise SystemExit(f"yaml_config not found: {cfg_src}")
    cfg_dst = cfg.data_output_dir / cfg_src.name

    resume_ok = False
    if cfg.continue_run:
        if not cfg.data_output_dir.is_dir():
            print(
                "[init] continue: true but data_output_dir does not exist yet; starting fresh.",
                flush=True,
            )
        elif not cfg_dst.is_file():
            raise SystemExit(
                f"continue: true requires the prior run's copied YAML at {cfg_dst}.\n"
                "Run once without continue (or copy your YAML there), then enable continue."
            )
        else:
            prev_raw = load_yaml_config(cfg_dst)
            prev_dp = Path(os.path.expanduser(str(prev_raw["data_parquet_dir"])))
            prev_paths = sorted_shard_paths(prev_dp)
            prev_mp = int(prev_raw.get("max_parquets", -1))
            if prev_mp >= 0:
                prev_paths = prev_paths[:prev_mp]
            fp_old = output_effect_fingerprint(prev_raw, shard_paths=prev_paths)
            if fp_old != fp_new:
                raise SystemExit(
                    "continue: true but output-affecting settings differ from the saved YAML "
                    f"in {cfg_dst} (model, prompt bytes, chunk_size, output_is_code, openrouter "
                    "extras, input shard list, data_parquet_dir/type, etc.). "
                    "Note: max_docs and max_chars are not fingerprinted—you may raise them freely.\n"
                    "Use a new data_output_dir, set continue: false, or align the YAML with that copy."
                )
            resume_ok = True
            print(
                f"[init] continue: output fingerprint matches {cfg_dst.name}; resuming in-place.",
                flush=True,
            )

    if cfg.data_output_dir.exists():
        if not cfg.allow_output_overwrite and not resume_ok:
            raise SystemExit(
                f"data_output_dir already exists: {cfg.data_output_dir}\n"
                "Choose a new path, set allow_output_overwrite: true, or use continue: true "
                "with an unchanged output-affecting config and the prior copied YAML present."
            )
    else:
        cfg.data_output_dir.mkdir(parents=True, exist_ok=False)

    if cfg_src.resolve() != cfg_dst.resolve():
        shutil.copy2(cfg_src, cfg_dst)
        print(f"[init] copied run YAML to {cfg_dst}", flush=True)

    print(
        f"[init] model={cfg.model!r} parquets={len(parquet_paths)} "
        f"max_docs={cfg.max_docs} max_chars={cfg.max_chars} max_tokens={cfg.max_tokens} "
        f"chunk_size={cfg.chunk_size} output_is_code={cfg.output_is_code} continue={cfg.continue_run}",
        flush=True,
    )

    shard_row_prefix: list[int] | None = None
    try:
        pfx = [0]
        for p in parquet_paths:
            pfx.append(pfx[-1] + precount_doc_rows([p]))
        row_total = pfx[-1]
        shard_row_prefix = pfx
    except Exception as e:  # pragma: no cover
        print(f"[init] row precount skipped: {e}", flush=True)
        row_total = -1
        shard_row_prefix = None

    print("[init] precounting chars + rough token estimate over selected parquets …", flush=True)
    try:
        parquet_chars, parquet_tok_est = precount_chars_and_token_est(
            parquet_paths, row_text_fn
        )
    except Exception as e:  # pragma: no cover
        print(f"[init] char/token precount skipped: {e}", flush=True)
        parquet_chars, parquet_tok_est = -1, -1

    metrics.total_parquets = len(parquet_paths)
    metrics.doc_denom = _doc_denom(cfg.max_docs, row_total)
    metrics.char_denom = _char_denom(cfg.max_chars, parquet_chars)
    metrics.api_tok_denom = _api_tok_denom(cfg.max_tokens, parquet_tok_est)

    print(
        f"[init] rows_in_parquets={row_total} parquet_chars={parquet_chars} "
        f"parquet_tok_est={parquet_tok_est} "
        f"doc_denom=min(max_docs,rows)={metrics.doc_denom} "
        f"char_denom=min(max_chars,parquet_chars)={metrics.char_denom} "
        f"api_tok_denom=min(max_tokens,parquet_tok_est)={metrics.api_tok_denom}",
        flush=True,
    )

    global_docs = 0
    global_chars = 0
    # Rows with chunk HTTP work in flight (not yet merged); with global_docs caps max_docs
    # so we do not queue unbounded rows before the first completion.
    inflight_docs = 0
    run_lock = threading.Lock()
    wall0 = time.perf_counter()
    last_metric_wall = wall0

    def maybe_print_metrics() -> None:
        nonlocal last_metric_wall
        now = time.perf_counter()
        if now - last_metric_wall < cfg.metrics_interval_sec:
            return
        last_metric_wall = now
        snap = metrics.snapshot()
        elapsed = max(now - wall0, 1e-9)
        p_cap = metrics.total_parquets
        doc_d = metrics.doc_denom
        ch_d = metrics.char_denom
        api_d = metrics.api_tok_denom

        docs_s = snap["completed_docs"] / elapsed
        toks = snap["prompt_tokens"] + snap["completion_tokens"]
        tok_s = toks / elapsed

        doc_frac = f"{snap['completed_docs']}/{doc_d}" if doc_d >= 0 else f"{snap['completed_docs']}/?"
        ch_frac = f"{global_chars}/{ch_d}" if ch_d >= 0 else f"{global_chars}/?"
        pq_frac = f"{snap['completed_parquets']}/{p_cap}"
        api_frac = f"{toks}/{api_d}" if api_d >= 0 else f"{toks}/?"

        chars_s = global_chars / elapsed
        eta_candidates: list[float] = []
        if doc_d >= 0:
            rem_d = doc_d - snap["completed_docs"]
            if rem_d > 0 and docs_s > 1e-12:
                eta_candidates.append(rem_d / docs_s)
        if ch_d >= 0:
            rem_c = ch_d - global_chars
            if rem_c > 0 and chars_s > 1e-12:
                eta_candidates.append(rem_c / chars_s)
        if api_d >= 0:
            rem_a = api_d - toks
            if rem_a > 0 and tok_s > 1e-12:
                eta_candidates.append(rem_a / tok_s)
        if eta_candidates:
            eta_s = f" eta_rem≈{_fmt_eta_sec(min(eta_candidates))} (min of doc/char/api-tok ETAs)"
        else:
            eta_s = " eta_rem=?"

        trunc = snap.get("chunks_truncated_max_tokens", 0)
        trunc_s = f" trunc_max_tok_chunks={trunc}" if trunc else ""
        print(
            f"[metrics] wall={elapsed:.1f}s docs/s={docs_s:.4f} toks/s={tok_s:.2f} "
            f"chunks={snap['completed_chunks']} docs={doc_frac} "
            f"chars={ch_frac} parquets={pq_frac} api_toks={api_frac} "
            f"prompt_tok={snap['prompt_tokens']} completion_tok={snap['completion_tokens']}"
            f"{trunc_s}{eta_s}",
            flush=True,
        )

    for pq_idx, pq_path in enumerate(parquet_paths):
        out_name = f"{pq_path.stem}_openrouter.parquet"
        out_path = cfg.data_output_dir / out_name
        if out_path.exists() and not cfg.allow_output_overwrite and not resume_ok:
            raise SystemExit(
                f"Output file already exists: {out_path}\n"
                "Remove it, pick a new data_output_dir, or set allow_output_overwrite: true."
            )

        results_by_row: dict[int, dict[str, Any]] = {}
        n_done = 0
        if resume_ok and out_path.is_file():
            existing = load_existing_output_rows(out_path)
            for r in existing:
                idx = int(r["source_row_index"])
                results_by_row[idx] = r
                rc = r.get("raw_content")
                if isinstance(rc, str):
                    global_chars += len(rc)
                metrics.add_doc()
                global_docs += 1
            n_done = len(existing)
            if n_done:
                print(
                    f"[init] resume shard {pq_path.name!r}: keeping {n_done} row(s) from {out_path.name}",
                    flush=True,
                )

        parquet_trunc_chunks = 0
        chunk_futures: set[Any] = set()
        fut_meta: dict[Any, tuple[RowChunkTracker, int]] = {}
        inflight_chars = 0

        def _finish_row_tracker(tr: RowChunkTracker) -> None:
            nonlocal inflight_chars, inflight_docs, global_docs, global_chars, parquet_trunc_chunks
            parts = [p if isinstance(p, str) else "" for p in tr.parts]
            out_row = _merge_row_from_chunk_outputs(
                cfg,
                pq_name=tr.pq_name,
                row_idx=tr.row_idx,
                full_text=tr.full_text,
                parts=parts,
            )
            results_by_row[tr.row_idx] = out_row
            if tr.trunc_chunks:
                metrics.add_truncated_chunks(tr.trunc_chunks)
                parquet_trunc_chunks += tr.trunc_chunks
                print(
                    f"[warn] OpenRouter completion max_tokens: {tr.trunc_chunks}/{tr.n_chunks} "
                    f"chunk(s) ended with finish_reason=length (output likely truncated). "
                    f"source_parquet={tr.pq_name} source_row_index={tr.row_idx}",
                    flush=True,
                )
            with run_lock:
                inflight_chars -= tr.doc_len
                inflight_docs -= 1
                global_docs += 1
                global_chars += tr.doc_len
            metrics.add_doc()
            maybe_print_metrics()

        def _drain_one_chunk_future() -> None:
            if not chunk_futures:
                return
            done, _ = wait(chunk_futures, return_when=FIRST_COMPLETED)
            chunk_futures.difference_update(done)
            for fut in done:
                tr, cidx = fut_meta.pop(fut)
                text, pt, ct, fr = fut.result()
                if tr.record_chunk(cidx, text, pt, ct, fr, metrics):
                    _finish_row_tracker(tr)

        def _drain_until_chunk_room() -> None:
            while len(chunk_futures) >= cfg.num_concurrent:
                _drain_one_chunk_future()

        with ThreadPoolExecutor(max_workers=max(1, cfg.num_concurrent)) as ex:
            for row_idx, row in enumerate(iter_shard_rows(pq_path)):
                if row_idx < n_done:
                    continue
                _drain_until_chunk_room()
                with run_lock:
                    if cfg.max_docs >= 0 and global_docs + inflight_docs >= cfg.max_docs:
                        break
                text = row_text_fn(row)
                doc_len = len(text)
                with run_lock:
                    if cfg.max_chars >= 0 and global_chars + inflight_chars + doc_len > cfg.max_chars:
                        break
                chunks = chunk_text(text, cfg.chunk_size)
                if not chunks:
                    results_by_row[row_idx] = _merge_row_from_chunk_outputs(
                        cfg,
                        pq_name=pq_path.name,
                        row_idx=row_idx,
                        full_text=text,
                        parts=[],
                    )
                    with run_lock:
                        global_docs += 1
                        global_chars += doc_len
                    metrics.add_doc()
                    maybe_print_metrics()
                    continue

                with run_lock:
                    inflight_chars += doc_len
                    inflight_docs += 1
                tr = RowChunkTracker(
                    row_idx=row_idx,
                    full_text=text,
                    n_chunks=len(chunks),
                    pq_name=pq_path.name,
                    doc_len=doc_len,
                )
                for cidx, ch in enumerate(chunks):
                    _drain_until_chunk_room()
                    gdi: int | None = None
                    if shard_row_prefix is not None:
                        gdi = shard_row_prefix[pq_idx] + row_idx
                    fut = ex.submit(
                        _openrouter_one_chunk,
                        prompt_prefix,
                        ch,
                        cfg,
                        api_key,
                        global_doc_index=gdi,
                        source_parquet_name=pq_path.name,
                        source_row_index=row_idx,
                        chunk_index0=cidx,
                        n_chunks=len(chunks),
                    )
                    chunk_futures.add(fut)
                    fut_meta[fut] = (tr, cidx)

            while chunk_futures:
                _drain_one_chunk_future()

        out_rows = [results_by_row[i] for i in sorted(results_by_row)]

        if out_rows:
            table = pa.Table.from_pylist(out_rows)
            pq.write_table(table, out_path)

        if parquet_trunc_chunks:
            print(
                f"[warn] Wrote {out_path.name}: {parquet_trunc_chunks} chunk response(s) "
                "hit the completion max_tokens cap (finish_reason=length). "
                "Raise openrouter.max_tokens or shorten chunks/prompt if this is unwanted.",
                flush=True,
            )

        metrics.add_parquet_done()
        maybe_print_metrics()

        if cfg.max_docs >= 0 and global_docs >= cfg.max_docs:
            break
        if cfg.max_chars >= 0 and global_chars >= cfg.max_chars:
            break

    elapsed = time.perf_counter() - wall0
    snap = metrics.snapshot()
    toks_done = snap["prompt_tokens"] + snap["completion_tokens"]
    doc_d = metrics.doc_denom
    ch_d = metrics.char_denom
    api_d = metrics.api_tok_denom
    doc_part = f"{snap['completed_docs']}/{doc_d}" if doc_d >= 0 else f"{snap['completed_docs']}/?"
    ch_part = f"{global_chars}/{ch_d}" if ch_d >= 0 else f"{global_chars}/?"
    api_part = f"{toks_done}/{api_d}" if api_d >= 0 else f"{toks_done}/?"
    trunc_total = snap.get("chunks_truncated_max_tokens", 0)
    trunc_warn = (
        f" [warn] {trunc_total} chunk(s) hit openrouter max_tokens (finish_reason=length); "
        "see logs above per row / per parquet."
        if trunc_total
        else ""
    )
    print(
        f"[done] wall={elapsed:.2f}s docs={doc_part} chars={ch_part} api_toks={api_part} "
        f"chunks={snap['completed_chunks']} parquets={snap['completed_parquets']} "
        f"prompt_tok={snap['prompt_tokens']} completion_tok={snap['completion_tokens']} "
        f"out_dir={cfg.data_output_dir}{trunc_warn}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml_config", type=Path, help="Path to YAML config file.")
    args = ap.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    run(args.yaml_config.expanduser().resolve())


if __name__ == "__main__":
    main()
