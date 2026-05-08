#!/usr/bin/env python3
"""
View **two** simple red/green character diffs (``difflib.SequenceMatcher``, ``autojunk=False``), served
on one ephemeral local server:

- ``/`` — **refiner / output stage**: input shard (``input_parquet_dir`` / ``data_parquet_dir``) vs
  denoised row from ``output_data_dir`` when the dataset YAML defines it (same as ``denoise_dataset``).
  Parquet-only mode without refiner parquets uses raw vs middle from the viewer YAML.
- ``/openrouter`` — **OpenRouter stage**: **same noisy/raw input string as tab 1’s first side** vs final
  model output (stored parquet or live API response). Requests still use denoised chunks internally for
  ``--live-query``; this diff only compares raw input text to the merged API reply. Chunk-merge markers
  stripped from the final string for diffing.

The default browser opens **both** tabs. Positional ``PATH`` arguments depend on mode (see below).

Use ``--largest-dif`` (and optional ``--largest-dif-k K``) to open the row with the **K-th largest**
**dissimilarity(middle, parquet output)** among all documents (``K=1`` = max). Ranking uses a fast
``difflib.SequenceMatcher.quick_ratio`` proxy (not exact edit distance). Compare a prefix unless
``--largest-dif-compare-chars 0``. Without ``--live-query``, pass **VIEWER_YAML** then **DOC_INDEX**
(unless ``--largest-dif`` with a single YAML). With ``--live-query``, pass dataset then model YAML plus **DOC_INDEX**, or **two arguments**
``OPENROUTER_CONFIG.yaml DOC_INDEX`` when the config lives under ``openrouter_data_generation/``. With
``--largest-dif``, use two YAMLs (dataset + model) or one YAML only under ``refiner_data_generation/``
(static refiner); do **not** use ``--largest-dif`` with a lone openrouter_generation YAML — pass an explicit
index instead (two-arg form above).

**No HTML file is written** (nothing under ``.cache/`` or the repo). One server serves **two** routes
(``/`` and ``/openrouter``); the default browser opens **two** tabs. The local server shuts down after a
short idle period; **tabs do not close** automatically, but **reload will fail** once the server has stopped.

Usage::

  python openrouter_data_generation/view_nth_input_output_diff.py \\
    openrouter_data_generation/example_openrouter_rewrite.yaml 7

  python openrouter_data_generation/view_nth_input_output_diff.py --one-based \\
    openrouter_data_generation/example_openrouter_rewrite.yaml 1

  python openrouter_data_generation/view_nth_input_output_diff.py --largest-dif config.yaml

  python openrouter_data_generation/view_nth_input_output_diff.py --largest-dif --largest-dif-k 3 config.yaml

Live OpenRouter (**two YAML paths**, then optional doc index; no parquet): dataset/denoising YAML first
(``input_parquet_dir``, ``output_data_dir`` / ``data_output_dir``), OpenRouter model YAML second::

  python openrouter_data_generation/view_nth_input_output_diff.py \\
    --live-query \\
    refiner_data_generation/example_denoise.yaml \\
    openrouter_data_generation/example_openrouter_rewrite.yaml \\
    99

Rank by middle vs parquet dissimilarity (no doc index). **Two YAMLs** + ``--largest-dif`` → live API plus
two tabs. **One YAML** under ``refiner_data_generation/`` + ``--largest-dif`` → refiner-only static page
**(no OpenRouter)**. Single ``openrouter_data_generation/`` config **cannot** use ``--largest-dif`` (no auto
ranking); give **explicit** ``DOC_INDEX`` with two arguments::

  python openrouter_data_generation/view_nth_input_output_diff.py \\
    --live-query \\
    openrouter_data_generation/example_openrouter_rewrite.yaml \\
    7

Other examples::

  python openrouter_data_generation/view_nth_input_output_diff.py \\
    --live-query --largest-dif --largest-dif-k 1 \\
    refiner_data_generation/example_denoise.yaml \\
    openrouter_data_generation/example_openrouter_rewrite.yaml

  python openrouter_data_generation/view_nth_input_output_diff.py \\
    --live-query --largest-dif --largest-dif-k 1 \\
    refiner_data_generation/example_denoise.yaml
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import os
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_MOD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MOD_DIR.parent
_REFINER_GEN_DIR = _REPO_ROOT / "refiner_data_generation"
_OPENROUTER_GEN_DIR = _REPO_ROOT / "openrouter_data_generation"
DEFAULT_QUERY_MODEL_YAML = _MOD_DIR / "example_openrouter_rewrite.yaml"


def _repo_subdir_yaml_kind(path: Path) -> tuple[Path, str | None]:
    """Return ``(resolved_path, \"refiner\"|\"openrouter\"|None)`` if YAML path is inside known dirs."""
    try:
        rp = path.expanduser().resolve()
    except OSError:
        return path, None
    for label, root in (
        ("refiner", _REFINER_GEN_DIR.resolve()),
        ("openrouter", _OPENROUTER_GEN_DIR.resolve()),
    ):
        try:
            rp.relative_to(root)
            return rp, label
        except ValueError:
            continue
    return rp, None


def _live_largest_dif_single_yaml_side(path: Path, ap: argparse.ArgumentParser) -> str:
    """``refiner`` only (never ``openrouter``): static input → denoised page."""
    rp, kind = _repo_subdir_yaml_kind(path)
    if kind == "openrouter":
        ap.error(
            "Cannot combine --live-query --largest-dif with **only** a YAML under "
            f"{_OPENROUTER_GEN_DIR}: there is no auto-ranked “max” row. "
            "Pass **DOC_INDEX** explicitly: `--live-query` `CONFIG.yaml` `N` "
            "(two arguments). Or pass **two YAMLs** (`refiner`/dataset + OpenRouter model) with "
            "`--largest-dif` to rank vs dataset parquets."
        )
    if kind != "refiner":
        ap.error(
            "Single YAML for --live-query --largest-dif must be under:\n"
            f"  {_REFINER_GEN_DIR}\n"
            "(openrouter_generation configs need an explicit DOC_INDEX or two YAMLs). "
            f"(got {rp})."
        )
    return kind

# Must match ``run_openrouter_chunked.PROGRAM_CHUNK_SEPARATOR``
PROGRAM_CHUNK_SEPARATOR = "\n\n### OPENROUTER_CHUNK_PROGRAM ###\n\n"


def strip_openrouter_chunk_delimiters(text: str) -> str:
    """Remove chunk-merge markers from stored OpenRouter output (HTML display only)."""
    if not text:
        return text
    return text.replace(PROGRAM_CHUNK_SEPARATOR, "")


def dissimilarity_middle_vs_final(a: str, b: str) -> float:
    """
    Fast ranking proxy for “how different” ``a`` and ``b`` are (middle vs parquet output).

    Uses ``SequenceMatcher(..., autojunk=False).quick_ratio()`` so typical cost is linear in string
    length, unlike classic Levenshtein **O(|a|·|b|)**. Score is ``(1 - quick_ratio) * (len(a)+len(b))``
    so longer mismatches tend to rank higher, similar in spirit to edit distance for ranking.
    """
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0:
        return float(lb)
    if lb == 0:
        return float(la)
    sm = SequenceMatcher(None, a, b, autojunk=False)
    r = sm.quick_ratio()
    return (1.0 - r) * float(la + lb)


def find_row_kth_largest_dissimilarity_middle_vs_final(
    pn: Any,
    yaml_config: Path,
    *,
    compare_chars: int,
    k: int,
) -> tuple[str, str, str, dict[str, Any], float]:
    """
    Scan all **paired** input/output rows, rank by **dissimilarity_middle_vs_final** (quick_ratio-based).

    Sort descending by score (ties: lower ``global_idx`` first); return the **k-th** row (1-based).

    When ``compare_chars > 0``, only prefixes of that length are compared.
    ``compare_chars == 0`` uses full strings (linear-time ranking per row, but large corpora can still
    add up).
    """
    if k < 1:
        raise SystemExit("--largest-dif-k must be >= 1")
    scored: list[tuple[float, int, str, str, str, dict[str, Any]]] = []
    for gidx, raw, mid, fin, meta in pn.iter_view_yaml_rows(yaml_config):
        if compare_chars > 0:
            a, b = mid[:compare_chars], fin[:compare_chars]
        else:
            a, b = mid, fin
        d = dissimilarity_middle_vs_final(a, b)
        scored.append((d, gidx, raw, mid, fin, meta))
    if not scored:
        raise SystemExit("No rows found for YAML paths.")
    scored.sort(key=lambda t: (-t[0], t[1]))
    if k > len(scored):
        raise SystemExit(
            f"--largest-dif-k={k} but only {len(scored)} row(s) in the selected corpus."
        )
    score, _gidx, raw_text, middle_text, out_text, meta0 = scored[k - 1]
    meta = dict(meta0)
    meta["largest_dif_score"] = score
    meta["largest_dif_compare_chars"] = compare_chars
    meta["largest_dif_k"] = k
    return raw_text, middle_text, out_text, meta, score


def _load_print_nth_module() -> Any:
    name = "print_nth_input_output"
    path = _MOD_DIR / "print_nth_input_output.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_run_openrouter_module() -> Any:
    name = "run_openrouter_chunked"
    path = _MOD_DIR / "run_openrouter_chunked.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Allow re-load if print_nth already registered a stub name — replace
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_run_config_for_openrouter_query(
    roc: Any,
    *,
    dataset_yaml: Path,
    model_yaml: Path | None,
) -> Any:
    """
    Merge a **dataset** YAML (``input_parquet_dir`` / ``data_parquet_dir``) with an OpenRouter
    model YAML (``model``, ``prompt_path``, ``chunk_size``, ``openrouter:``, …).
    """
    ds_raw = roc.load_yaml_config(dataset_yaml)
    dp = ds_raw.get("data_parquet_dir") or ds_raw.get("input_parquet_dir")
    if dp is None:
        raise SystemExit(
            "Dataset YAML must set ``data_parquet_dir`` or ``input_parquet_dir`` "
            "(directory of ``*.parquet`` or ``*.jsonl.zst`` shards)."
        )
    dtype = ds_raw.get("data_parquet_type") or ds_raw.get("dataset_type") or "fineweb"
    max_parquets = int(ds_raw.get("max_parquets", -1))

    mod_path = (
        model_yaml.expanduser().resolve()
        if model_yaml is not None
        else DEFAULT_QUERY_MODEL_YAML.resolve()
    )
    if not mod_path.is_file():
        raise SystemExit(f"OpenRouter model YAML not found: {mod_path}")

    m_raw = roc.load_yaml_config(mod_path)
    merged: dict[str, Any] = dict(m_raw)
    merged["data_parquet_dir"] = dp
    merged["data_parquet_type"] = dtype
    merged["max_parquets"] = max_parquets
    return roc.parse_config(merged)


def resolve_live_query_run_config(
    roc: Any,
    dataset_yaml: Path,
    openrouter_yaml: Path | None,
) -> tuple[Any, Path]:
    """Merge **dataset** YAML with **OpenRouter** model YAML.

    If ``openrouter_yaml`` is ``None``, uses ``DEFAULT_QUERY_MODEL_YAML`` for API/model keys.
    """
    ds = dataset_yaml.expanduser().resolve()
    mq = (
        openrouter_yaml.expanduser().resolve()
        if openrouter_yaml is not None
        else DEFAULT_QUERY_MODEL_YAML.resolve()
    )
    if not mq.is_file():
        raise SystemExit(f"OpenRouter model YAML not found: {mq}")
    return (
        parse_run_config_for_openrouter_query(
            roc,
            dataset_yaml=ds,
            model_yaml=openrouter_yaml,
        ),
        mq,
    )


def _show_live_query_diff_html(
    input_raw_text: str,
    denoised_text: str,
    openrouter_response: str,
    meta: dict[str, Any],
    *,
    refiner_parquet_loaded: bool,
    live_tabs: str,
    max_chars: int,
    no_browser: bool,
    wait_sec: float,
    keep_alive_sec: float,
    port: int,
) -> None:
    ir, dn, ot = input_raw_text, denoised_text, openrouter_response
    truncated = False
    if len(ir) + len(dn) + len(ot) > max_chars:
        n = max(1, max_chars // 3)
        ir = ir[:n]
        dn = dn[:n]
        ot = ot[:n]
        truncated = True
    html_out, html_or = _html_pair_for_live_query(
        ir,
        dn,
        ot,
        meta,
        truncated,
        refiner_parquet_loaded=refiner_parquet_loaded,
        title_suffix=" — OpenRouter live query",
    )
    if live_tabs == "openrouter_only":
        routes = {"/openrouter": html_or}
        open_paths = ["/openrouter"]
    else:
        routes = {"/": html_out, "/openrouter": html_or}
        open_paths = ["/", "/openrouter"]
    if not no_browser:
        _serve_ephemeral_html_multi(
            routes,
            open_browser=True,
            open_paths=open_paths,
            wait_first_request_sec=wait_sec,
            keep_alive_after_first_sec=keep_alive_sec,
            bind_port=port,
            quiet=True,
        )
    else:
        print("Built HTML in memory only (--no-browser); not serving.", flush=True)


def run_live_openrouter_query(
    roc: Any,
    *,
    cfg: Any,
    doc_index: int,
    one_based: bool,
    dataset_yaml: Path,
    model_yaml: Path,
    max_chars: int,
    no_browser: bool,
    wait_sec: float,
    keep_alive_sec: float,
    port: int,
    live_tabs: str = "both",
) -> None:
    """Send one corpus row through OpenRouter (same path as ``run_openrouter_chunked.run``), then HTML diff."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment.")

    if not cfg.prompt_path.is_file():
        raise SystemExit(f"prompt_path not found: {cfg.prompt_path}")
    prompt_prefix = cfg.prompt_path.read_text(encoding="utf-8").strip()
    if not cfg.data_parquet_dir.is_dir():
        raise SystemExit(f"data_parquet_dir is not a directory: {cfg.data_parquet_dir}")

    row_text_fn = roc.get_row_text_fn(cfg.data_parquet_type)
    all_shards = roc.sorted_shard_paths(cfg.data_parquet_dir)
    if not all_shards:
        raise SystemExit(f"No shards under {cfg.data_parquet_dir}")
    shard_paths = (
        all_shards[: cfg.max_parquets] if cfg.max_parquets >= 0 else all_shards
    )

    idx = doc_index - 1 if one_based else doc_index
    shard_path, local_i = roc._resolve_global_doc_index(shard_paths, idx)
    row = roc._nth_row_from_shard(shard_path, local_i)
    input_raw_text = row_text_fn(row)

    ds_raw = roc.load_yaml_config(dataset_yaml.expanduser().resolve())
    refiner_root = ds_raw.get("output_data_dir") or ds_raw.get("data_output_dir")
    denoised_text = input_raw_text
    refiner_parquet_loaded = False
    refiner_parquet_path: Path | None = None
    if refiner_root:
        rr = Path(os.path.expanduser(str(refiner_root))).resolve()
        try:
            refiner_parquet_path = roc.resolve_view_output_parquet(rr, shard_path.stem)
            denoised_row = roc._nth_row_from_shard(refiner_parquet_path, local_i)
            denoised_text = row_text_fn(denoised_row)
            refiner_parquet_loaded = True
        except (FileNotFoundError, IndexError, OSError, ValueError) as e:
            print(
                f"[warn] Live-query: could not load denoised row from output_data_dir ({e!s}); "
                f"using raw input for refiner diff and for OpenRouter.",
                file=sys.stderr,
            )
            denoised_text = input_raw_text

    text = denoised_text
    chunks = roc.chunk_text(text, cfg.chunk_size)

    yc_lines = [
        f"dataset_yaml={dataset_yaml}",
        f"model_yaml={model_yaml}",
    ]
    meta_base: dict[str, Any] = {
        "global_idx": idx,
        "local_row": local_i,
        "yaml_config": "\n".join(yc_lines),
        "shard_path": shard_path,
        "output_parquet": "(live OpenRouter response)",
        "raw_parquet_dir": None,
        "live_input_dir": str(cfg.data_parquet_dir.resolve()),
        "live_refiner_parquet": str(refiner_parquet_path) if refiner_parquet_path else None,
    }

    if not chunks:
        merged_row = roc._merge_row_from_chunk_outputs(
            cfg,
            pq_name=shard_path.name,
            row_idx=local_i,
            full_text=text,
            parts=[],
        )
        key = "programs_delimited" if cfg.output_is_code else "text"
        after_text = merged_row.get(key, "")
        _show_live_query_diff_html(
            input_raw_text,
            denoised_text,
            after_text,
            meta_base,
            refiner_parquet_loaded=refiner_parquet_loaded,
            live_tabs=live_tabs,
            max_chars=max_chars,
            no_browser=no_browser,
            wait_sec=wait_sec,
            keep_alive_sec=keep_alive_sec,
            port=port,
        )
        return

    n_ch = len(chunks)
    workers = max(1, cfg.num_concurrent)

    def _run_query_chunk(cidx: int, chunk: str) -> tuple[int, str]:
        print(
            f"[live-query] OpenRouter request chunk {cidx + 1}/{n_ch} "
            f"({len(chunk)} chars), workers≤{workers}",
            flush=True,
        )
        out_text, _, _, _ = roc._openrouter_one_chunk(
            prompt_prefix,
            chunk,
            cfg,
            api_key,
            global_doc_index=idx,
            source_parquet_name=shard_path.name,
            source_row_index=local_i,
            chunk_index0=cidx,
            n_chunks=n_ch,
            quiet=True,
        )
        return cidx, out_text

    results: list[str | None] = [None] * n_ch
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_query_chunk, cidx, ch) for cidx, ch in enumerate(chunks)]
        for fut in as_completed(futures):
            cidx, out_text = fut.result()
            results[cidx] = out_text
    parts = [r if r is not None else "" for r in results]

    merged_row = roc._merge_row_from_chunk_outputs(
        cfg,
        pq_name=shard_path.name,
        row_idx=local_i,
        full_text=text,
        parts=parts,
    )
    key = "programs_delimited" if cfg.output_is_code else "text"
    after_text = merged_row.get(key, "")
    _show_live_query_diff_html(
        input_raw_text,
        denoised_text,
        after_text,
        meta_base,
        refiner_parquet_loaded=refiner_parquet_loaded,
        live_tabs=live_tabs,
        max_chars=max_chars,
        no_browser=no_browser,
        wait_sec=wait_sec,
        keep_alive_sec=keep_alive_sec,
        port=port,
    )


def _simple_diff_merge_html(before: str, after: str) -> str:
    """Single-stream diff: red strikethrough deletions, green insertions (``SequenceMatcher`` opcodes)."""
    sm = SequenceMatcher(None, before, after, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(before[i1:i2]))
        elif tag == "delete":
            parts.append(f'<del class="diff-del">{html.escape(before[i1:i2])}</del>')
        elif tag == "insert":
            parts.append(f'<ins class="diff-ins">{html.escape(after[j1:j2])}</ins>')
        elif tag == "replace":
            parts.append(f'<del class="diff-del">{html.escape(before[i1:i2])}</del>')
            parts.append(f'<ins class="diff-ins">{html.escape(after[j1:j2])}</ins>')
    return "".join(parts)


def _meta_lines(meta: dict[str, Any], *, raw_text: str, middle_text: str) -> str:
    raw_note = ""
    rdir = meta.get("raw_parquet_dir")
    if rdir is not None and raw_text != middle_text:
        raw_note = f"\nraw corpus dir: {html.escape(str(rdir))}"
    ld_note = ""
    if meta.get("largest_dif_score") is not None:
        cc = meta.get("largest_dif_compare_chars", 0)
        cc_s = "full strings" if cc == 0 else str(cc)
        rk = meta.get("largest_dif_k")
        rank_s = f"  k={rk} (rank by dissimilarity score)" if rk is not None else ""
        ld_note = (
            f"\n[largest-dif] dissimilarity score (quick_ratio proxy; middle vs output for ranking)="
            f"{meta['largest_dif_score']:.6g}  compare_chars={cc_s}{rank_s}"
        )
    live_note = ""
    lid = meta.get("live_input_dir")
    lrp = meta.get("live_refiner_parquet")
    if lid is not None:
        live_note += f"\ninput_parquet_dir: {html.escape(str(lid))}"
    if lrp is not None:
        live_note += f"\nrefiner output parquet: {html.escape(str(lrp))}"
    elif lid is not None:
        live_note += (
            "\nrefiner output parquet: (none — set output_data_dir in dataset YAML and run denoise)"
        )
    return (
        f"{html.escape(str(meta['yaml_config']))}\n"
        f"shard: {html.escape(str(meta['shard_path']))}\n"
        f"OpenRouter / stored output: {html.escape(str(meta['output_parquet']))}"
        f"{raw_note}{live_note}{ld_note}"
    )


def _build_simple_diff_page(
    *,
    doc_title: str,
    h1: str,
    legend: str,
    merged: str,
    meta_block: str,
    truncated: bool,
) -> str:
    warn = (
        '<p class="warn">Truncated; see script <code>--max-chars</code>.</p>' if truncated else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(doc_title)}</title>
<style>
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         margin: 1rem 1.5rem; line-height: 1.45; background: #1e1e1e; color: #d4d4d4; }}
  .meta {{ color: #9cdcfe; margin-bottom: 1rem; font-size: 0.85rem; white-space: pre-wrap; }}
  .warn {{ color: #f9c74f; }}
  .legend {{ color: #858585; font-size: 0.82rem; margin: 0 0 1rem; max-width: 52rem; }}
  pre.diff {{ white-space: pre-wrap; word-break: break-word; margin: 0; }}
  .diff-del {{ background: rgba(244, 67, 54, 0.42); text-decoration: line-through; color: #ffc9c9; }}
  .diff-ins {{ background: rgba(76, 175, 80, 0.48); color: #e8ffe8; text-decoration: none; }}
</style></head><body>
<h1>{html.escape(h1)}</h1>
<p class="legend">{html.escape(legend)}</p>
{warn}
<div class="meta">{meta_block}</div>
<pre class="diff">{merged}</pre>
</body></html>"""


def _html_pair_for_row(
    raw_text: str,
    middle_text: str,
    out_text: str,
    meta: dict[str, Any],
    truncated: bool,
    *,
    title_suffix: str = "",
) -> tuple[str, str]:
    """Return (output_stage_html, openrouter_stage_html)."""
    row_label = f"row {meta['global_idx']} (local {meta['local_row']}){title_suffix}"
    merged_out = _simple_diff_merge_html(raw_text, middle_text)
    final_clean = strip_openrouter_chunk_delimiters(out_text)
    merged_or = _simple_diff_merge_html(raw_text, final_clean)
    meta_block = _meta_lines(meta, raw_text=raw_text, middle_text=middle_text)
    html_out = _build_simple_diff_page(
        doc_title=f"Output · {row_label}",
        h1=f"Output (raw → middle) · {row_label}",
        legend="Red = removed going from first to second string; green = added. Compares raw corpus vs middle (OpenRouter input row).",
        merged=merged_out,
        meta_block=meta_block,
        truncated=truncated,
    )
    html_or = _build_simple_diff_page(
        doc_title=f"OpenRouter · {row_label}",
        h1=f"OpenRouter (raw input → model output) · {row_label}",
        legend=(
            "Red = removed from raw/noisy input; green = added in OpenRouter output. "
            "Chunk-merge markers are stripped from the stored output before diffing."
        ),
        merged=merged_or,
        meta_block=meta_block,
        truncated=truncated,
    )
    return html_out, html_or


def _html_pair_for_live_query(
    input_raw_text: str,
    denoised_text: str,
    openrouter_response: str,
    meta: dict[str, Any],
    truncated: bool,
    *,
    refiner_parquet_loaded: bool,
    title_suffix: str = "",
) -> tuple[str, str]:
    """Live query: tab 1 = input vs denoised; tab 2 = raw input vs merged API response (not denoised vs API)."""
    row_label = f"row {meta['global_idx']} (local {meta['local_row']}){title_suffix}"
    merged_out = _simple_diff_merge_html(input_raw_text, denoised_text)
    merged_or = _simple_diff_merge_html(
        input_raw_text, strip_openrouter_chunk_delimiters(openrouter_response)
    )
    meta_block = _meta_lines(meta, raw_text=input_raw_text, middle_text=denoised_text)
    legend_out = (
        "Red = removed from raw input; green = added in denoised/refiner output. "
        "Compares ``input_parquet_dir`` vs ``output_data_dir`` row."
        if refiner_parquet_loaded
        else (
            "No denoised parquet loaded (missing ``output_data_dir`` or shard file); "
            "input compared to itself."
        )
    )
    html_out = _build_simple_diff_page(
        doc_title=f"Refiner · {row_label}",
        h1=f"Input → denoised (refiner) · {row_label}",
        legend=legend_out,
        merged=merged_out,
        meta_block=meta_block,
        truncated=truncated,
    )
    html_or = _build_simple_diff_page(
        doc_title=f"OpenRouter · {row_label}",
        h1=f"OpenRouter (raw input → API response) · {row_label}",
        legend=(
            "Red = removed from raw/noisy input; green = added in OpenRouter response. "
            "The API is still called on denoised chunks; this view only diffs the original input string "
            "to the merged reply. Chunk-merge markers stripped from the response before diffing."
        ),
        merged=merged_or,
        meta_block=meta_block,
        truncated=truncated,
    )
    return html_out, html_or


def _serve_ephemeral_html_multi(
    routes: dict[str, str],
    *,
    open_browser: bool,
    open_paths: list[str],
    wait_first_request_sec: float,
    keep_alive_after_first_sec: float,
    bind_port: int,
    quiet: bool = False,
) -> None:
    """Serve multiple paths on one server; optionally open several browser URLs."""
    bodies: dict[str, bytes] = {p: doc.encode("utf-8") for p, doc in routes.items()}
    got_any = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/index.html":
                path = "/"
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            body = bodies.get(path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            got_any.set()

        def log_message(self, *_args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", bind_port), _Handler)
    host, port = httpd.server_address[:2]
    base = f"http://{host}:{port}"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        if open_browser:
            for p in open_paths:
                webbrowser.open(f"{base}{p}")
                time.sleep(0.12)
        if not quiet:
            paths_s = " ".join(open_paths)
            print(f"Ephemeral diff server: {base}  (paths: {paths_s})", flush=True)
        if not got_any.wait(timeout=wait_first_request_sec):
            if not quiet:
                print(
                    "[warn] No GET received; open the URL(s) manually if needed.",
                    file=sys.stderr,
                )
        else:
            time.sleep(keep_alive_after_first_sec)
    finally:
        httpd.shutdown()
        httpd.server_close()


def _serve_ephemeral_html(
    html_doc: str,
    *,
    open_browser: bool,
    wait_first_request_sec: float,
    keep_alive_after_first_sec: float,
    bind_port: int,
    quiet: bool = False,
) -> None:
    """Serve a single document at ``/`` (compat wrapper)."""
    _serve_ephemeral_html_multi(
        {"/": html_doc},
        open_browser=open_browser,
        open_paths=["/"],
        wait_first_request_sec=wait_first_request_sec,
        keep_alive_after_first_sec=keep_alive_after_first_sec,
        bind_port=bind_port,
        quiet=quiet,
    )


def _parse_doc_index_str(s: str, ap: argparse.ArgumentParser) -> int:
    try:
        return int(s)
    except ValueError:
        ap.error(f"Doc index must be an integer, got {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "paths",
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Without ``--live-query``: ``VIEWER_YAML DOC_INDEX`` for parquet diff, or a single "
            "``VIEWER_YAML`` with ``--largest-dif``. "
            "With ``--live-query``: three args ``DATASET_YAML OPENROUTER_YAML DOC_INDEX``, or "
            "two args ``openrouter_generation/CONFIG.yaml DOC_INDEX``. With ``--largest-dif``, one "
            "refiner_generation YAML "
            "or two YAMLs (never largest-dif-only with a lone openrouter_generation YAML)."
        ),
    )
    ap.add_argument(
        "--largest-dif",
        action="store_true",
        help=(
            "Pick the document by ranked dissimilarity (fast ``quick_ratio`` proxy: middle vs parquet "
            "output); use ``--largest-dif-k``. Do not pass the trailing doc index."
        ),
    )
    ap.add_argument(
        "--largest-dif-k",
        type=int,
        default=1,
        metavar="K",
        help=(
            "With --largest-dif: ``1`` = largest dissimilarity score, ``2`` = second largest, … "
            "(default 1)."
        ),
    )
    ap.add_argument(
        "--largest-dif-compare-chars",
        type=int,
        default=8192,
        metavar="N",
        help=(
            "With --largest-dif: compare only the first N characters of middle vs output for ranking "
            "(default 8192). Use 0 for full strings (linear-time proxy per row; large corpora still add up)."
        ),
    )
    ap.add_argument("--one-based", action="store_true", help="Interpret n as 1-based.")
    ap.add_argument(
        "--live-query",
        action="store_true",
        help=(
            "HTTP OpenRouter for one row (requires OPENROUTER_API_KEY unless refiner-static largest-dif). "
            "Pass three args ``DATASET_YAML OPENROUTER_YAML DOC_INDEX``, or two args "
            "``openrouter_generation/*.yaml DOC_INDEX``. "
            "With ``--largest-dif``: two YAMLs (rank + API) **or** one YAML only under ``refiner_data_generation/`` "
            "(static). Lone ``openrouter_data_generation/`` + largest-dif is **disallowed** — use explicit index."
        ),
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=800_000,
        help="Max combined chars before truncation for diff / live HTML (default 800000).",
    )
    ap.add_argument("--no-browser", action="store_true", help="Do not open the default web browser.")
    ap.add_argument(
        "--keep-alive-sec",
        type=float,
        default=3.0,
        help="Seconds to keep serving after the first GET / (default 3).",
    )
    ap.add_argument(
        "--wait-sec",
        type=float,
        default=90.0,
        help="Max seconds to wait for the browser to request / before giving up (default 90).",
    )
    ap.add_argument("--port", type=int, default=0, help="Bind port (0 = ephemeral).")
    args = ap.parse_args()

    if args.largest_dif and args.largest_dif_k < 1:
        ap.error("--largest-dif-k must be >= 1")

    raw_paths = args.paths
    if not raw_paths:
        ap.error("Provide YAML path(s); see --help")

    dataset_yaml: Path | None = None
    openrouter_yaml: Path | None = None
    viewer_yaml: Path | None = None
    doc_index: int | None = None
    live_query_single_side: str | None = None  # "refiner" | "openrouter" when len==1 + largest-dif

    if args.live_query:
        if args.largest_dif:
            if len(raw_paths) == 1:
                dataset_yaml = Path(raw_paths[0]).expanduser().resolve()
                live_query_single_side = _live_largest_dif_single_yaml_side(dataset_yaml, ap)
                openrouter_yaml = None
            elif len(raw_paths) == 2:
                dataset_yaml = Path(raw_paths[0]).expanduser().resolve()
                openrouter_yaml = Path(raw_paths[1]).expanduser().resolve()
            else:
                ap.error(
                    "--live-query --largest-dif takes 1 or 2 YAML paths: "
                    "REFINER_YAML only (static) or DATASET_YAML MODEL_YAML (rank + live API). "
                    "A lone openrouter_generation YAML with --largest-dif is not allowed — use explicit DOC_INDEX "
                    "`--live-query CONFIG.yaml N` instead."
                )
        else:
            if len(raw_paths) == 3:
                dataset_yaml = Path(raw_paths[0]).expanduser().resolve()
                openrouter_yaml = Path(raw_paths[1]).expanduser().resolve()
                doc_index = _parse_doc_index_str(raw_paths[2], ap)
            elif len(raw_paths) == 2:
                p0 = Path(raw_paths[0]).expanduser().resolve()
                rp, og_kind = _repo_subdir_yaml_kind(p0)
                if og_kind != "openrouter":
                    ap.error(
                        "--live-query with **2** arguments is only allowed when YAML is under "
                        f"{_OPENROUTER_GEN_DIR}: OPENROUTER_CONFIG.yaml DOC_INDEX (got {rp})."
                    )
                dataset_yaml = p0
                openrouter_yaml = p0
                doc_index = _parse_doc_index_str(raw_paths[1], ap)
            else:
                ap.error(
                    "--live-query requires **3** arguments DATASET_YAML OPENROUTER_YAML DOC_INDEX, "
                    f"or **2** OPENROUTER_GEN_CONFIG.yaml DOC_INDEX (under {_OPENROUTER_GEN_DIR})."
                )
    elif args.largest_dif:
        if len(raw_paths) != 1:
            ap.error("--largest-dif requires exactly 1 YAML (viewer config with existing output parquets)")
        viewer_yaml = Path(raw_paths[0]).expanduser().resolve()
    else:
        if len(raw_paths) != 2:
            ap.error(
                "Without --live-query: pass two arguments: VIEWER_YAML DOC_INDEX "
                "(or one YAML with --largest-dif)"
            )
        viewer_yaml = Path(raw_paths[0]).expanduser().resolve()
        doc_index = _parse_doc_index_str(raw_paths[1], ap)

    if args.live_query and args.largest_dif:
        assert dataset_yaml is not None
        pn = _load_print_nth_module()
        os.chdir(_REPO_ROOT)
        roc = _load_run_openrouter_module()
        try:
            raw_text, middle_text, out_text, meta, score = (
                find_row_kth_largest_dissimilarity_middle_vs_final(
                    pn,
                    dataset_yaml,
                    compare_chars=args.largest_dif_compare_chars,
                    k=args.largest_dif_k,
                )
            )
            cc = args.largest_dif_compare_chars
            cc_desc = "full strings" if cc == 0 else f"first {cc} chars of each"
            doc_idx = meta["global_idx"]
            meta = dict(meta)

            if live_query_single_side == "refiner":
                print(
                    f"[largest-dif→refiner] k={args.largest_dif_k}  global_row={doc_idx}  "
                    f"score={score:.6g}  ({cc_desc} vs stored denoised); static HTML only (no OpenRouter).",
                    flush=True,
                )
                lim = args.max_chars
                truncated = False
                if len(raw_text) + len(middle_text) + len(out_text) > lim:
                    n = max(1, lim // 3)
                    raw_text = raw_text[:n]
                    middle_text = middle_text[:n]
                    out_text = out_text[:n]
                    truncated = True
                merged_rf = _simple_diff_merge_html(
                    middle_text, strip_openrouter_chunk_delimiters(out_text)
                )
                row_label = (
                    f"row {meta['global_idx']} (local {meta['local_row']}) — "
                    "largest-dif (refiner)"
                )
                html_doc = _build_simple_diff_page(
                    doc_title=f"Refiner · {row_label}",
                    h1=f"Input → denoised (refiner) · {row_label}",
                    legend=(
                        "Red = removed from refiner input; green = added in denoised output. "
                        "Ranked vs stored parquet; no HTTP request."
                    ),
                    merged=merged_rf,
                    meta_block=_meta_lines(meta, raw_text=raw_text, middle_text=middle_text),
                    truncated=truncated,
                )
                if not args.no_browser:
                    _serve_ephemeral_html(
                        html_doc,
                        open_browser=True,
                        wait_first_request_sec=args.wait_sec,
                        keep_alive_after_first_sec=args.keep_alive_sec,
                        bind_port=args.port,
                    )
                else:
                    print("Built HTML in memory only (--no-browser); not serving.", flush=True)
            else:
                print(
                    f"[largest-dif→live-query] k={args.largest_dif_k}  global_row={doc_idx}  "
                    f"dissimilarity score={score:.6g}  ({cc_desc} for ranking); calling OpenRouter…",
                    flush=True,
                )
                cfg, mq_resolved = resolve_live_query_run_config(
                    roc,
                    dataset_yaml,
                    openrouter_yaml,
                )
                if live_query_single_side == "openrouter":
                    print(
                        "Live query (single openrouter_generation YAML): /openrouter only; "
                        "chunks echo to the terminal.",
                        flush=True,
                    )
                else:
                    print(
                        "Live OpenRouter query: two tabs ( / = refiner, /openrouter = noisy vs API ); "
                        "chunk lines echo to the terminal.",
                        flush=True,
                    )
                run_live_openrouter_query(
                    roc,
                    cfg=cfg,
                    doc_index=doc_idx,
                    one_based=False,
                    dataset_yaml=dataset_yaml,
                    model_yaml=mq_resolved,
                    max_chars=args.max_chars,
                    no_browser=args.no_browser,
                    wait_sec=args.wait_sec,
                    keep_alive_sec=args.keep_alive_sec,
                    port=args.port,
                    live_tabs=(
                        "openrouter_only"
                        if live_query_single_side == "openrouter"
                        else "both"
                    ),
                )
        except IndexError as e:
            print(e, file=sys.stderr)
            sys.exit(2)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        return

    if args.live_query:
        assert dataset_yaml is not None and openrouter_yaml is not None and doc_index is not None
        os.chdir(_REPO_ROOT)
        roc = _load_run_openrouter_module()
        try:
            cfg, mq_resolved = resolve_live_query_run_config(
                roc,
                dataset_yaml,
                openrouter_yaml,
            )
            print(
                "Live OpenRouter query: two tabs ( / = output stage, /openrouter = model vs row ); "
                "chunk lines echo to the terminal.",
                flush=True,
            )
            run_live_openrouter_query(
                roc,
                cfg=cfg,
                doc_index=doc_index,
                one_based=args.one_based,
                dataset_yaml=dataset_yaml,
                model_yaml=mq_resolved,
                max_chars=args.max_chars,
                no_browser=args.no_browser,
                wait_sec=args.wait_sec,
                keep_alive_sec=args.keep_alive_sec,
                port=args.port,
            )
        except IndexError as e:
            print(e, file=sys.stderr)
            sys.exit(2)
        return

    assert viewer_yaml is not None
    pn = _load_print_nth_module()
    try:
        if args.largest_dif:
            raw_text, middle_text, out_text, meta, score = (
                find_row_kth_largest_dissimilarity_middle_vs_final(
                    pn,
                    viewer_yaml,
                    compare_chars=args.largest_dif_compare_chars,
                    k=args.largest_dif_k,
                )
            )
            cc = args.largest_dif_compare_chars
            cc_desc = "full strings" if cc == 0 else f"first {cc} chars of each"
            print(
                f"[largest-dif] k={args.largest_dif_k}  global_row={meta['global_idx']}  "
                f"dissimilarity score={score:.6g}  ({cc_desc} for ranking)",
                flush=True,
            )
        else:
            assert doc_index is not None
            raw_text, middle_text, out_text, meta = pn.load_input_output_for_yaml_row(
                viewer_yaml, doc_index, one_based=args.one_based
            )
    except IndexError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    lim = args.max_chars
    truncated = False
    if len(raw_text) + len(middle_text) + len(out_text) > lim:
        third = max(1, lim // 3)
        raw_text = raw_text[:third]
        middle_text = middle_text[:third]
        out_text = out_text[:third]
        truncated = True

    meta = dict(meta)
    html_out, html_or = _html_pair_for_row(
        raw_text, middle_text, out_text, meta, truncated
    )
    print(
        "Simple red/green diffs (SequenceMatcher, autojunk=False): "
        "/ = raw→middle, /openrouter = raw input→OpenRouter output.",
        flush=True,
    )

    if not args.no_browser:
        _serve_ephemeral_html_multi(
            {"/": html_out, "/openrouter": html_or},
            open_browser=True,
            open_paths=["/", "/openrouter"],
            wait_first_request_sec=args.wait_sec,
            keep_alive_after_first_sec=args.keep_alive_sec,
            bind_port=args.port,
        )
    else:
        print("Built HTML in memory only (--no-browser); not serving.", flush=True)


if __name__ == "__main__":
    main()
