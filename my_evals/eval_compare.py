#!/usr/bin/env python3
"""
Compare lm-eval results across multiple checkpoints with a grouped bar chart.

Usage:
    python my_evals/eval_compare.py my_evals/example_eval_compare.yaml

Input YAML (see ``my_evals/example_eval_compare.yaml``):

    models:
      step_4:    /path/to/global_step_4/hf_ckpt
      step_2000: /path/to/global_step_2000/hf_ckpt
    tasks:
      - mmlu
      - mmlu_stem
      - wikitext            # auto-picks ``word_perplexity``
      # - wikitext:bits_per_byte   # explicit metric override

Each ``models[*]`` value is the directory you previously passed to
``my_evals/eval_lm_eval.sh``. The script searches for the most recent
``results_*.json`` under ``<path>/lm_eval_results/`` (or
``<path>/hf_ckpt/lm_eval_results/`` if the former doesn't exist), reads the
metric for each requested task, and draws one subplot per task with bars for
each model. The figure is written to a tempfile and opened with the system's
default image viewer (``xdg-open`` on Linux, ``open`` on macOS); we don't
keep a copy in the repo.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------
def _all_results_jsons(model_dir: Path) -> List[Path]:
    """Return every ``results_*.json`` under the model dir, newest first.

    Looks under ``<model_dir>/lm_eval_results/**/results_*.json``. Also tries
    ``<model_dir>/hf_ckpt/lm_eval_results/...`` since users sometimes point
    at ``global_step_<N>`` rather than ``.../hf_ckpt``.
    """
    candidates: List[Path] = []
    for root in (model_dir, model_dir / "hf_ckpt"):
        candidates.extend((root / "lm_eval_results").glob("**/results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No results_*.json found under {model_dir}/lm_eval_results "
            f"(or {model_dir}/hf_ckpt/lm_eval_results). Run "
            f"my_evals/eval_lm_eval.sh against this checkpoint first."
        )
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_task_aware(model_dir: Path, tasks: List[str]) -> Dict[str, Dict[str, Any]]:
    """For each task, return the parsed JSON of the *newest* run that
    contains that task. A user might have run ``--tasks mmlu,wikitext`` once
    and ``--tasks wikitext`` again later: blindly taking the latest file
    would lose mmlu. Indexing per task keeps every previously-evaluated
    metric available.
    """
    out: Dict[str, Dict[str, Any]] = {}
    files = _all_results_jsons(model_dir)
    cache: Dict[Path, Dict[str, Any]] = {}
    for task in tasks:
        for fp in files:
            data = cache.get(fp)
            if data is None:
                with fp.open() as f:
                    cache[fp] = data = json.load(f)
            if task in data.get("results", {}) or task in data.get("groups", {}):
                out[task] = data
                break
    return out


def _split_task_spec(spec: str) -> Tuple[str, Optional[str]]:
    """``"wikitext:bits_per_byte"`` -> ``("wikitext", "bits_per_byte")``."""
    if ":" in spec:
        task, metric = spec.split(":", 1)
        return task.strip(), metric.strip()
    return spec.strip(), None


_PERPLEXITY_METRIC_PRIORITY = ("word_perplexity", "byte_perplexity", "bits_per_byte")


def _auto_pick_metric(task_entry: Dict[str, Any], higher_is_better: Dict[str, bool]) -> str:
    """Pick a sensible default metric for a task entry.

    Strategy: prefer ``acc`` if the task reports it (MMLU et al.). Else fall
    back to the first known perplexity-style metric. Else first numeric
    metric in ``higher_is_better``. We want a single, reproducible choice
    rather than something fancy.
    """
    metric_keys = {k.split(",", 1)[0] for k in task_entry if "," in k and not k.endswith("_stderr")}
    if "acc" in metric_keys and (higher_is_better is None or higher_is_better.get("acc", True)):
        return "acc"
    for m in _PERPLEXITY_METRIC_PRIORITY:
        if m in metric_keys:
            return m
    if higher_is_better:
        for m in higher_is_better:
            if m in metric_keys:
                return m
    if metric_keys:
        return sorted(metric_keys)[0]
    raise KeyError(f"no usable metric found in task entry; keys={list(task_entry)}")


def _read_task_metric(
    results_json: Dict[str, Any], task: str, metric_override: Optional[str]
) -> Tuple[float, str, bool]:
    """Return ``(value, metric_name, higher_is_better_flag)`` for ``task``.

    Searches both ``results`` (covers groups + subtasks in lm_eval >= 0.4) and
    the older ``groups`` block, in that order.
    """
    sources = [results_json.get("results", {}), results_json.get("groups", {})]
    entry = next((s[task] for s in sources if isinstance(s, dict) and task in s), None)
    if entry is None:
        all_tasks = sorted(set(results_json.get("results", {})) | set(results_json.get("groups", {})))
        raise KeyError(
            f"task {task!r} not found in results JSON. Available: {all_tasks[:12]}"
            + (" ..." if len(all_tasks) > 12 else "")
        )
    hib = (results_json.get("higher_is_better") or {}).get(task) or {}
    metric = metric_override or _auto_pick_metric(entry, hib)
    val = entry.get(f"{metric},none")
    if val is None:
        raise KeyError(
            f"metric {metric!r} not present for task {task!r}. "
            f"Available metrics: {sorted({k.split(',', 1)[0] for k in entry if ',' in k})}"
        )
    if not isinstance(val, (int, float)):
        raise TypeError(f"metric {task}:{metric} is non-numeric ({val!r}); cannot plot.")
    return float(val), metric, bool(hib.get(metric, True))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot(
    task_specs: List[Tuple[str, Optional[str]]],
    model_names: List[str],
    metrics_grid: List[List[Optional[Tuple[float, str, bool]]]],
    out_path: Path,
) -> None:
    """One subplot per task; bars per model on each subplot.

    Different tasks usually use different metric scales (accuracy in [0,1] vs
    perplexity in [1, ~50]), so plotting them on a shared y-axis would just
    flatten one of them. Subplotting keeps each task readable.
    """
    n = len(task_specs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, (ax, (task, _override)) in enumerate(zip(axes_flat, task_specs)):
        col = metrics_grid[ax_idx]
        ys: List[float] = []
        for cell in col:
            ys.append(float("nan") if cell is None else cell[0])

        bars = ax.bar(model_names, ys)
        metric_name = next((c[1] for c in col if c is not None), "?")
        higher_better = next((c[2] for c in col if c is not None), True)
        arrow = "↑" if higher_better else "↓"
        ax.set_title(f"{task}\n({metric_name} {arrow})")
        ax.tick_params(axis="x", rotation=20)
        for bar, y in zip(bars, ys):
            if y == y:  # not nan
                ax.annotate(
                    f"{y:.3f}" if abs(y) < 100 else f"{y:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

    for ax in axes_flat[len(task_specs):]:
        ax.set_visible(False)

    fig.suptitle("lm_eval comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _open_image(path: Path) -> None:
    """Open ``path`` in whatever viewer is available. Best-effort, non-blocking.

    We try a list of openers because remote dev boxes rarely have ``xdg-open``
    installed, but they very often have the IDE's CLI (``cursor``/``code``)
    on PATH which can preview PNGs inline. Order:

      * macOS  -> ``open``
      * Windows -> ``os.startfile``
      * Linux  -> ``cursor`` / ``code`` (Cursor / VSCode remote-cli, inline
                  preview), then graphical fallbacks (``xdg-open``,
                  ``gio open``, ``eog``, ``feh``, ``display``).
    """
    if sys.platform == "darwin":
        candidates = [["open", str(path)]]
    elif os.name == "nt":
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as e:
            print(f"[eval_compare] could not open {path}: {e}", flush=True)
        return
    else:
        candidates = [
            ["cursor", str(path)],
            ["code", str(path)],
            ["xdg-open", str(path)],
            ["gio", "open", str(path)],
            ["eog", str(path)],
            ["feh", str(path)],
            ["display", str(path)],
        ]

    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            print(f"[eval_compare] {cmd[0]!r} failed: {e}", flush=True)
            continue
        print(f"[eval_compare] opened with {cmd[0]!r}: {path}", flush=True)
        return

    print(
        f"[eval_compare] no image viewer found on PATH; image is at {path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} <config.yaml>")
    cfg_path = Path(argv[1]).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    raw_models = cfg.get("models")
    if isinstance(raw_models, dict):
        models = list(raw_models.items())
    elif isinstance(raw_models, list):
        models = []
        for item in raw_models:
            if isinstance(item, dict) and "name" in item and "path" in item:
                models.append((item["name"], item["path"]))
            else:
                raise ValueError(f"unexpected model entry: {item!r}")
    else:
        raise ValueError("config 'models' must be a mapping or list of {name, path}.")
    if not models:
        raise SystemExit("config has no models to compare.")

    raw_tasks = cfg.get("tasks") or []
    task_specs: List[Tuple[str, Optional[str]]] = [_split_task_spec(t) for t in raw_tasks]
    if not task_specs:
        raise SystemExit("config has no tasks to plot.")

    metrics_grid: List[List[Optional[Tuple[float, str, bool]]]] = [
        [None] * len(models) for _ in task_specs
    ]
    for mi, (mname, mpath) in enumerate(models):
        try:
            per_task = _load_task_aware(
                Path(mpath).expanduser(), [t for t, _ in task_specs]
            )
        except FileNotFoundError as e:
            print(f"[eval_compare] WARN: skipping {mname!r}: {e}", file=sys.stderr)
            continue
        if not per_task:
            print(
                f"[eval_compare] WARN {mname!r}: none of the requested tasks were "
                f"found in any results_*.json under {mpath}",
                file=sys.stderr,
            )
        print(
            f"[eval_compare] {mname}: matched tasks -> "
            f"{ {t: '<run found>' for t in per_task} }",
            flush=True,
        )
        for ti, (task, override) in enumerate(task_specs):
            r = per_task.get(task)
            if r is None:
                print(
                    f"[eval_compare]   WARN {mname}/{task}: no results JSON "
                    f"under {mpath} contains this task; skipping bar.",
                    file=sys.stderr,
                )
                continue
            try:
                metrics_grid[ti][mi] = _read_task_metric(r, task, override)
            except (KeyError, TypeError) as e:
                print(f"[eval_compare]   WARN {mname}/{task}: {e}", file=sys.stderr)

    if all(cell is None for row in metrics_grid for cell in row):
        raise SystemExit("No data points found; nothing to plot.")

    out = Path(tempfile.mkdtemp(prefix="eval_compare_")) / "eval_compare.png"
    _plot(task_specs, [m for m, _ in models], metrics_grid, out)
    print(f"[eval_compare] image: {out}", flush=True)
    _open_image(out)


if __name__ == "__main__":
    main(sys.argv)
