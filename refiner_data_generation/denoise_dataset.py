#!/usr/bin/env python3
"""
Denoise web/code corpus shards with a local model. Two pipelines share the same shard-iteration,
chunking, batching, output, and resume scaffolding:

  - **Autoregressive (default)**: load the model under vLLM and rewrite each chunk with greedy
    generation (same chunking + chat pattern as ``finetuning/finetune.py`` eval / OpenRouter
    generation).
  - **Embedding (per-token binary keep/discard)**: set ``is_embedding_model: true`` to load the
    finetuned ``Qwen2EmbeddingForTokenClassification`` model from
    ``finetuning/finetune.py is_embedding_model: true`` runs. The pipeline does **not** use vLLM:
    each chunk is tokenized with ``return_offsets_mapping=True``, run through the model, and the
    chunk is reconstructed by concatenating the source characters covered by tokens whose binary
    score-logit is greater than ``keep_logit_threshold`` (default ``0.0``; equivalent to
    ``sigmoid(logit) > 0.5``). Tokens whose offset span is empty (special tokens) are skipped.

Output schema (``{shard_stem}_denoised.parquet``) is identical for both pipelines so downstream
tools (parquet readers, the OpenRouter-style viewer, etc.) need no changes.

YAML keys (required unless noted):
  - input_parquet_dir: directory containing ``*.parquet`` or ``*.jsonl.zst`` shards (recursive)
  - dataset_type: fineweb | fine_web | dclm | redpajama | redpajama-v2 | passthrough
  - output_data_dir: output directory for denoised parquets + copied YAML
  - overwrite: allow writing into an existing output dir (except resume)
  - continue: resume when output YAML matches a **resume** fingerprint (data shards, chunking,
    etc.). ``model_checkpoint_path`` may change (e.g. a newer ``global_step_*`` from continued
    training); incomplete ``hf_ckpt/`` from a failed DCP merge is removed and re-merged automatically.
  - chunk_size: character chunk size (same helper as OpenRouter / finetune)
  - model_arch: string (informational; merged into fingerprint like finetune; vLLM loads HF config)
  - model_checkpoint_path: HuggingFace-format weights directory **or** a VeOmni DCP step folder
    (``.../checkpoints/global_step_<N>`` from ``finetuning.py``). DCP is merged on first run into
    ``<step_dir>/hf_ckpt`` using ``finetuning/veomni/scripts/merge_dcp_to_hf.py`` (needs
    ``<run_root>/.finetune_inline_config/config.json`` or ``model_assets`` from the same finetune run).
  - tokenizer_path: optional; defaults to merged/HF checkpoint dir — set explicitly when that dir has no tokenizer (typical for DCP→HF export + hub tokenizer)
  - output_data_dir, overwrite, continue, ...
  - num_gpus: tensor parallel size for vLLM **per replica** (must divide model attention heads).
    Embedding mode does NOT use TP; set ``num_gpus: 1`` and use ``data_parallel_replicas`` for throughput.
  - data_parallel_replicas: number of independent full-model workers (each uses ``num_gpus`` GPUs with TP).
    Set to ``1`` (default) for a single process. Use ``8`` with ``num_gpus: 1`` for eight GPUs each running TP=1 (throughput on small models). Shards are partitioned round-robin across replicas; output files do not overlap.
  - data_parallel_gpu_ids: optional explicit CUDA device indices (length must equal ``data_parallel_replicas``); default ``0 .. replicas-1``
  - max_docs, max_chars, max_chunks, max_tokens: caps (-1 = unused); **stop when any cap hits**
  - max_parquets: max input shard files after sorting (-1 = all)
  - is_code: false = rewrite ``text`` / primary column with merged chunk outputs;
              true = fill ``programs_delimited`` + ``programs`` like OpenRouter code mode
  - chunk_batch_size: number of chunks per forward pass (throughput).
    Autoregressive mode → ``llm.generate`` batch; embedding mode → tokenizer + ``model(...)`` batch.
  - max_new_tokens: autoregressive only — ignored in embedding mode.
  - max_model_len: max prompt/forward sequence length. Autoregressive mode → vLLM ``max_model_len``;
    embedding mode → tokenizer ``max_length=`` and the chunk's tail is dropped if truncated.
  - trust_remote_code: generation / engine
  - metrics_interval_sec: periodic throughput logs (same style as OpenRouter runner)
  - vllm: optional mapping of extra kwargs forwarded to ``vllm.LLM(...)`` (autoregressive only)
  - is_embedding_model: bool (default false). Switch to the embedding (per-token keep/discard) pipeline.
  - keep_logit_threshold: float (default 0.0). Embedding mode only. Tokens with score-logit greater
    than this value are kept; everything else is dropped from the reconstructed chunk text.

Environment:
  - PROX_DCP_MERGE_PYTHON — optional path to a Python executable for **DCP→HF merge only**
    (defaults to ``sys.executable``). Use when your vLLM conda env has an old PyTorch (e.g. 2.1)
    without ``torch.distributed.checkpoint.load``, but you have another env (often the VeOmni
    training env) with PyTorch≥2.4 and a compatible Transformers for VeOmni.

The resolved YAML path is copied into ``output_data_dir`` at run start.

Within each output shard parquet, rows are appended in **strict input row order** (same order as
``iter_shard_rows``), even when chunk batches finish out of order across documents.

Usage:
  python refiner_data_generation/denoise_dataset.py refiner_data_generation/example_denoise.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field


def _exclude_user_site_from_sys_path() -> None:
    """Drop PEP 370 user-site (``~/.local/.../site-packages``) entries from ``sys.path``.

    When ``pip install --user`` has placed ``numpy`` / ``scikit-learn`` there, they can be
    imported ahead of the active conda/venv and disagree with that env's NumPy ABI (breaking
    ``transformers`` → ``sklearn``). Conda env wins once user-site is not on the path.
    """
    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site:
        return
    nu = user_site.rstrip(os.sep)
    sys.path[:] = [p for p in sys.path if p.rstrip(os.sep) != nu]


_exclude_user_site_from_sys_path()
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from e

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required in the active conda/venv (not ``pip install --user``): "
        "``pip install pyarrow`` or ``pip install -r refiner_data_generation/requirements_denoise_conda.txt``"
    ) from e

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openrouter_data_generation.run_openrouter_chunked import (  # noqa: E402
    PROGRAM_CHUNK_SEPARATOR,
    chunk_text,
    get_row_text_fn,
    iter_shard_rows,
    precount_chars_and_token_est,
    precount_doc_rows,
    sorted_shard_paths,
)
from openrouter_data_generation.run_openrouter_chunked import (  # noqa: E402
    Metrics,
)
from openrouter_data_generation.run_openrouter_chunked import (  # noqa: E402
    _api_tok_denom,
    _char_denom,
    _doc_denom,
    _boolish,
)

SYSTEM_MSG = "You are a helpful, respectful and honest assistant."

_VEOMNI_SRC = _REPO_ROOT / "finetuning" / "veomni"

# HuggingFace ``auto_map`` for VeOmni-only ``model_type`` values so ``transformers`` / vLLM can import
# modeling code when ``PYTHONPATH`` includes ``finetuning/veomni`` (this script adds it).
_VEOMNI_AUTO_MAP_BY_MODEL_TYPE: dict[str, dict[str, str]] = {
    "test_everything": {
        "AutoConfig": (
            "veomni.models.transformers.test_everything.configuration_test_everything.TestEverythingConfig"
        ),
        "AutoModelForCausalLM": (
            "veomni.models.transformers.test_everything.modeling_test_everything.TestEverythingForCausalLM"
        ),
    },
    # Per-token binary classifier (Qwen2 backbone with last decoder block dropped + 1-D score head).
    # Used by the embedding-mode denoise pipeline (no vLLM); we still register the auto_map so
    # ``AutoConfig.from_pretrained`` works on the merged HF export.
    "qwen2_embedding": {
        "AutoConfig": (
            "veomni.models.transformers.qwen2_embedding.configuration_qwen2_embedding.Qwen2EmbeddingConfig"
        ),
        "AutoModelForTokenClassification": (
            "veomni.models.transformers.qwen2_embedding.modeling_qwen2_embedding.Qwen2EmbeddingForTokenClassification"
        ),
    },
}


def _ensure_veomni_on_path() -> None:
    p = str(_VEOMNI_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def _find_veomni_model_assets_dir(step_dir: Path) -> Path | None:
    """Locate ``config.json`` written by ``finetuning.py`` (``.finetune_inline_config``) or ``model_assets``."""
    candidates: list[Path] = []
    if step_dir.parent.name == "checkpoints":
        run_root = step_dir.parent.parent
        candidates.extend(
            [
                run_root / ".finetune_inline_config",
                run_root / "model_assets",
            ]
        )
    for ancestor in (step_dir.parent, step_dir.parent.parent, step_dir.parent.parent.parent):
        candidates.extend(
            [
                ancestor / ".finetune_inline_config",
                ancestor / "model_assets",
            ]
        )
    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if (c / "config.json").is_file():
            return c
    return None


def _is_veomni_dcp_step_dir(path: Path) -> bool:
    """Heuristic: VeOmni saves DCP under ``.../global_step_<n>/`` without top-level ``config.json``."""
    if not path.is_dir() or not path.name.startswith("global_step_"):
        return False
    return not (path / "config.json").is_file()


def _maybe_patch_veomni_auto_map(hf_ckpt_dir: Path) -> None:
    cfg_path = hf_ckpt_dir / "config.json"
    if not cfg_path.is_file():
        return
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    mt = data.get("model_type")
    if mt not in _VEOMNI_AUTO_MAP_BY_MODEL_TYPE or data.get("auto_map"):
        return
    data["auto_map"] = dict(_VEOMNI_AUTO_MAP_BY_MODEL_TYPE[mt])
    cfg_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"[init] Patched HuggingFace config auto_map for model_type={mt!r}", flush=True)


def _hf_export_looks_complete(hf_dir: Path) -> bool:
    """Whether ``hf_dir`` has ``config.json`` and at least one weight artifact vLLM can load."""
    if not (hf_dir / "config.json").is_file():
        return False
    if (hf_dir / "model.safetensors").is_file():
        return True
    if (hf_dir / "pytorch_model.bin").is_file():
        return True
    if (hf_dir / "model.safetensors.index.json").is_file():
        return True
    return any(hf_dir.glob("model-*.safetensors"))


def _maybe_remove_stale_hf_ckpt(hf_out: Path) -> None:
    """Drop ``hf_ckpt`` left empty or half-merged so DCP→HF can run again."""
    if not hf_out.is_dir():
        return
    if _hf_export_looks_complete(hf_out):
        return
    if not any(hf_out.iterdir()):
        return
    print(
        f"[init] Removing incomplete HuggingFace export at {hf_out} "
        "(missing config or weights); re-merging from DCP.",
        flush=True,
    )
    shutil.rmtree(hf_out)


def _probe_torch_has_distributed_checkpoint_load(py: Path) -> bool:
    """VeOmni DCP merge needs ``torch.distributed.checkpoint.load`` (PyTorch>=~2.4)."""
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        r = subprocess.run(
            [str(py), "-c", "from torch.distributed.checkpoint import load"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
            env=env,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _infer_conda_env_python(env_name: str) -> Path | None:
    """If ``sys.executable`` is ``.../envs/<name>/bin/python``, try sibling ``.../envs/<env_name>/bin/python``."""
    exe = Path(sys.executable).resolve()
    cur_env_dir = exe.parent.parent
    if cur_env_dir.parent.name != "envs":
        return None
    cand_bin = cur_env_dir.parent / env_name / "bin"
    if not cand_bin.is_dir():
        return None
    for cand in (cand_bin / exe.name, cand_bin / "python"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def _python_major_minor(py: Path) -> tuple[int, int] | None:
    try:
        r = subprocess.run(
            [str(py), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0:
            return None
        parts = r.stdout.strip().split()
        return int(parts[0]), int(parts[1])
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


def _conda_base_python_candidates() -> list[Path]:
    """``conda install python=3.13`` leaves ``<base>/bin/python3.13`` — often matches Py3.13+ DCP pickles."""
    try:
        r = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=dict(os.environ),
        )
        if r.returncode != 0:
            return []
        base = Path(r.stdout.strip())
        out: list[Path] = []
        for name in ("python3.13", "python3.12", "python"):
            p = base / "bin" / name
            if p.is_file() and os.access(p, os.X_OK):
                out.append(p)
        return out
    except OSError:
        return []


def _ensure_hf_ckpt_has_model_assets(hf_out: Path, assets_dir: Path) -> None:
    """If merge did not write ``config.json``, copy tokenizer/config from VeOmni ``model_assets``."""
    if (hf_out / "config.json").is_file():
        return
    if not (assets_dir / "config.json").is_file():
        return
    print(
        f"[init] Copying tokenizer/config from {assets_dir} into {hf_out} (merge did not embed assets).",
        flush=True,
    )
    for p in assets_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.name.lower()
        if stem.startswith("model") and stem.endswith((".safetensors", ".bin", ".pt")):
            continue
        shutil.copy2(p, hf_out / p.name)


def _iter_conda_sibling_merge_candidates(same_minor_as_current: bool) -> list[Path]:
    """Other conda env Pythones (same install prefix). Prefer matching Python minor with training/DCP pickles."""
    exe = Path(sys.executable).resolve()
    cur_env_dir = exe.parent.parent
    if cur_env_dir.parent.name != "envs":
        return []
    envs_root = cur_env_dir.parent
    want_mm = (sys.version_info.major, sys.version_info.minor)
    priority = ("vllm", "veomni", "prox-llada-hf")
    out: list[tuple[int, Path]] = []
    seen: set[str] = set()
    for rank, name in enumerate(priority):
        cand = _infer_conda_env_python(name)
        if cand is None:
            continue
        key = str(cand.resolve())
        if key in seen:
            continue
        seen.add(key)
        mm = _python_major_minor(cand)
        if mm is None:
            continue
        if same_minor_as_current and mm != want_mm:
            continue
        if not same_minor_as_current and mm == want_mm:
            continue
        out.append((rank, cand))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def _dcp_merge_python_exe() -> Path:
    raw = os.environ.get("PROX_DCP_MERGE_PYTHON", "").strip()
    if raw:
        p = Path(os.path.expanduser(raw))
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise SystemExit(f"PROX_DCP_MERGE_PYTHON is not an executable file: {p}")
        return p

    cur = Path(sys.executable)
    if _probe_torch_has_distributed_checkpoint_load(cur):
        return cur

    # Prefer sibling envs with the same Python minor as the active interpreter (pickle compat).
    for inferred in _iter_conda_sibling_merge_candidates(same_minor_as_current=True):
        if _probe_torch_has_distributed_checkpoint_load(inferred):
            print(
                f"[init] Using sibling conda Python for DCP merge (matching Python minor): {inferred}\n"
                "      (override with PROX_DCP_MERGE_PYTHON if needed)",
                flush=True,
            )
            return inferred

    # Conda base ``python3.13`` often unpickles DCP metadata saved under Python 3.13+ (pathlib layout).
    for py in _conda_base_python_candidates():
        if _probe_torch_has_distributed_checkpoint_load(py):
            print(
                f"[init] Using conda base interpreter for DCP merge: {py}\n"
                "      (needed when checkpoint metadata requires Python≥3.13; install torch in base or set "
                "PROX_DCP_MERGE_PYTHON)",
                flush=True,
            )
            return py

    for inferred in _iter_conda_sibling_merge_candidates(same_minor_as_current=False):
        if _probe_torch_has_distributed_checkpoint_load(inferred):
            print(
                f"[init] Using sibling conda Python for DCP merge (fallback): {inferred}\n"
                "      (override with PROX_DCP_MERGE_PYTHON if needed)",
                flush=True,
            )
            return inferred

    raise SystemExit(
        "Cannot run VeOmni DCP→HF merge: no Python with ``torch.distributed.checkpoint.load`` "
        "(PyTorch≥~2.4). Examples:\n"
        "  export PROX_DCP_MERGE_PYTHON=$(conda info --base)/bin/python3.13\n"
        "  export PROX_DCP_MERGE_PYTHON=$HOME/miniconda3/envs/veomni/bin/python\n"
        "If merge fails while unpickling metadata, use Python 3.13+ (``conda install python=3.13`` "
        "in ``conda info --base``)."
    )


def _merge_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    inject = f"{_REPO_ROOT}{os.pathsep}{_VEOMNI_SRC}"
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = inject if not prev else f"{inject}{os.pathsep}{prev}"
    # Avoid ~/.local shadowing conda (same issue as denoise_dataset startup).
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run_veomni_dcp_to_hf_merge(
    *,
    dcp_dir: Path,
    hf_out: Path,
    model_assets_dir: Path,
    shard_size: int,
) -> None:
    script = _VEOMNI_SRC / "scripts" / "merge_dcp_to_hf.py"
    if not script.is_file():
        raise SystemExit(f"Missing DCP merge script (expected at {script}).")
    exe = _dcp_merge_python_exe()
    cmd: list[str] = [
        str(exe),
        str(script),
        "--load-dir",
        str(dcp_dir),
        "--save-dir",
        str(hf_out),
        "--model-assets-dir",
        str(model_assets_dir),
        "--shard-size",
        str(shard_size),
    ]
    print(
        f"[init] DCP merge subprocess ({exe}); override with PROX_DCP_MERGE_PYTHON if needed.",
        flush=True,
    )
    try:
        subprocess.run(cmd, check=True, env=_merge_subprocess_env())
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            "VeOmni DCP→HF merge failed.\n"
            "- If you saw Transformers/VeOmni import errors: use a Python env with PyTorch≥2.4 and "
            "Transformers compatible with this repo's VeOmni (often ~4.52+), e.g. your VeOmni "
            "training conda env:\n"
            "    export PROX_DCP_MERGE_PYTHON=/path/to/miniconda3/envs/veomni/bin/python\n"
            "- Your vLLM-only env may keep an older PyTorch (e.g. 2.1); merge runs in the "
            "subprocess interpreter above, not the vLLM stack.\n"
            f"Exit code {e.returncode}"
        ) from e


def _resolve_model_checkpoint_for_vllm(model_checkpoint_path: str) -> str:
    """Return a HuggingFace-format directory suitable for ``vllm.LLM(model=...)``.

    Accepts either a plain HF export or ``.../checkpoints/global_step_<N>`` VeOmni DCP (merged lazily).
    """
    mp = Path(os.path.expanduser(model_checkpoint_path)).resolve()
    if (mp / "config.json").is_file():
        _maybe_patch_veomni_auto_map(mp)
        return str(mp)
    hf_sub = mp / "hf_ckpt"
    _maybe_remove_stale_hf_ckpt(hf_sub)
    if hf_sub.is_dir() and _hf_export_looks_complete(hf_sub):
        _maybe_patch_veomni_auto_map(hf_sub)
        return str(hf_sub)

    if _is_veomni_dcp_step_dir(mp):
        assets = _find_veomni_model_assets_dir(mp)
        if assets is None:
            raise SystemExit(
                f"VeOmni DCP at {mp} needs model config from the same finetune run "
                f"(e.g. {mp.parent.parent / '.finetune_inline_config' / 'config.json'}). "
                "Train with finetuning/finetune.py so that directory exists, or pass a HuggingFace export path."
            )
        hf_out = mp / "hf_ckpt"
        _maybe_remove_stale_hf_ckpt(hf_out)
        if not (hf_out / "config.json").is_file() or not _hf_export_looks_complete(hf_out):
            print(
                f"[init] Converting VeOmni DCP -> HuggingFace under {hf_out} (one-time; may take a while) …",
                flush=True,
            )
            hf_out.mkdir(parents=True, exist_ok=True)
            _run_veomni_dcp_to_hf_merge(
                dcp_dir=mp,
                hf_out=hf_out,
                model_assets_dir=assets,
                shard_size=2_000_000_000,
            )
            _ensure_hf_ckpt_has_model_assets(hf_out, assets)
            print(f"[init] DCP merge finished: {hf_out}", flush=True)
        _maybe_patch_veomni_auto_map(hf_out)
        return str(hf_out)

    return str(mp)


def _fmt_eta_sec(sec: float) -> str:
    if not math.isfinite(sec) or sec <= 0:
        return "~0s"
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.2f}h"


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SystemExit("YAML root must be a mapping (dict).")
    return raw


def output_effect_fingerprint(raw: dict[str, Any], *, shard_paths: list[Path]) -> str:
    """Stable digest including checkpoint path (full run identity)."""
    ckpt = Path(os.path.expanduser(str(raw["model_checkpoint_path"]))).resolve()
    tok_p = raw.get("tokenizer_path")
    if tok_p in (None, ""):
        tok_s = str(ckpt)
    else:
        tok_s = str(Path(os.path.expanduser(str(tok_p))).resolve())
    dp = Path(os.path.expanduser(str(raw["input_parquet_dir"]))).resolve()
    blob: dict[str, Any] = {
        "model_checkpoint_path": str(ckpt),
        "tokenizer_path": tok_s,
        "input_parquet_dir": str(dp),
        "dataset_type": str(raw.get("dataset_type", "fineweb")),
        "chunk_size": int(raw["chunk_size"]),
        "model_arch": str(raw.get("model_arch", "auto")),
        "is_code": raw.get("is_code"),
        "max_new_tokens": int(raw.get("max_new_tokens", 4096)),
        "is_embedding_model": _boolish(raw.get("is_embedding_model", False)),
        "keep_logit_threshold": float(raw.get("keep_logit_threshold", 0.0)),
        "input_shards": [str(p.resolve()) for p in shard_paths],
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def output_resume_fingerprint(raw: dict[str, Any], *, shard_paths: list[Path]) -> str:
    """Digest for ``continue: true``: same as full fingerprint but omits ``model_checkpoint_path``.

    Default tokenizer (unset ``tokenizer_path``) is represented as a sentinel so upgrading to a
    new training step does not break resume; an explicit ``tokenizer_path`` still must match.
    """
    tok_p = raw.get("tokenizer_path")
    if tok_p in (None, ""):
        tok_s = "__from_checkpoint__"
    else:
        tok_s = str(Path(os.path.expanduser(str(tok_p))).resolve())
    dp = Path(os.path.expanduser(str(raw["input_parquet_dir"]))).resolve()
    blob: dict[str, Any] = {
        "tokenizer_path": tok_s,
        "input_parquet_dir": str(dp),
        "dataset_type": str(raw.get("dataset_type", "fineweb")),
        "chunk_size": int(raw["chunk_size"]),
        "model_arch": str(raw.get("model_arch", "auto")),
        "is_code": raw.get("is_code"),
        "max_new_tokens": int(raw.get("max_new_tokens", 4096)),
        "is_embedding_model": _boolish(raw.get("is_embedding_model", False)),
        "keep_logit_threshold": float(raw.get("keep_logit_threshold", 0.0)),
        "input_shards": [str(p.resolve()) for p in shard_paths],
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _primary_write_keys(dataset_type: str) -> tuple[str, ...]:
    k = (dataset_type or "fineweb").strip().lower()
    if k in ("redpajama", "redpajama-v2"):
        return ("raw_content", "text")
    return ("text",)


def _merge_denoised_into_row(
    base_row: dict[str, Any],
    *,
    dataset_type: str,
    is_code: bool,
    full_source_text: str,
    chunk_parts: list[str],
) -> dict[str, Any]:
    """Copy input row; replace primary text / code fields with merged model outputs."""
    out = dict(base_row)
    if is_code:
        programs = [p if isinstance(p, str) else "" for p in chunk_parts]
        out["programs"] = programs
        out["programs_delimited"] = PROGRAM_CHUNK_SEPARATOR.join(programs)
        if "raw_content" in out or dataset_type.lower().startswith("redpajama"):
            out["raw_content"] = full_source_text
        return out

    merged = "".join(chunk_parts)
    for key in _primary_write_keys(dataset_type):
        if key in out:
            out[key] = merged
            return out
    out["text"] = merged
    return out


@dataclass
class _ChunkJob:
    pq_name: str
    row_idx: int
    chunk_idx: int
    n_chunks: int
    doc_len: int
    prompt: str


@dataclass
class RowAccum:
    pq_name: str
    row_idx: int
    n_chunks: int
    full_text: str
    doc_len: int
    parts: list[str | None] = field(default_factory=list)
    trunc_chunks: int = 0

    def __post_init__(self) -> None:
        self.parts = [None] * self.n_chunks


def _maybe_print_metrics(
    cfg: Any,
    metrics: Metrics,
    *,
    wall0: float,
    last_wall: list[float],
    global_chars: int,
    global_chunks: int,
    log_prefix: str = "",
) -> None:
    now = time.perf_counter()
    if now - last_wall[0] < cfg.metrics_interval_sec:
        return
    last_wall[0] = now
    snap = metrics.snapshot()
    elapsed = max(now - wall0, 1e-9)
    docs_s = snap["completed_docs"] / elapsed
    toks = snap["prompt_tokens"] + snap["completion_tokens"]
    tok_s = toks / elapsed
    p_cap = metrics.total_parquets
    doc_d = metrics.doc_denom
    ch_d = metrics.char_denom
    api_d = metrics.api_tok_denom

    doc_frac = f"{snap['completed_docs']}/{doc_d}" if doc_d >= 0 else f"{snap['completed_docs']}/?"
    ch_frac = f"{global_chars}/{ch_d}" if ch_d >= 0 else f"{global_chars}/?"
    pq_frac = f"{snap['completed_parquets']}/{p_cap}"
    api_frac = f"{toks}/{api_d}" if api_d >= 0 else f"{toks}/?"
    chars_s = global_chars / elapsed

    ch_lim = getattr(metrics, "chunk_denom", -1)
    ch_run_frac = (
        f"{global_chunks}/{ch_lim}" if ch_lim >= 0 else f"{global_chunks}/?"
    )

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
    if ch_lim >= 0:
        rem_ch = ch_lim - global_chunks
        ch_rate = global_chunks / elapsed
        if rem_ch > 0 and ch_rate > 1e-12:
            eta_candidates.append(rem_ch / ch_rate)

    if eta_candidates:
        eta_s = f" eta_rem≈{_fmt_eta_sec(min(eta_candidates))} (min of caps)"
    else:
        eta_s = " eta_rem=?"

    print(
        f"{log_prefix}[metrics] wall={elapsed:.1f}s docs/s={docs_s:.4f} toks/s={tok_s:.2f} "
        f"chunks={snap['completed_chunks']} "
        f"docs={doc_frac} chars={ch_frac} chunks_cap={ch_run_frac} "
        f"parquets={pq_frac} api_toks={api_frac} "
        f"prompt_tok={snap['prompt_tokens']} completion_tok={snap['completion_tokens']}"
        f"{eta_s}",
        flush=True,
    )


def _parse_cfg(raw: dict[str, Any], cfg_path: Path) -> Any:
    from types import SimpleNamespace

    try:
        ns = SimpleNamespace(
            config_path=cfg_path,
            input_parquet_dir=Path(os.path.expanduser(str(raw["input_parquet_dir"]))),
            dataset_type=str(raw.get("dataset_type", "fineweb")),
            output_data_dir=Path(os.path.expanduser(str(raw["output_data_dir"]))),
            overwrite=_boolish(raw.get("overwrite", False)),
            continue_run=_boolish(raw.get("continue", False)),
            chunk_size=int(raw["chunk_size"]),
            model_arch=str(raw.get("model_arch", "auto")),
            model_checkpoint_path=str(raw["model_checkpoint_path"]),
            tokenizer_path=(
                str(raw["tokenizer_path"])
                if raw.get("tokenizer_path") not in (None, "")
                else None
            ),
            num_gpus=max(1, int(raw.get("num_gpus", 1))),
            data_parallel_replicas=max(1, int(raw.get("data_parallel_replicas", 1))),
            data_parallel_gpu_ids=(
                [int(x) for x in raw["data_parallel_gpu_ids"]]
                if isinstance(raw.get("data_parallel_gpu_ids"), list)
                else None
            ),
            max_docs=int(raw.get("max_docs", -1)),
            max_chars=int(raw.get("max_chars", -1)),
            max_chunks=int(raw.get("max_chunks", -1)),
            max_tokens=int(raw.get("max_tokens", -1)),
            max_parquets=int(raw.get("max_parquets", -1)),
            is_code=_boolish(raw.get("is_code", False)),
            chunk_batch_size=max(1, int(raw.get("chunk_batch_size", 64))),
            max_new_tokens=int(raw.get("max_new_tokens", 4096)),
            max_model_len=int(raw.get("max_model_len", 16384)),
            trust_remote_code=_boolish(raw.get("trust_remote_code", True)),
            metrics_interval_sec=float(raw.get("metrics_interval_sec", 15.0)),
            vllm_extras=dict(raw["vllm"]) if isinstance(raw.get("vllm"), dict) else {},
            # Embedding (per-token binary keep/discard) pipeline flags. When ``is_embedding_model``
            # is True, the script bypasses vLLM and loads the finetuned token-classification model
            # with ``transformers`` + PyTorch (HuggingFace ``Qwen2EmbeddingForTokenClassification``).
            # Each chunk is tokenized with ``return_offsets_mapping=True``; tokens whose score-logit
            # is at most ``keep_logit_threshold`` are dropped from the reconstructed chunk text.
            is_embedding_model=_boolish(raw.get("is_embedding_model", False)),
            keep_logit_threshold=float(raw.get("keep_logit_threshold", 0.0)),
        )
    except KeyError as e:
        raise SystemExit(f"Missing required config key: {e}") from e
    return ns


def _dp_gpu_ids_or_exit(cfg: Any) -> list[int]:
    r = int(cfg.data_parallel_replicas)
    raw_ids = cfg.data_parallel_gpu_ids
    if raw_ids is None:
        return list(range(r))
    if len(raw_ids) != r:
        raise SystemExit(
            f"data_parallel_gpu_ids must have length data_parallel_replicas ({r}), "
            f"got {len(raw_ids)} id(s)."
        )
    return list(raw_ids)


def _dp_worker_main(
    cfg_path_str: str,
    shard_paths_strs: list[str],
    gpu_id: int,
    replica_idx: int,
    n_replicas: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg_path = Path(cfg_path_str).expanduser().resolve()
    raw = load_yaml_config(cfg_path)
    cfg = _parse_cfg(raw, cfg_path)
    shard_paths = [Path(s) for s in shard_paths_strs]
    log_prefix = f"[dp {replica_idx + 1}/{n_replicas}] "
    print(
        f"{log_prefix}visible GPU map: physical cuda:{gpu_id} -> worker cuda:0 "
        f"shards={len(shard_paths)} mode={'embedding' if cfg.is_embedding_model else 'denoise(vllm)'}",
        flush=True,
    )
    if not shard_paths:
        return
    _ensure_veomni_on_path()
    resolved_ckpt = _resolve_model_checkpoint_for_vllm(cfg.model_checkpoint_path)
    if cfg.is_embedding_model:
        _execute_embedding_pipeline(
            cfg,
            shard_paths,
            resolved_ckpt,
            resume_ok=False,
            log_prefix=log_prefix,
        )
    else:
        _execute_denoise_pipeline(
            cfg,
            shard_paths,
            resolved_ckpt,
            resume_ok=False,
            log_prefix=log_prefix,
        )


def _run_data_parallel(
    *,
    cfg_path: Path,
    cfg: Any,
    shard_paths: list[Path],
    gpu_ids: list[int],
) -> None:
    _ensure_veomni_on_path()
    resolved_ckpt = _resolve_model_checkpoint_for_vllm(cfg.model_checkpoint_path)
    print(
        f"[init] data_parallel_replicas={cfg.data_parallel_replicas} "
        f"tensor_parallel_per_replica={cfg.num_gpus} "
        f"mode={'embedding' if cfg.is_embedding_model else 'denoise(vllm)'}",
        flush=True,
    )
    if cfg.is_embedding_model:
        print(
            f"[init] HuggingFace model path (resolved, pre-workers, embedding mode): {resolved_ckpt}",
            flush=True,
        )
    else:
        print(f"[init] vLLM model path (resolved, pre-workers): {resolved_ckpt}", flush=True)

    R = int(cfg.data_parallel_replicas)
    ctx = mp.get_context("spawn")
    jobs: list[tuple[int, mp.Process]] = []
    cfg_abs = str(cfg_path.expanduser().resolve())
    for replica_idx in range(R):
        subset = shard_paths[replica_idx::R]
        if not subset:
            continue
        gpu_id = gpu_ids[replica_idx]
        proc = ctx.Process(
            target=_dp_worker_main,
            args=(
                cfg_abs,
                [str(p.resolve()) for p in subset],
                gpu_id,
                replica_idx,
                R,
            ),
        )
        proc.start()
        jobs.append((replica_idx, proc))

    worst = 0
    for replica_idx, proc in jobs:
        proc.join()
        if proc.exitcode != 0:
            print(
                f"[error] data_parallel worker idx={replica_idx} "
                f"exit_code={proc.exitcode}",
                flush=True,
            )
            worst = proc.exitcode if proc.exitcode else 1
    if worst != 0:
        raise SystemExit(worst)
    print("[done] data_parallel: all workers finished.", flush=True)


def _execute_denoise_pipeline(
    cfg: Any,
    shard_paths: list[Path],
    resolved_ckpt: str,
    *,
    resume_ok: bool,
    log_prefix: str = "",
) -> None:
    row_text_fn: Callable[[dict[str, Any]], str] = get_row_text_fn(cfg.dataset_type)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok_path = cfg.tokenizer_path or resolved_ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_prompt(chunk_body: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": chunk_body},
        ]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"[SYSTEM]\n{SYSTEM_MSG}\n[USER]\n{chunk_body}\n[ASSISTANT]\n"

    llm_kw: dict[str, Any] = {
        "model": resolved_ckpt,
        "tokenizer": tok_path,
        "tensor_parallel_size": cfg.num_gpus,
        "trust_remote_code": cfg.trust_remote_code,
        "max_model_len": cfg.max_model_len,
        "dtype": "bfloat16",
        "enforce_eager": False,
    }
    llm_kw.update(cfg.vllm_extras)
    llm = LLM(**llm_kw)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=cfg.max_new_tokens,
    )

    wall0 = time.perf_counter()
    last_wall = [wall0]

    global_docs = 0
    global_chars = 0
    global_chunks = 0
    inflight_docs = 0
    run_lock = threading.Lock()

    shard_writers: dict[str, dict[str, Any]] = {}
    input_row_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    pending_jobs: list[_ChunkJob] = []
    pending_trackers: dict[tuple[str, int], RowAccum] = {}

    metrics = Metrics()
    metrics.total_parquets = len(shard_paths)
    try:
        row_total = sum(precount_doc_rows([p]) for p in shard_paths)
    except Exception as e:
        print(f"{log_prefix}[init] row precount skipped: {e}", flush=True)
        row_total = -1
    try:
        parquet_chars, parquet_tok_est = precount_chars_and_token_est(
            shard_paths, row_text_fn
        )
    except Exception as e:
        print(f"{log_prefix}[init] char/token precount skipped: {e}", flush=True)
        parquet_chars, parquet_tok_est = -1, -1
    metrics.doc_denom = _doc_denom(cfg.max_docs, row_total)
    metrics.char_denom = _char_denom(cfg.max_chars, parquet_chars)
    metrics.api_tok_denom = _api_tok_denom(cfg.max_tokens, parquet_tok_est)
    chunk_denom = cfg.max_chunks if cfg.max_chunks >= 0 else -1
    setattr(metrics, "chunk_denom", chunk_denom)

    def caps_hit() -> bool:
        if cfg.max_docs >= 0 and global_docs + inflight_docs >= cfg.max_docs:
            return True
        if cfg.max_chars >= 0 and global_chars >= cfg.max_chars:
            return True
        if cfg.max_chunks >= 0 and global_chunks >= cfg.max_chunks:
            return True
        return False

    def _drain_shard_row_buffer(wb: dict[str, Any]) -> None:
        """Move completed rows from ``_pending_by_row`` to ``rows`` in ascending ``row_idx`` order."""
        pend: dict[int, Any] = wb["_pending_by_row"]
        nxt: int = wb["_next_emit_row"]
        rows_list: list[Any] = wb["rows"]
        while nxt in pend:
            rows_list.append(pend.pop(nxt))
            nxt += 1
        wb["_next_emit_row"] = nxt

    def process_chunk_batch(
        jobs: list[_ChunkJob],
        trackers: dict[tuple[str, int], RowAccum],
        metrics: Metrics,
    ) -> None:
        nonlocal global_chunks, global_docs, global_chars, inflight_docs
        if not jobs:
            return
        prompts = [j.prompt for j in jobs]
        outputs = llm.generate(prompts, sampling)
        done_keys: set[tuple[str, int]] = set()
        for job, out in zip(jobs, outputs):
            text = ""
            if out.outputs:
                text = out.outputs[0].text or ""
            pt = len(out.prompt_token_ids) if out.prompt_token_ids else 0
            ct = 0
            if out.outputs and getattr(out.outputs[0], "token_ids", None):
                ct = len(out.outputs[0].token_ids)
            finish_reason = (
                str(out.outputs[0].finish_reason or "") if out.outputs else ""
            )
            metrics.add_usage(pt, ct)

            fr_trunc = finish_reason.lower() == "length"
            acc = trackers[(job.pq_name, job.row_idx)]
            if acc.parts[job.chunk_idx] is not None:
                raise RuntimeError("duplicate chunk slot")
            acc.parts[job.chunk_idx] = text if isinstance(text, str) else ""
            if fr_trunc:
                acc.trunc_chunks += 1

            with run_lock:
                global_chunks += 1

            key = (job.pq_name, job.row_idx)
            if all(p is not None for p in trackers[key].parts):
                done_keys.add(key)

        for key in done_keys:
            acc = trackers.pop(key)
            parts = [p if isinstance(p, str) else "" for p in acc.parts]
            row_out = input_row_by_key.pop(key, None)
            if row_out is None:
                continue
            if acc.trunc_chunks:
                print(
                    f"{log_prefix}[warn] max_tokens: {acc.trunc_chunks}/{acc.n_chunks} chunk(s) "
                    f"length-stopped source_shard={acc.pq_name} row={acc.row_idx}",
                    flush=True,
                )
            merged = _merge_denoised_into_row(
                row_out,
                dataset_type=cfg.dataset_type,
                is_code=cfg.is_code,
                full_source_text=acc.full_text,
                chunk_parts=parts,
            )
            writer_bundle = shard_writers[key[0]]
            writer_bundle["_pending_by_row"][acc.row_idx] = merged
            _drain_shard_row_buffer(writer_bundle)
            metrics.add_doc()
            with run_lock:
                inflight_docs -= 1
                global_docs += 1
                global_chars += acc.doc_len
            _maybe_print_metrics(
                cfg,
                metrics,
                wall0=wall0,
                last_wall=last_wall,
                global_chars=global_chars,
                global_chunks=global_chunks,
                log_prefix=log_prefix,
            )

    for pq_path in shard_paths:
        out_name = f"{pq_path.stem}_denoised.parquet"
        out_path = cfg.output_data_dir / out_name

        if out_path.exists() and not cfg.overwrite and not resume_ok:
            raise SystemExit(f"Output exists: {out_path}")

        n_existing = 0
        if resume_ok and out_path.is_file():
            n_existing = pq.ParquetFile(out_path).metadata.num_rows

        writers_state: dict[str, Any] = {
            "rows": [],
            "_pending_by_row": {},
            "_next_emit_row": n_existing,
        }
        shard_writers[pq_path.name] = writers_state

        row_iter = enumerate(iter_shard_rows(pq_path))

        def flush_batch() -> None:
            if not pending_jobs:
                return
            process_chunk_batch(pending_jobs, pending_trackers, metrics)
            pending_jobs.clear()

        for row_idx, row in row_iter:
            if row_idx < n_existing:
                continue
            if caps_hit():
                break

            text = row_text_fn(row)
            doc_len = len(text)
            with run_lock:
                if cfg.max_docs >= 0 and global_docs + inflight_docs >= cfg.max_docs:
                    break
                if cfg.max_chars >= 0 and global_chars + doc_len > cfg.max_chars:
                    break
                if cfg.max_chunks >= 0 and global_chunks >= cfg.max_chunks:
                    break

            chunks = chunk_text(text, cfg.chunk_size)
            if cfg.max_chunks >= 0:
                budget = cfg.max_chunks - global_chunks
                if budget <= 0:
                    break
                if len(chunks) > budget:
                    chunks = chunks[:budget]

            if not chunks:
                merged = _merge_denoised_into_row(
                    row,
                    dataset_type=cfg.dataset_type,
                    is_code=cfg.is_code,
                    full_source_text=text,
                    chunk_parts=[],
                )
                writers_state["_pending_by_row"][row_idx] = merged
                _drain_shard_row_buffer(writers_state)
                metrics.add_doc()
                global_docs += 1
                global_chars += doc_len
                _maybe_print_metrics(
                    cfg,
                    metrics,
                    wall0=wall0,
                    last_wall=last_wall,
                    global_chars=global_chars,
                    global_chunks=global_chunks,
                    log_prefix=log_prefix,
                )
                continue

            key = (pq_path.name, row_idx)
            tr = RowAccum(
                pq_name=pq_path.name,
                row_idx=row_idx,
                n_chunks=len(chunks),
                full_text=text,
                doc_len=doc_len,
                parts=[None] * len(chunks),
            )
            pending_trackers[key] = tr
            input_row_by_key[key] = row
            with run_lock:
                inflight_docs += 1

            for cidx, ch in enumerate(chunks):
                pending_jobs.append(
                    _ChunkJob(
                        pq_name=pq_path.name,
                        row_idx=row_idx,
                        chunk_idx=cidx,
                        n_chunks=len(chunks),
                        doc_len=doc_len,
                        prompt=format_prompt(ch),
                    )
                )
                if len(pending_jobs) >= cfg.chunk_batch_size:
                    flush_batch()

            if caps_hit():
                flush_batch()
                break

        flush_batch()
        _drain_shard_row_buffer(writers_state)

        # Write shard parquet
        st = shard_writers.pop(pq_path.name, {"rows": [], "_pending_by_row": {}, "_next_emit_row": 0})
        rows_out = st["rows"]
        pend_left = st.get("_pending_by_row") or {}
        if pend_left:
            nxt = int(st.get("_next_emit_row", 0))
            raise RuntimeError(
                f"Internal alignment error for {pq_path.name}: {len(pend_left)} row(s) merged but not "
                f"emitted (next_emit_row={nxt}, pending_keys={sorted(pend_left.keys())}). "
                "Likely incomplete chunk work at shard end."
            )
        if resume_ok and n_existing and rows_out:
            old = pq.read_table(out_path).to_pylist()
            rows_out = old + rows_out
        if rows_out:
            table = pa.Table.from_pylist(rows_out)
            pq.write_table(table, out_path)

        metrics.add_parquet_done()
        _maybe_print_metrics(
            cfg,
            metrics,
            wall0=wall0,
            last_wall=last_wall,
            global_chars=global_chars,
            global_chunks=global_chunks,
            log_prefix=log_prefix,
        )

        if caps_hit():
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
    ch_lim = getattr(metrics, "chunk_denom", -1)
    ch_run = f"{global_chunks}/{ch_lim}" if ch_lim >= 0 else f"{global_chunks}/?"
    print(
        f"{log_prefix}[done] wall={elapsed:.2f}s docs={doc_part} chars={ch_part} chunks={ch_run} "
        f"api_toks={api_part} "
        f"parquets={snap['completed_parquets']} "
        f"prompt_tok={snap['prompt_tokens']} completion_tok={snap['completion_tokens']} "
        f"out_dir={cfg.output_data_dir}",
        flush=True,
    )


def _execute_embedding_pipeline(
    cfg: Any,
    shard_paths: list[Path],
    resolved_ckpt: str,
    *,
    resume_ok: bool,
    log_prefix: str = "",
) -> None:
    """Per-token binary keep/discard pipeline (no vLLM).

    Loads ``Qwen2EmbeddingForTokenClassification`` (the ``qwen2_embedding`` model from
    ``finetuning/finetune.py`` with ``is_embedding_model: true``) using HuggingFace transformers
    + PyTorch and produces denoised text by **dropping characters** belonging to tokens whose
    binary score-logit falls at or below ``cfg.keep_logit_threshold`` (default ``0.0``, i.e.
    ``sigmoid(logit) > 0.5``).

    For each chunk of size ``cfg.chunk_size`` characters:
      1. Tokenize with ``add_special_tokens=False`` and ``return_offsets_mapping=True`` (matches
         the ``text_per_token_binary`` data transform used at training time).
      2. Run the model's forward pass (no causal LM, no generation): the score head emits a
         scalar logit per token.
      3. Reconstruct the chunk by concatenating ``chunk[s:e]`` for every kept token whose offset
         span ``(s, e)`` is non-empty.

    The output schema and shard partitioning are identical to ``_execute_denoise_pipeline`` so
    downstream code (parquet readers, ``data_parallel_replicas`` round-robin sharding, resume,
    etc.) does not change.
    """
    row_text_fn: Callable[[dict[str, Any]], str] = get_row_text_fn(cfg.dataset_type)

    import torch
    from transformers import AutoTokenizer

    # transformers>=4.51 (and even some 4.50 dev tags) call ``torch.get_default_device()`` inside
    # ``PreTrainedModel.from_pretrained`` -> ``get_torch_context_manager_or_global_device``.
    # That helper was added in PyTorch 2.3; the ``refining`` conda env (built for vLLM) commonly
    # pins an older torch, so the call raises ``AttributeError`` before any weights are loaded.
    # Polyfill matches the upstream default (no ``torch.set_default_device`` call -> CPU).
    if not hasattr(torch, "get_default_device"):
        def _polyfill_get_default_device() -> "torch.device":
            return torch.tensor([]).device
        torch.get_default_device = _polyfill_get_default_device  # type: ignore[attr-defined]

    _ensure_veomni_on_path()
    from veomni.models.transformers.qwen2_embedding.configuration_qwen2_embedding import (
        Qwen2EmbeddingConfig,
    )
    from veomni.models.transformers import qwen2_embedding as _qwen2emb_pkg  # noqa: F401  (path init)
    from veomni.models.transformers.qwen2_embedding import (
        modeling_qwen2_embedding as _qwen2emb_mod,
    )
    from veomni.models.transformers.qwen2_embedding.modeling_qwen2_embedding import (
        Qwen2EmbeddingForTokenClassification,
    )

    # The qwen2 backbone uses ``_import_qwen2_model`` which (on transformers < 5) imports
    # ``apply_veomni_qwen2_patch``. That patch module imports ``TransformersKwargs`` from
    # ``transformers.utils`` — symbol added in transformers ~4.55, missing in the ``refining``
    # conda env (4.52). The patch is for FlashAttention/sequence-parallel niceties; pure
    # inference is correct with the stock HuggingFace ``Qwen2Model``. Fall back when the
    # patch import is unavailable so the embedding pipeline runs in older environments too.
    try:
        _qwen2emb_mod._import_qwen2_model()
    except (ImportError, AttributeError) as _e:
        from transformers import Qwen2Model as _HF_Qwen2Model

        _qwen2emb_mod._import_qwen2_model = lambda: _HF_Qwen2Model
        print(
            f"{log_prefix}[init] VeOmni qwen2 patch unavailable in this env "
            f"({type(_e).__name__}: {_e}); falling back to stock transformers ``Qwen2Model`` "
            "for inference (FlashAttention/SP patches not required for the per-token classifier).",
            flush=True,
        )

    tok_path = cfg.tokenizer_path or resolved_ckpt
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=cfg.trust_remote_code, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit(
            "Embedding pipeline requires a fast tokenizer (offset_mapping). "
            f"Tokenizer at {tok_path!r} is slow; install ``tokenizers`` or pass a different "
            "``tokenizer_path``."
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen2EmbeddingConfig.from_pretrained(resolved_ckpt)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2EmbeddingForTokenClassification.from_pretrained(
        resolved_ckpt,
        config=config,
        torch_dtype=model_dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(
        f"{log_prefix}[init] embedding model loaded: {type(model).__name__} "
        f"hidden={config.hidden_size} layers={config.num_hidden_layers} "
        f"dtype={model_dtype} device={device} threshold={cfg.keep_logit_threshold}",
        flush=True,
    )

    threshold = float(getattr(cfg, "keep_logit_threshold", 0.0))
    max_len = int(cfg.max_model_len)

    @torch.no_grad()
    def predict_kept_chunks(chunks_text: list[str]) -> tuple[list[str], int, int]:
        """Run one forward pass over a batch of chunk strings.

        Returns ``(kept_text_per_chunk, total_input_tokens, n_truncated_chunks)``.
        """
        if not chunks_text:
            return [], 0, 0
        enc = tokenizer(
            chunks_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)
        offsets = enc["offset_mapping"]
        offsets_list = offsets.tolist() if torch.is_tensor(offsets) else offsets

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, L]
        keep_bool = (logits > threshold).cpu().tolist()
        am = attention_mask.cpu().tolist()
        n_input_tokens = int(attention_mask.sum().item())

        n_truncated = 0
        results: list[str] = []
        for i, chunk in enumerate(chunks_text):
            kept_parts: list[str] = []
            covered_end = 0
            for tok_i in range(len(am[i])):
                if not am[i][tok_i]:
                    continue
                s, e = offsets_list[i][tok_i]
                if e <= s:
                    continue
                if e > covered_end:
                    covered_end = e
                if keep_bool[i][tok_i]:
                    kept_parts.append(chunk[s:e])
            if covered_end < len(chunk):
                # Tokenizer truncation dropped the tail; the discarded tail is treated as
                # "not kept" (matches training, which truncates labels to ``max_seq_len`` too).
                n_truncated += 1
            results.append("".join(kept_parts))
        return results, n_input_tokens, n_truncated

    wall0 = time.perf_counter()
    last_wall = [wall0]

    global_docs = 0
    global_chars = 0
    global_chunks = 0
    inflight_docs = 0
    run_lock = threading.Lock()

    shard_writers: dict[str, dict[str, Any]] = {}
    input_row_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    pending_jobs: list[_ChunkJob] = []
    pending_trackers: dict[tuple[str, int], RowAccum] = {}

    metrics = Metrics()
    metrics.total_parquets = len(shard_paths)
    try:
        row_total = sum(precount_doc_rows([p]) for p in shard_paths)
    except Exception as e:
        print(f"{log_prefix}[init] row precount skipped: {e}", flush=True)
        row_total = -1
    try:
        parquet_chars, parquet_tok_est = precount_chars_and_token_est(
            shard_paths, row_text_fn
        )
    except Exception as e:
        print(f"{log_prefix}[init] char/token precount skipped: {e}", flush=True)
        parquet_chars, parquet_tok_est = -1, -1
    metrics.doc_denom = _doc_denom(cfg.max_docs, row_total)
    metrics.char_denom = _char_denom(cfg.max_chars, parquet_chars)
    metrics.api_tok_denom = _api_tok_denom(cfg.max_tokens, parquet_tok_est)
    chunk_denom = cfg.max_chunks if cfg.max_chunks >= 0 else -1
    setattr(metrics, "chunk_denom", chunk_denom)

    def caps_hit() -> bool:
        if cfg.max_docs >= 0 and global_docs + inflight_docs >= cfg.max_docs:
            return True
        if cfg.max_chars >= 0 and global_chars >= cfg.max_chars:
            return True
        if cfg.max_chunks >= 0 and global_chunks >= cfg.max_chunks:
            return True
        return False

    def _drain_shard_row_buffer(wb: dict[str, Any]) -> None:
        pend: dict[int, Any] = wb["_pending_by_row"]
        nxt: int = wb["_next_emit_row"]
        rows_list: list[Any] = wb["rows"]
        while nxt in pend:
            rows_list.append(pend.pop(nxt))
            nxt += 1
        wb["_next_emit_row"] = nxt

    def process_chunk_batch(
        jobs: list[_ChunkJob],
        trackers: dict[tuple[str, int], RowAccum],
        metrics: Metrics,
    ) -> None:
        nonlocal global_chunks, global_docs, global_chars, inflight_docs
        if not jobs:
            return
        chunks_text = [j.prompt for j in jobs]
        kept_texts, n_input_tok, n_truncated = predict_kept_chunks(chunks_text)
        # Embedding mode has no completion tokens; report only input tokens via ``add_usage``.
        metrics.add_usage(n_input_tok, 0)

        if n_truncated:
            print(
                f"{log_prefix}[warn] {n_truncated}/{len(jobs)} chunk(s) hit max_model_len="
                f"{cfg.max_model_len}; the trailing characters were tokenizer-truncated and "
                "dropped from output.",
                flush=True,
            )

        done_keys: set[tuple[str, int]] = set()
        for job, kept in zip(jobs, kept_texts):
            acc = trackers[(job.pq_name, job.row_idx)]
            if acc.parts[job.chunk_idx] is not None:
                raise RuntimeError("duplicate chunk slot")
            acc.parts[job.chunk_idx] = kept if isinstance(kept, str) else ""
            with run_lock:
                global_chunks += 1
            key = (job.pq_name, job.row_idx)
            if all(p is not None for p in trackers[key].parts):
                done_keys.add(key)

        for key in done_keys:
            acc = trackers.pop(key)
            parts = [p if isinstance(p, str) else "" for p in acc.parts]
            row_out = input_row_by_key.pop(key, None)
            if row_out is None:
                continue
            merged = _merge_denoised_into_row(
                row_out,
                dataset_type=cfg.dataset_type,
                is_code=cfg.is_code,
                full_source_text=acc.full_text,
                chunk_parts=parts,
            )
            writer_bundle = shard_writers[key[0]]
            writer_bundle["_pending_by_row"][acc.row_idx] = merged
            _drain_shard_row_buffer(writer_bundle)
            metrics.add_doc()
            with run_lock:
                inflight_docs -= 1
                global_docs += 1
                global_chars += acc.doc_len
            _maybe_print_metrics(
                cfg,
                metrics,
                wall0=wall0,
                last_wall=last_wall,
                global_chars=global_chars,
                global_chunks=global_chunks,
                log_prefix=log_prefix,
            )

    for pq_path in shard_paths:
        out_name = f"{pq_path.stem}_denoised.parquet"
        out_path = cfg.output_data_dir / out_name

        if out_path.exists() and not cfg.overwrite and not resume_ok:
            raise SystemExit(f"Output exists: {out_path}")

        n_existing = 0
        if resume_ok and out_path.is_file():
            n_existing = pq.ParquetFile(out_path).metadata.num_rows

        writers_state: dict[str, Any] = {
            "rows": [],
            "_pending_by_row": {},
            "_next_emit_row": n_existing,
        }
        shard_writers[pq_path.name] = writers_state

        row_iter = enumerate(iter_shard_rows(pq_path))

        def flush_batch() -> None:
            if not pending_jobs:
                return
            process_chunk_batch(pending_jobs, pending_trackers, metrics)
            pending_jobs.clear()

        for row_idx, row in row_iter:
            if row_idx < n_existing:
                continue
            if caps_hit():
                break

            text = row_text_fn(row)
            doc_len = len(text)
            with run_lock:
                if cfg.max_docs >= 0 and global_docs + inflight_docs >= cfg.max_docs:
                    break
                if cfg.max_chars >= 0 and global_chars + doc_len > cfg.max_chars:
                    break
                if cfg.max_chunks >= 0 and global_chunks >= cfg.max_chunks:
                    break

            chunks = chunk_text(text, cfg.chunk_size)
            if cfg.max_chunks >= 0:
                budget = cfg.max_chunks - global_chunks
                if budget <= 0:
                    break
                if len(chunks) > budget:
                    chunks = chunks[:budget]

            if not chunks:
                merged = _merge_denoised_into_row(
                    row,
                    dataset_type=cfg.dataset_type,
                    is_code=cfg.is_code,
                    full_source_text=text,
                    chunk_parts=[],
                )
                writers_state["_pending_by_row"][row_idx] = merged
                _drain_shard_row_buffer(writers_state)
                metrics.add_doc()
                global_docs += 1
                global_chars += doc_len
                _maybe_print_metrics(
                    cfg,
                    metrics,
                    wall0=wall0,
                    last_wall=last_wall,
                    global_chars=global_chars,
                    global_chunks=global_chunks,
                    log_prefix=log_prefix,
                )
                continue

            key = (pq_path.name, row_idx)
            tr = RowAccum(
                pq_name=pq_path.name,
                row_idx=row_idx,
                n_chunks=len(chunks),
                full_text=text,
                doc_len=doc_len,
                parts=[None] * len(chunks),
            )
            pending_trackers[key] = tr
            input_row_by_key[key] = row
            with run_lock:
                inflight_docs += 1

            for cidx, ch in enumerate(chunks):
                pending_jobs.append(
                    _ChunkJob(
                        pq_name=pq_path.name,
                        row_idx=row_idx,
                        chunk_idx=cidx,
                        n_chunks=len(chunks),
                        doc_len=doc_len,
                        prompt=ch,
                    )
                )
                if len(pending_jobs) >= cfg.chunk_batch_size:
                    flush_batch()

            if caps_hit():
                flush_batch()
                break

        flush_batch()
        _drain_shard_row_buffer(writers_state)

        st = shard_writers.pop(pq_path.name, {"rows": [], "_pending_by_row": {}, "_next_emit_row": 0})
        rows_out = st["rows"]
        pend_left = st.get("_pending_by_row") or {}
        if pend_left:
            nxt = int(st.get("_next_emit_row", 0))
            raise RuntimeError(
                f"Internal alignment error for {pq_path.name}: {len(pend_left)} row(s) merged but not "
                f"emitted (next_emit_row={nxt}, pending_keys={sorted(pend_left.keys())}). "
                "Likely incomplete chunk work at shard end."
            )
        if resume_ok and n_existing and rows_out:
            old = pq.read_table(out_path).to_pylist()
            rows_out = old + rows_out
        if rows_out:
            table = pa.Table.from_pylist(rows_out)
            pq.write_table(table, out_path)

        metrics.add_parquet_done()
        _maybe_print_metrics(
            cfg,
            metrics,
            wall0=wall0,
            last_wall=last_wall,
            global_chars=global_chars,
            global_chunks=global_chunks,
            log_prefix=log_prefix,
        )

        if caps_hit():
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
    ch_lim = getattr(metrics, "chunk_denom", -1)
    ch_run = f"{global_chunks}/{ch_lim}" if ch_lim >= 0 else f"{global_chunks}/?"
    print(
        f"{log_prefix}[done][embedding] wall={elapsed:.2f}s docs={doc_part} chars={ch_part} "
        f"chunks={ch_run} input_toks={api_part} parquets={snap['completed_parquets']} "
        f"out_dir={cfg.output_data_dir}",
        flush=True,
    )


def run(cfg_path: Path) -> None:
    raw = load_yaml_config(cfg_path)
    cfg = _parse_cfg(raw, cfg_path.expanduser().resolve())

    row_text_fn: Callable[[dict[str, Any]], str] = get_row_text_fn(cfg.dataset_type)

    if not cfg.input_parquet_dir.is_dir():
        raise SystemExit(f"input_parquet_dir is not a directory: {cfg.input_parquet_dir}")

    all_shards = sorted_shard_paths(cfg.input_parquet_dir)
    if not all_shards:
        raise SystemExit(f"No shards under {cfg.input_parquet_dir}")

    shard_paths = (
        all_shards[: cfg.max_parquets] if cfg.max_parquets >= 0 else all_shards
    )

    cfg_src = cfg_path.expanduser().resolve()
    cfg_dst = cfg.output_data_dir / cfg_src.name

    resume_ok = False
    if cfg.continue_run:
        if not cfg.output_data_dir.is_dir():
            print(
                "[init] continue: true but output_data_dir does not exist yet; starting fresh.",
                flush=True,
            )
        elif not cfg_dst.is_file():
            raise SystemExit(
                f"continue: true requires the prior run's copied YAML at {cfg_dst}.\n"
                "Run once without continue, then enable continue."
            )
        else:
            prev_raw = load_yaml_config(cfg_dst)
            prev_dp = Path(os.path.expanduser(str(prev_raw["input_parquet_dir"])))
            prev_paths = sorted_shard_paths(prev_dp)
            if int(prev_raw.get("max_parquets", -1)) >= 0:
                prev_paths = prev_paths[: int(prev_raw["max_parquets"])]
            fp_old_res = output_resume_fingerprint(prev_raw, shard_paths=prev_paths)
            fp_new_res = output_resume_fingerprint(raw, shard_paths=shard_paths)
            if fp_old_res != fp_new_res:
                raise SystemExit(
                    "continue: true but data/chunk/tokenizer settings differ from saved YAML in "
                    f"{cfg_dst}. Use a new output_data_dir or align settings."
                )
            prev_ckpt = Path(os.path.expanduser(str(prev_raw["model_checkpoint_path"]))).resolve()
            new_ckpt = Path(os.path.expanduser(str(raw["model_checkpoint_path"]))).resolve()
            if prev_ckpt != new_ckpt:
                print(
                    "[init] continue: model_checkpoint_path updated (resume allowed).\n"
                    f"       saved in output dir: {prev_ckpt}\n"
                    f"       current YAML:        {new_ckpt}",
                    flush=True,
                )
            resume_ok = True
            print(
                f"[init] continue: resume fingerprint matches {cfg_dst.name}; resuming.",
                flush=True,
            )

    if cfg.output_data_dir.exists():
        if not cfg.overwrite and not resume_ok:
            raise SystemExit(
                f"output_data_dir already exists: {cfg.output_data_dir}\n"
                "Pick a new path, set overwrite: true, or continue: true with matching YAML."
            )
    else:
        cfg.output_data_dir.mkdir(parents=True, exist_ok=False)

    if cfg_src.resolve() != cfg_dst.resolve():
        shutil.copy2(cfg_src, cfg_dst)
        print(f"[init] copied run YAML to {cfg_dst}", flush=True)

    # Precount denominators (OpenRouter-style)
    metrics = Metrics()
    metrics.total_parquets = len(shard_paths)
    try:
        row_total = sum(precount_doc_rows([p]) for p in shard_paths)
    except Exception as e:
        print(f"[init] row precount skipped: {e}", flush=True)
        row_total = -1

    print("[init] precounting chars + rough token estimate …", flush=True)
    try:
        parquet_chars, parquet_tok_est = precount_chars_and_token_est(
            shard_paths, row_text_fn
        )
    except Exception as e:
        print(f"[init] char/token precount skipped: {e}", flush=True)
        parquet_chars, parquet_tok_est = -1, -1

    metrics.doc_denom = _doc_denom(cfg.max_docs, row_total)
    metrics.char_denom = _char_denom(cfg.max_chars, parquet_chars)
    metrics.api_tok_denom = _api_tok_denom(cfg.max_tokens, parquet_tok_est)

    chunk_denom = cfg.max_chunks if cfg.max_chunks >= 0 else -1
    setattr(metrics, "chunk_denom", chunk_denom)

    mode_label = "embedding (per-token keep/discard)" if cfg.is_embedding_model else "denoise (autoregressive)"
    print(
        f"[init] mode={mode_label} checkpoint={cfg.model_checkpoint_path!r} arch={cfg.model_arch!r} "
        f"shards={len(shard_paths)} chunk_size={cfg.chunk_size} "
        f"max_docs={cfg.max_docs} max_chars={cfg.max_chars} max_chunks={cfg.max_chunks} "
        f"max_tokens={cfg.max_tokens} tensor_parallel={cfg.num_gpus} "
        f"data_parallel_replicas={cfg.data_parallel_replicas} is_code={cfg.is_code} "
        f"keep_logit_threshold={cfg.keep_logit_threshold}",
        flush=True,
    )
    print(
        f"[init] rows_in_shards={row_total} parquet_chars={parquet_chars} "
        f"parquet_tok_est={parquet_tok_est} doc_denom={metrics.doc_denom} "
        f"char_denom={metrics.char_denom} api_tok_denom={metrics.api_tok_denom} "
        f"chunk_denom={chunk_denom}",
        flush=True,
    )

    if cfg.data_parallel_replicas > 1:
        if cfg.continue_run:
            raise SystemExit(
                "continue: true is not supported when data_parallel_replicas > 1."
            )
        gpu_ids = _dp_gpu_ids_or_exit(cfg)
        if (
            cfg.max_docs >= 0
            or cfg.max_chars >= 0
            or cfg.max_chunks >= 0
            or cfg.max_tokens >= 0
        ):
            print(
                "[init] warning: with data_parallel_replicas > 1, max_docs / max_chars / "
                "max_chunks / max_tokens apply per replica (not as one global cap across GPUs).",
                flush=True,
            )
        print(
            f"[init] data_parallel_replicas={cfg.data_parallel_replicas} "
            f"tensor_parallel_per_replica={cfg.num_gpus} gpu_ids={gpu_ids}",
            flush=True,
        )
        _run_data_parallel(
            cfg_path=cfg_path.expanduser().resolve(),
            cfg=cfg,
            shard_paths=shard_paths,
            gpu_ids=gpu_ids,
        )
        return

    _ensure_veomni_on_path()
    resolved_ckpt = _resolve_model_checkpoint_for_vllm(cfg.model_checkpoint_path)
    if cfg.is_embedding_model:
        print(f"[init] HuggingFace model path (resolved, embedding mode): {resolved_ckpt}", flush=True)
        _execute_embedding_pipeline(
            cfg,
            shard_paths,
            resolved_ckpt,
            resume_ok=resume_ok,
            log_prefix="",
        )
        return

    print(f"[init] vLLM model path (resolved): {resolved_ckpt}", flush=True)
    _execute_denoise_pipeline(
        cfg,
        shard_paths,
        resolved_ckpt,
        resume_ok=resume_ok,
        log_prefix="",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("yaml_config", type=Path, help="Path to YAML config file.")
    args = ap.parse_args()
    os.chdir(_REPO_ROOT)
    run(args.yaml_config.expanduser().resolve())


if __name__ == "__main__":
    main()
