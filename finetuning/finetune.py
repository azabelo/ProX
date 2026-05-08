#!/usr/bin/env python3
"""
Finetune on (input chunk, OpenRouter output chunk) pairs from parquet dirs.

Usage (multi-GPU, VeOmni path):
  torchrun --standalone --nproc_per_node=8 finetuning/finetune.py path/to/your_finetune.yaml

VeOmni loads ``torch.distributed`` even for one GPU. When ``LOCAL_RANK`` is unset, this script either
re-execs via ``torch.distributed.run`` (``num_gpus`` > 1) or sets single-process ``RANK``/``WORLD_SIZE`` env vars.

YAML schema (top-level keys):
  - model_arch: str (informational; also merged into ``veomni.model.foundation`` when provided)
  - model_type: veomni_supported | huggingface
  - init_model_path: HuggingFace hub id or local path for base weights (maps to VeOmni ``model.model_path``)
  - resume_checkpoint_path: optional VeOmni DCP dir (must contain ``.metadata``). Relative paths: try ``cwd`` first,
    then the YAML file's directory. Invalid/missing checkpoints fail fast before distributed launch.
  - inputs_dir_path: directory of source shards (``.parquet`` or ``*.jsonl.zst``)
  - outputs_dir_path: OpenRouter output directory (copied YAML + ``*_openrouter.parquet``)
  - dataset_type: fineweb | dclm | redpajama | redpajama-v2 | passthrough (row text adapter, same as OpenRouter runner)
  - chunk_size: int (character chunks; must match OpenRouter run — mismatched input vs output chunk counts drop the whole doc)
  - is_code: bool — read ``programs_delimited`` vs ``text`` from output rows
  - packing: bool — VeOmni: enables ``dyn_bsz`` token packing; HF: packs examples in the collator up to ``max_seq_len``
  - include_loss_from_input: bool default false — false: CE only on assistant/output tokens (VeOmni ``text_target``)
  - diffusion_lm: bool default false — for masked diffusion LMs (e.g. ``llada_mini``), patches the VeOmni collator after trainer init so labels stay aligned with ``input_ids`` (no causal shift)
  - num_gpus: int — used only for auto ``torchrun`` re-exec
  - num_validation_documents: int — documents held out entirely for validation (by ``source_parquet`` + ``source_row_index``)
  - max_seq_len, max_steps, global_batch_size, micro_batch_size, learning_rate, weight_decay, warmup_ratio,
    lr_decay_style, max_grad_norm, save_every_steps, eval_every_steps, checkpoint_output_dir, seed, wandb_project,
    wandb_run_name, gradient_checkpointing, tokenizer_path (optional override)
  - model_config: optional mapping — HF-style model hyperparameters for random-init/custom architectures; merged last,
    overrides ``veomni_overrides.model.config_path``, written to ``checkpoint_output_dir/.finetune_inline_config/config.json``

Optional nested ``veomni_overrides:`` (legacy ``veomni:``) dict is deep-merged into the constructed VeOmni YAML dict.

For random-init / custom stacks (``test_everything``, ``llada_mini``, ``qwen2_swa``, ``qwen2_mamba``, …), put the HuggingFace-style
model JSON fields under top-level ``model_config:`` (same keys as ``config.json``). At startup, ``finetune.py`` writes
``<checkpoint_output_dir>/.finetune_inline_config/config.json`` so VeOmni can load it—no checked-in ``config.json``
required.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Repo-relative imports (OpenRouter delimiter + text adapters + chunking)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openrouter_data_generation.run_openrouter_chunked import (  # noqa: E402
    PROGRAM_CHUNK_SEPARATOR,
    chunk_text,
    get_row_text_fn,
    iter_shard_rows,
    sorted_shard_paths,
)

_VEOMNI_SRC = Path(__file__).resolve().parent / "veomni"


def _ensure_single_process_dist_env() -> None:
    """VeOmni calls ``dist.init_process_group()``; plain ``python`` leaves ``RANK`` unset (torchrun sets it)."""
    if os.environ.get("RANK") is not None:
        return
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _host, port = s.getsockname()
    s.close()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"


def _maybe_reexec_torchrun(cfg: "FinetuneConfig") -> None:
    """VeOmni requires a distributed launch; multi-GPU re-execs under torch.distributed.run."""
    if os.environ.get("LOCAL_RANK") is not None:
        return
    if cfg.model_type != "veomni_supported":
        return
    n = max(1, int(cfg.num_gpus))
    if n > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={n}",
            __file__,
            cfg.config_path,
        ]
        print("[finetune] re-exec:", " ".join(cmd), flush=True)
        os.execvpe(sys.executable, cmd, os.environ)
    _ensure_single_process_dist_env()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SystemExit("Config root must be a mapping.")
    return raw


def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    raise SystemExit(f"Invalid boolean: {v!r}")


def _warn_output_chunk_size(outputs_dir: Path, training_chunk_size: int) -> None:
    """Warn if copied OpenRouter YAML ``chunk_size`` differs from training ``chunk_size``."""
    for p in sorted(outputs_dir.glob("*.yaml")) + sorted(outputs_dir.glob("*.yml")):
        try:
            cfg = _load_yaml(p)
        except Exception:
            continue
        if "chunk_size" not in cfg:
            continue
        oc = int(cfg["chunk_size"])
        if oc != int(training_chunk_size):
            warnings.warn(
                f"Training chunk_size={training_chunk_size} differs from chunk_size={oc} "
                f"in output dir YAML {p.name!r}. Input chunks are built with the training "
                f"value; OpenRouter outputs were produced with {oc}-char chunks — alignment may be wrong.",
                stacklevel=2,
            )
        return
    print(
        "[finetune] No YAML with chunk_size found under outputs_dir; "
        "could not verify OpenRouter chunk_size.",
        flush=True,
    )


def _output_merged_text(row: dict[str, Any], is_code: bool) -> str:
    if is_code:
        return str(row.get("programs_delimited") or "")
    return str(row.get("text") or "")


def _split_output_parts(merged: str) -> list[str]:
    if not merged:
        return []
    if PROGRAM_CHUNK_SEPARATOR in merged:
        return [p for p in merged.split(PROGRAM_CHUNK_SEPARATOR)]
    return [merged]


def _strip_chunk_separators(text: str) -> str:
    if not text:
        return text
    return text.replace(PROGRAM_CHUNK_SEPARATOR, "")


def character_keep_mask(reference: str, revised: str) -> str:
    """``'0'``/``'1'`` per reference codepoint; ``'1'`` iff inside a SequenceMatcher ``equal`` opcode.

    Same rule as ``openrouter_data_generation/view_nth_input_output_diff.py`` (``_simple_diff_merge_html``):
    insertions on the revised side do not change mask length; only ``equal`` reference spans are kept.
    """
    from difflib import SequenceMatcher

    n = len(reference)
    if n == 0:
        return ""
    sm = SequenceMatcher(None, reference, revised, autojunk=False)
    parts = ["0"] * n
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal" and i2 > i1:
            parts[i1:i2] = ["1"] * (i2 - i1)
    return "".join(parts)


def _read_row_by_index(shard: Path, row_index: int) -> dict[str, Any]:
    idx = 0
    for row in iter_shard_rows(shard):
        if idx == row_index:
            return row
        idx += 1
    raise KeyError(f"Row {row_index} not found in {shard}")


def _resolve_input_shard(inputs_dir: Path, source_parquet: str) -> Path:
    cand = inputs_dir / source_parquet
    if cand.is_file():
        return cand
    base = Path(source_parquet).name
    cand2 = inputs_dir / base
    if cand2.is_file():
        return cand2
    # recursive search by basename
    for p in inputs_dir.rglob(base):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not resolve input shard for source_parquet={source_parquet!r} under {inputs_dir}")


def _discover_output_parquets(outputs_dir: Path) -> list[Path]:
    out = sorted({p for p in outputs_dir.rglob("*.parquet") if p.is_file()})
    if not out:
        raise SystemExit(f"No parquet files under outputs_dir_path={outputs_dir}")
    return out


@dataclass
class FinetuneConfig:
    config_path: str
    model_arch: str = "auto"
    model_type: str = "veomni_supported"
    init_model_path: str = ""
    tokenizer_path: str | None = None
    resume_checkpoint_path: str | None = None
    inputs_dir_path: str = ""
    outputs_dir_path: str = ""
    dataset_type: str = "fineweb"
    chunk_size: int = 4096
    is_code: bool = False
    packing: bool = False
    include_loss_from_input: bool = False
    num_gpus: int = 1
    num_validation_documents: int = 8
    max_seq_len: int = 2048
    max_steps: int | None = None
    num_train_epochs: int = 1
    global_batch_size: int = 8
    micro_batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_decay_style: str = "cosine"
    max_grad_norm: float = 1.0
    save_every_steps: int = 500
    eval_every_steps: int = 0
    eval_max_batches: int = 32
    checkpoint_output_dir: str = "./finetune_outputs/run1"
    seed: int = 42
    wandb_project: str = "prox-finetune"
    wandb_run_name: str | None = None
    gradient_checkpointing: bool = True
    veomni_overrides: dict[str, Any] = field(default_factory=dict)
    # If True, append tokenizer eos to each target string when encode(target) does not already end with eos_token_id
    # so the last supervised token in chat/text_target loss can be EOS (helps learned stopping in generate()).
    ensure_eos_on_target: bool = True
    diffusion_lm: bool = False
    # When True, train a per-token binary keep/discard classifier (e.g. ``qwen2_embedding``) instead of a causal LM.
    #
    # The training pair becomes (raw input chunk, per-character ``'0'``/``'1'`` keep mask of equal length).
    # The mask is computed on the fly via ``difflib.SequenceMatcher.get_opcodes()`` (same rule as
    # ``openrouter_data_generation/view_nth_input_output_diff.py`` and the standalone
    # ``finetuning/create_keep_discard_dataset.py`` helper) — only ``equal`` reference spans are kept.
    # During tokenization the per-character mask is folded into a per-token mask via logical AND
    # (any deleted character in a token's offset span -> token label = 0; otherwise 1). Tokens with
    # empty character spans (e.g. special tokens) are labeled ``IGNORE_INDEX`` and excluded from loss.
    is_embedding_model: bool = False
    # HuggingFace-style model config dict (optional). When set, overrides ``veomni_overrides.model.config_path`` and
    # is written to ``checkpoint_output_dir/.finetune_inline_config/config.json`` at run time.
    model_config: dict[str, Any] | None = None

    @staticmethod
    def from_dict(raw: dict[str, Any], config_path: str) -> "FinetuneConfig":
        def g(key: str, default: Any = None) -> Any:
            return raw[key] if key in raw else default

        _mc_raw = g("model_config")
        return FinetuneConfig(
            config_path=config_path,
            model_arch=str(g("model_arch", "auto")),
            model_type=str(g("model_type", "veomni_supported")),
            init_model_path=str(g("init_model_path", g("model_path", ""))),
            tokenizer_path=g("tokenizer_path"),
            resume_checkpoint_path=g("resume_checkpoint_path", g("resume_checkpoint", None)),
            inputs_dir_path=str(g("inputs_dir_path", "")),
            outputs_dir_path=str(g("outputs_dir_path", "")),
            dataset_type=str(g("dataset_type", "fineweb")),
            chunk_size=int(g("chunk_size", 4096)),
            is_code=_bool(g("is_code", False)),
            packing=_bool(g("packing", False)),
            include_loss_from_input=_bool(g("include_loss_from_input", False)),
            num_gpus=int(g("num_gpus", 1)),
            num_validation_documents=int(g("num_validation_documents", 8)),
            max_seq_len=int(g("max_seq_len", 2048)),
            max_steps=g("max_steps", None),
            num_train_epochs=int(g("num_train_epochs", 1)),
            global_batch_size=int(g("global_batch_size", 8)),
            micro_batch_size=int(g("micro_batch_size", 1)),
            learning_rate=float(g("learning_rate", 2e-5)),
            weight_decay=float(g("weight_decay", 0.01)),
            warmup_ratio=float(g("warmup_ratio", 0.03)),
            lr_decay_style=str(g("lr_decay_style", "cosine")),
            max_grad_norm=float(g("max_grad_norm", 1.0)),
            save_every_steps=int(g("save_every_steps", 500)),
            eval_every_steps=int(g("eval_every_steps", 0)),
            eval_max_batches=int(g("eval_max_batches", 32)),
            checkpoint_output_dir=str(g("checkpoint_output_dir", "./finetune_outputs/run1")),
            seed=int(g("seed", 42)),
            wandb_project=str(g("wandb_project", "prox-finetune")),
            wandb_run_name=g("wandb_run_name", None),
            gradient_checkpointing=_bool(g("gradient_checkpointing", True)),
            veomni_overrides=dict(g("veomni_overrides", g("veomni", {})) or {}),
            ensure_eos_on_target=_bool(g("ensure_eos_on_target", True)),
            diffusion_lm=_bool(g("diffusion_lm", False)),
            is_embedding_model=_bool(g("is_embedding_model", False)),
            model_config=_mc_raw if _mc_raw else None,
        )


def _finalize_resume_checkpoint_path(cfg: FinetuneConfig) -> None:
    """Resolve ``resume_checkpoint_path`` and require a valid DCP tree before VeOmni touches ``load_path``.

    VeOmni always consumes ``train.checkpoint.load_path`` when non-null, so a stale YAML path must not
    reach the trainer (otherwise every rank fails opening ``.metadata``).
    """
    raw = cfg.resume_checkpoint_path
    if raw is None:
        return
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        cfg.resume_checkpoint_path = None
        return
    cfg_yaml = Path(cfg.config_path).resolve()
    rel = Path(s).expanduser()
    if rel.is_absolute():
        p = rel.resolve()
    else:
        cwd_p = (Path.cwd() / rel).resolve()
        yaml_p = (cfg_yaml.parent / rel).resolve()
        if (cwd_p / ".metadata").is_file():
            p = cwd_p
        elif (yaml_p / ".metadata").is_file():
            p = yaml_p
        else:
            # Prefer cwd in errors (typical: paths mirror ``checkpoint_output_dir`` from repo root).
            p = cwd_p
    meta = p / ".metadata"
    if not p.is_dir():
        alt = ""
        if not rel.is_absolute():
            other = (yaml_p if p == cwd_p else cwd_p).resolve()
            if other != p:
                alt = f"\n  Also checked: {other}"
        raise SystemExit(
            f"[finetune] resume_checkpoint_path is not a directory:\n  {p}\n"
            f"  (YAML had {raw!r}; relative paths try cwd={Path.cwd().resolve()!s} then {cfg_yaml.parent!s}){alt}"
        )
    if not meta.is_file():
        ms = cfg.max_steps
        step_hint = (
            f"With save_every_steps={cfg.save_every_steps} and max_steps={ms!r}, "
            f"intermediate saves only occur at multiples of save_every_steps; "
            f"train-end may write global_step_<last_step> instead."
            if ms is not None
            else f"With save_every_steps={cfg.save_every_steps}, check which global_step_* dirs exist."
        )
        raise SystemExit(
            f"[finetune] resume_checkpoint_path is not a VeOmni DCP checkpoint (missing .metadata):\n  {meta}\n"
            f"  {step_hint}\n"
            f"  List checkpoints: ls {p.parent.resolve()!s}/global_step_* 2>/dev/null || true"
        )
    cfg.resume_checkpoint_path = str(p)


def _materialize_inline_model_config(cfg: FinetuneConfig) -> str | None:
    """Write ``cfg.model_config`` to disk for VeOmni ``AutoConfig.from_pretrained(dir)``.

    For ``is_embedding_model: true`` without an explicit ``model_config``, derive one from the
    pretrained ``init_model_path`` HF config (Qwen2.5-0.5B etc.): copy every hyperparameter, override
    ``model_type`` to ``qwen2_embedding``, set ``architectures`` to
    ``['Qwen2EmbeddingForTokenClassification']``, force ``tie_word_embeddings: false`` (no
    ``lm_head``), and append ``drop_last_n_layers: 1`` so the inner Qwen2 backbone has one fewer
    decoder block. Pretrained Qwen2 weights then load by name into the truncated backbone (last
    layer + ``lm_head`` are skipped as unexpected keys; the ``score`` head is initialized fresh by
    ``post_process_after_weight_loading``).
    """
    mc = getattr(cfg, "model_config", None)
    if not mc and getattr(cfg, "is_embedding_model", False) and cfg.init_model_path:
        try:
            from transformers import AutoConfig as _AutoConfig

            base = _AutoConfig.from_pretrained(cfg.init_model_path, trust_remote_code=True)
            mc = base.to_dict()
        except Exception as e:
            raise SystemExit(
                f"[finetune] is_embedding_model=true but could not autoload base config from "
                f"init_model_path={cfg.init_model_path!r}: {e}"
            )
        mc["model_type"] = "qwen2_embedding"
        mc["architectures"] = ["Qwen2EmbeddingForTokenClassification"]
        mc["tie_word_embeddings"] = False
        mc.setdefault("drop_last_n_layers", 1)
    if not mc:
        return None
    out_dir = Path(cfg.checkpoint_output_dir).expanduser().resolve() / ".finetune_inline_config"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "config.json"
    dest.write_text(json.dumps(mc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"[finetune] inline model_config written to {dest}", flush=True)
    return str(out_dir)


def _ensure_eos_on_targets(cfg: FinetuneConfig, rows: list[dict[str, str]]) -> None:
    """Append tokenizer EOS to ``target`` when missing so CE can supervise EOS as last assistant token."""
    if not getattr(cfg, "ensure_eos_on_target", True) or not rows:
        return
    tok_path = cfg.tokenizer_path or cfg.init_model_path
    if not tok_path:
        print("[finetune][warn] ensure_eos_on_target needs tokenizer_path or init_model_path; skipping.", flush=True)
        return
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except Exception as e:
        print(f"[finetune][warn] ensure_eos_on_target skipped: {e}", flush=True)
        return
    eos_id = tok.eos_token_id
    eos_piece = tok.eos_token
    if eos_id is None or not eos_piece:
        print("[finetune][warn] tokenizer has no eos_token_id/eos_token; cannot append EOS.", flush=True)
        return
    n_changed = 0
    for row in rows:
        t = row["target"]
        ids = tok.encode(t, add_special_tokens=False)
        if not ids or ids[-1] != eos_id:
            row["target"] = t + eos_piece
            n_changed += 1
    if n_changed:
        print(
            f"[finetune] ensure_eos_on_target: appended EOS to {n_changed}/{len(rows)} targets "
            f"(so labels can include eos_token_id where chat template expects it).",
            flush=True,
        )


def build_pair_rows(cfg: FinetuneConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Build train/val rows with keys ``text`` (input chunk) and ``target`` (output chunk).

    Documents are split wholly into train or validation using ``num_validation_documents``
    unique (source_parquet, source_row_index) keys.

    If re-chunking the source text and splitting the OpenRouter output yield different
    chunk counts for a document, that document is skipped entirely (no partial pairs).
    """
    inputs_dir = Path(cfg.inputs_dir_path).expanduser().resolve()
    outputs_dir = Path(cfg.outputs_dir_path).expanduser().resolve()
    if not inputs_dir.is_dir():
        raise SystemExit(f"inputs_dir_path is not a directory: {inputs_dir}")
    if not outputs_dir.is_dir():
        raise SystemExit(f"outputs_dir_path is not a directory: {outputs_dir}")

    _warn_output_chunk_size(outputs_dir, cfg.chunk_size)

    row_text_fn = get_row_text_fn(cfg.dataset_type)
    doc_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for pq_out in _discover_output_parquets(outputs_dir):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(pq_out).iter_batches(columns=["source_parquet", "source_row_index"]):
            sp = batch.column(0).to_pylist()
            sr = batch.column(1).to_pylist()
            for a, b in zip(sp, sr):
                key = (str(a), int(b))
                if key not in seen:
                    seen.add(key)
                    doc_keys.append(key)

    rng = random.Random(cfg.seed)
    rng.shuffle(doc_keys)
    n_val = min(cfg.num_validation_documents, len(doc_keys))
    val_set = set(doc_keys[:n_val])
    train_set = set(doc_keys[n_val:])

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []

    dropped_mismatch = 0
    for pq_out in _discover_output_parquets(outputs_dir):
        import pyarrow.parquet as pq

        for row in pq.ParquetFile(pq_out).iter_batches():
            col_names = row.schema.names
            for i in range(row.num_rows):
                rec = {name: row.column(j)[i].as_py() for j, name in enumerate(col_names)}
                sp = str(rec.get("source_parquet") or "")
                sr = rec.get("source_row_index")
                if sr is None:
                    continue
                key = (sp, int(sr))
                merged = _output_merged_text(rec, cfg.is_code)
                out_parts = _split_output_parts(merged)
                try:
                    shard = _resolve_input_shard(inputs_dir, sp)
                    src_row = _read_row_by_index(shard, int(sr))
                except Exception as e:
                    print(f"[finetune][warn] skip row {key}: {e}", flush=True)
                    continue
                doc_text = row_text_fn(src_row)
                in_chunks = chunk_text(doc_text, cfg.chunk_size)
                n_in, n_out = len(in_chunks), len(out_parts)
                if n_in != n_out:
                    dropped_mismatch += 1
                    continue
                if n_in == 0:
                    continue
                bucket = val_rows if key in val_set else train_rows
                for ci in range(n_in):
                    bucket.append({"text": in_chunks[ci], "target": out_parts[ci]})

    _ensure_eos_on_targets(cfg, train_rows)
    _ensure_eos_on_targets(cfg, val_rows)

    if dropped_mismatch:
        print(
            f"[finetune] dropped {dropped_mismatch} document(s) with mismatched input/output chunk counts "
            f"(expected equal counts per doc; check chunk_size vs OpenRouter run and delimiters).",
            flush=True,
        )
    if not train_rows:
        raise SystemExit("No training pairs built; check paths, keys, and filters.")
    print(
        f"[finetune] built {len(train_rows)} train pairs, {len(val_rows)} val pairs "
        f"({len(train_set)} train docs, {len(val_set)} val docs).",
        flush=True,
    )
    _rng_sanity = random.Random(cfg.seed + 1337)
    pool = train_rows if train_rows else val_rows
    if pool:
        ex = _rng_sanity.choice(pool)
        t, u = ex["text"], ex["target"]
        print(
            "[finetune] sanity sample (one random pair; full strings):\n"
            "--- input (text) ---\n"
            f"{t}\n"
            "--- output (target) ---\n"
            f"{u}",
            flush=True,
        )
    return train_rows, val_rows


def build_embedding_pair_rows(cfg: FinetuneConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Build train/val rows for the per-token keep/discard binary task.

    Each row's ``text`` is a raw input chunk (same chunking rule as ``build_pair_rows``); ``target``
    is a string of ``'0'``/``'1'`` of equal length to ``text``, where ``'1'`` marks characters that
    appear in a SequenceMatcher ``equal`` span vs the OpenRouter rewritten chunk (everything else =
    deleted/replaced -> ``'0'``). No giant char-mask parquet is materialized; the masks are computed
    here at row-build time from raw input + OpenRouter output (same pipeline as
    ``finetuning/create_keep_discard_dataset.py``, but kept inline so output parquet of ``0``/``1``
    is not needed).
    """
    inputs_dir = Path(cfg.inputs_dir_path).expanduser().resolve()
    outputs_dir = Path(cfg.outputs_dir_path).expanduser().resolve()
    if not inputs_dir.is_dir():
        raise SystemExit(f"inputs_dir_path is not a directory: {inputs_dir}")
    if not outputs_dir.is_dir():
        raise SystemExit(f"outputs_dir_path is not a directory: {outputs_dir}")

    _warn_output_chunk_size(outputs_dir, cfg.chunk_size)

    row_text_fn = get_row_text_fn(cfg.dataset_type)
    doc_keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    import pyarrow.parquet as pq

    for pq_out in _discover_output_parquets(outputs_dir):
        for batch in pq.ParquetFile(pq_out).iter_batches(columns=["source_parquet", "source_row_index"]):
            sp = batch.column(0).to_pylist()
            sr = batch.column(1).to_pylist()
            for a, b in zip(sp, sr):
                key = (str(a), int(b))
                if key not in seen:
                    seen.add(key)
                    doc_keys.append(key)

    rng = random.Random(cfg.seed)
    rng.shuffle(doc_keys)
    n_val = min(cfg.num_validation_documents, len(doc_keys))
    val_set = set(doc_keys[:n_val])

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    dropped_mismatch = 0

    for pq_out in _discover_output_parquets(outputs_dir):
        for row in pq.ParquetFile(pq_out).iter_batches():
            col_names = row.schema.names
            for i in range(row.num_rows):
                rec = {name: row.column(j)[i].as_py() for j, name in enumerate(col_names)}
                sp = str(rec.get("source_parquet") or "")
                sr = rec.get("source_row_index")
                if sr is None:
                    continue
                key = (sp, int(sr))
                merged = _output_merged_text(rec, cfg.is_code)
                merged = _strip_chunk_separators(merged)
                out_parts = _split_output_parts(merged) if not merged or PROGRAM_CHUNK_SEPARATOR in merged else [merged]
                try:
                    shard = _resolve_input_shard(inputs_dir, sp)
                    src_row = _read_row_by_index(shard, int(sr))
                except Exception as e:
                    print(f"[finetune][warn] skip row {key}: {e}", flush=True)
                    continue
                doc_text = row_text_fn(src_row)
                in_chunks = chunk_text(doc_text, cfg.chunk_size)
                n_in, n_out = len(in_chunks), len(out_parts)
                if n_in != n_out:
                    dropped_mismatch += 1
                    continue
                if n_in == 0:
                    continue
                bucket = val_rows if key in val_set else train_rows
                for ci in range(n_in):
                    src_chunk = in_chunks[ci]
                    rev_chunk = out_parts[ci]
                    mask = character_keep_mask(src_chunk, rev_chunk)
                    if len(mask) != len(src_chunk):
                        continue
                    bucket.append({"text": src_chunk, "target": mask})

    if dropped_mismatch:
        print(
            f"[finetune][embedding] dropped {dropped_mismatch} document(s) with mismatched "
            f"input/output chunk counts.",
            flush=True,
        )
    if not train_rows:
        raise SystemExit("No keep/discard training pairs built; check paths and chunk_size.")
    print(
        f"[finetune][embedding] built {len(train_rows)} train pairs, {len(val_rows)} val pairs.",
        flush=True,
    )
    return train_rows, val_rows


def _register_text_per_token_binary() -> None:
    """Register VeOmni transform for keep/discard token-level binary classification.

    Tokenizes ``text`` (no special tokens / no chat template — special tokens have no character span
    and would always be excluded anyway). For each resulting token, builds a per-token label by AND-ing
    every character of ``target`` (the per-character ``'0'``/``'1'`` mask) inside the token's offset
    range: any deleted character in the token -> label ``0``, otherwise ``1``. Tokens with empty offset
    spans get ``IGNORE_INDEX``.
    """
    from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
    from veomni.utils.constants import IGNORE_INDEX

    # ``Registry.__contains__`` raises on missing keys (delegates to ``__getitem__``); use the safe iter form.
    if "text_per_token_binary" in list(DATA_TRANSFORM_REGISTRY):
        return

    @DATA_TRANSFORM_REGISTRY.register("text_per_token_binary")
    def process_text_per_token_binary(
        example: dict[str, Any],
        tokenizer: Any,
        max_seq_len: int,
        chat_template: Any = None,
        text_keys: Any = "text",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del chat_template, text_keys, kwargs
        import torch

        text = str(example.get("text", "") or "")
        target = str(example.get("target", "") or "")
        if len(target) != len(text):
            # length mismatch should never happen; fall back to "all ignore" rather than corrupt training.
            target = "0" * len(text)

        if not text:
            fallback_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            )
            return [
                {
                    "input_ids": torch.tensor([int(fallback_id)], dtype=torch.long),
                    "attention_mask": torch.tensor([1], dtype=torch.long),
                    "labels": torch.tensor([IGNORE_INDEX], dtype=torch.long),
                }
            ]

        # Fast tokenizer required for offsets. Hugging Face Qwen2 tokenizer has it.
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_seq_len,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        n = len(text)
        labels: list[int] = []
        for s, e in offsets:
            if e <= s or s >= n:
                labels.append(IGNORE_INDEX)
                continue
            e_clamped = min(int(e), n)
            chunk = target[int(s) : e_clamped]
            if not chunk:
                labels.append(IGNORE_INDEX)
                continue
            labels.append(0 if "0" in chunk else 1)

        if not ids:
            return [
                {
                    "input_ids": torch.tensor([0], dtype=torch.long),
                    "attention_mask": torch.tensor([1], dtype=torch.long),
                    "labels": torch.tensor([IGNORE_INDEX], dtype=torch.long),
                }
            ]
        return [
            {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        ]


def _deep_update(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def _register_full_loss_text_target() -> None:
    """Register ``text_target_all`` — same formatting as ``text_target`` but CE on every token (like plaintext)."""
    import torch
    from typing import List, Union

    from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY, _truncate_ids_labels

    if "text_target_all" in list(DATA_TRANSFORM_REGISTRY):
        return

    @DATA_TRANSFORM_REGISTRY.register("text_target_all")
    def process_text_target_all(
        example: dict[str, Any],
        tokenizer: Any,
        max_seq_len: int,
        chat_template: Any = None,
        text_keys: Union[str, list[str]] = "text",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del chat_template, text_keys, kwargs
        src = str(example.get("text", "") or "")
        tgt = str(example.get("target", "") or "")
        system_msg = "You are a helpful, respectful and honest assistant."

        fallback_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0

        if not src.strip() and not tgt.strip():
            tid = int(fallback_id)
            input_ids = [tid]
            return [
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),
                }
            ]

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": src},
            {"role": "assistant", "content": tgt},
        ]
        prompt_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": src},
        ]

        if getattr(tokenizer, "chat_template", None) is None:
            prompt_txt = f"[SYSTEM]\n{system_msg}\n[USER]\n{src}\n[ASSISTANT]\n"
            full_txt = prompt_txt + tgt
            full_ids = tokenizer.encode(full_txt, add_special_tokens=True)
            labels_list = list(full_ids)
            input_ids, labels_list = _truncate_ids_labels(full_ids, labels_list, max_seq_len)
            return [
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels_list, dtype=torch.long),
                }
            ]

        full_ids: List[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids: List[int] = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        del prompt_ids  # formatting parity with text_target; loss still applies to full_ids
        labels_list = list(full_ids)

        input_ids, labels_list = _truncate_ids_labels(full_ids, labels_list, max_seq_len)
        return [
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels_list, dtype=torch.long),
            }
        ]


def _veomni_dict(cfg: FinetuneConfig, train_parquet: str, eval_parquet: str | None) -> dict[str, Any]:
    foundation: dict[str, str] = {}
    if cfg.model_arch and cfg.model_arch != "auto":
        foundation["architecture"] = cfg.model_arch

    dyn_bsz = bool(cfg.packing)
    if cfg.is_embedding_model:
        data_type = "text_per_token_binary"
    elif cfg.include_loss_from_input:
        data_type = "text_target_all"
    else:
        data_type = "text_target"

    base: dict[str, Any] = {
        "model": {
            "model_path": cfg.init_model_path,
            "tokenizer_path": cfg.tokenizer_path or cfg.init_model_path,
            "foundation": foundation,
            "ops_implementation": {"attn_implementation": "sdpa"},
        },
        "data": {
            "train_path": train_parquet,
            "eval_path": eval_parquet,
            "datasets_type": "mapping",
            "data_type": data_type,
            "text_keys": "text",
            "max_seq_len": cfg.max_seq_len,
            "train_sample": 1_000_000,
            "dataloader": {
                "type": "native",
                "num_workers": 2,
                "drop_last": True,
            },
        },
        "train": {
            "dyn_bsz": dyn_bsz,
            "micro_batch_size": cfg.micro_batch_size,
            "global_batch_size": cfg.global_batch_size,
            "num_train_epochs": cfg.num_train_epochs,
            "max_steps": cfg.max_steps,
            "init_device": "cuda",
            "seed": cfg.seed,
            "gradient_checkpointing": {"enable": cfg.gradient_checkpointing},
            "broadcast_model_weights_from_rank0": False,
            "eval_steps": cfg.eval_every_steps,
            "eval_epochs": 0,
            "optimizer": {
                "type": "adamw",
                "lr": cfg.learning_rate,
                "lr_warmup_ratio": cfg.warmup_ratio,
                "lr_decay_style": cfg.lr_decay_style,
                "weight_decay": cfg.weight_decay,
                "max_grad_norm": cfg.max_grad_norm,
            },
            "checkpoint": {
                "output_dir": cfg.checkpoint_output_dir,
                "manager": "dcp",
                "save_steps": cfg.save_every_steps,
                "save_hf_weights": False,
                "load_path": cfg.resume_checkpoint_path,
            },
            "wandb": {
                "enable": True,
                "project": cfg.wandb_project,
                "name": cfg.wandb_run_name or Path(cfg.checkpoint_output_dir).name,
            },
            "accelerator": {
                "ulysses_size": 1,
                "fsdp_config": {
                    "fsdp_mode": "ddp",
                    "full_shard": False,
                    "offload": False,
                    "mixed_precision": {"enable": True},
                },
            },
        },
    }
    merged = _deep_update(base, dict(cfg.veomni_overrides))
    inline_cfg_dir = _materialize_inline_model_config(cfg)
    if inline_cfg_dir:
        merged.setdefault("model", {})["config_path"] = inline_cfg_dir
    # ``DistributedDataParallel`` + ``MixedPrecision`` around ``nn.Embedding`` breaks autograd hooks on
    # PyTorch 2.x (SIGSEGV / None grad_fn / mixed_precision_hooks NoneType). Llada uses FP32 embedding
    # tables and BF16 blocks without DDP's MP wrapper (see ``modeling_llada_mini``).
    if cfg.diffusion_lm:
        merged.setdefault("train", {}).setdefault("accelerator", {}).setdefault("fsdp_config", {}).setdefault(
            "mixed_precision", {}
        )["enable"] = False
    # Per-token binary classifier: same DDP+MP embedding fragility as ``llada_mini`` (FP32 embed table /
    # BF16 blocks) — and there is no ``lm_head`` to back-tie. Keep mixed precision off so the autograd
    # hooks installed by DDP's MixedPrecision wrapper don't NPE on the embedding parameter.
    if cfg.is_embedding_model:
        merged.setdefault("train", {}).setdefault("accelerator", {}).setdefault("fsdp_config", {}).setdefault(
            "mixed_precision", {}
        )["enable"] = False
    return merged


def _write_pair_parquet(rows: list[dict[str, str]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _unwrap_for_generate(model: Any) -> Any:
    """Peel DDP / FSDP wrappers so ``generate`` can run when supported."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    while hasattr(m, "_fsdp_wrapped_module"):
        m = m._fsdp_wrapped_module
    return m


@contextmanager
def _inference_inner(model: Any):
    """Yield the unwrapped inner module for **rank-0-only** eval, with DDP comm + MP hooks neutralized.

    Two distinct multi-GPU pitfalls are addressed:

    1. **DDP buffer-sync collective.** ``DistributedDataParallel`` defaults to ``broadcast_buffers=True`` and
       runs ``_sync_buffers()`` (a NCCL broadcast) inside ``_pre_forward`` on every call. Models with
       persistent buffers (e.g. Qwen ``rotary_emb.inv_freq``) deadlock if only rank 0 calls forward while
       the other ranks wait at ``dist.barrier()``. Calling the **inner unwrapped module** bypasses
       ``_pre_forward`` entirely.

    2. **DDP mixed-precision forward pre-hooks.** ``_root_copy_hook`` (on the inner root) and
       ``_module_wait_for_copy_hook`` (on every submodule) get installed when MP is enabled. Under
       ``torch.no_grad()``, ``tmp = p.expand_as(p); tmp.grad_fn.next_functions[0][0]`` raises
       ``'NoneType' object has no attribute 'next_functions'``. We detach both for the scope; they are
       not needed for rank-0 forward (no comm, no MP copies needed for inference).

    Also clears ``requires_grad`` defensively so any other autograd-aware hook short-circuits.
    """
    try:
        from torch.nn.parallel.distributed import DistributedDataParallel as _DDP
    except Exception:
        _DDP = None  # type: ignore[assignment]

    inner = _unwrap_for_generate(model)
    ddp = model if (_DDP is not None and isinstance(model, _DDP)) else None

    saved_hooks: list[tuple[Any, int, Any, bool]] = []
    if ddp is not None and getattr(ddp, "mixed_precision", None) is not None:
        target_funcs = {
            getattr(ddp._root_copy_hook, "__func__", None),
            getattr(ddp._module_wait_for_copy_hook, "__func__", None),
        }
        target_funcs.discard(None)
        for sub in inner.modules():
            pre = getattr(sub, "_forward_pre_hooks", None)
            if not pre:
                continue
            wk = getattr(sub, "_forward_pre_hooks_with_kwargs", {}) or {}
            for hid in list(pre.keys()):
                hcall = pre[hid]
                f = getattr(hcall, "__func__", None)
                s = getattr(hcall, "__self__", None)
                if f in target_funcs and s is ddp:
                    saved_hooks.append((sub, hid, hcall, bool(wk.get(hid, False))))
                    del pre[hid]
                    wk.pop(hid, None)

    params = list(inner.parameters())
    prev_rg = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield inner
    finally:
        for p, r in zip(params, prev_rg):
            p.requires_grad_(r)
        for sub, hid, hcall, with_kwargs in saved_hooks:
            sub._forward_pre_hooks[hid] = hcall
            if with_kwargs:
                sub._forward_pre_hooks_with_kwargs[hid] = True


# Back-compat alias: existing callers that only need the autograd/hook neutralization (no inner unwrap).
@contextmanager
def _ddp_mp_safe_no_grad(model: Any):
    """Backward-compatible shim around :func:`_inference_inner` (discards the yielded inner module)."""
    with _inference_inner(model):
        yield


def _masked_lm_token_sums(
    logits: Any, labels: Any, ignore_index: int
) -> tuple[float, float, int]:
    """Sum CE and entropy over supervised next-token positions (causal shift). Returns (ce_sum, ent_sum, n_tokens)."""
    import torch
    import torch.nn.functional as F

    sl = logits[:, :-1, :].contiguous()
    sy = labels[:, 1:].contiguous()
    v = sl.size(-1)
    flat_l = sl.view(-1, v)
    flat_y = sy.view(-1)
    m = flat_y != ignore_index
    if not m.any():
        return 0.0, 0.0, 0
    ce_sum = F.cross_entropy(flat_l[m], flat_y[m], reduction="sum")
    probs = F.softmax(flat_l[m].float(), dim=-1)
    ent_sum = (-(probs * probs.clamp_min(1e-12).log()).sum(-1)).sum()
    n = int(m.sum().item())
    return float(ce_sum.item()), float(ent_sum.item()), n


def _masked_lm_means_from_logits(
    logits: Any, labels: Any, ignore_index: int
) -> tuple[float | None, float | None, float | None]:
    """Mean CE, mean entropy, perplexity exp(mean CE); None if no supervised tokens."""
    ce_sum, ent_sum, n = _masked_lm_token_sums(logits, labels, ignore_index)
    if n == 0:
        return None, None, None
    mean_ce = ce_sum / n
    mean_ent = ent_sum / n
    return mean_ce, mean_ent, math.exp(mean_ce)


def _attach_veomni_lm_entropy_wandb(trainer: Any) -> None:
    """Each train step, log mean LM entropy from ``lm_head`` outputs (sequence mean; not label-masked)."""
    import torch
    import torch.nn.functional as F

    args = trainer.base.args
    if not args.train.wandb.enable:
        return
    inner = _unwrap_for_generate(trainer.base.model)
    head = getattr(inner, "lm_head", None)
    if head is None:
        return

    acc = {"sum": 0.0, "n": 0}

    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        # Parquet eval runs ``model.eval()`` on rank 0 only; skip hook work so we do not touch
        # ``lm_head`` outputs outside normal training forwards (avoids autograd/DDP edge cases).
        if not _mod.training:
            return
        if out is None:
            return
        with torch.no_grad():
            p = F.softmax(out.float(), dim=-1)
            e = -(p * p.clamp_min(1e-12).log()).sum(-1)
            acc["sum"] += float(e.mean().item())
            acc["n"] += 1

    head.register_forward_hook(_hook)

    _orig = trainer.on_step_end

    def _wrapped_step_end(self: Any, loss: Any = None, loss_dict: Any = None, grad_norm: Any = None) -> None:
        _orig(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        if self.base.args.train.global_rank != 0 or not self.base.args.train.wandb.enable:
            acc["sum"], acc["n"] = 0.0, 0
            return
        if acc["n"] <= 0:
            return
        try:
            import wandb

            wandb.log(
                {"train/entropy_lm_mean": acc["sum"] / acc["n"]},
                step=int(self.base.state.global_step),
            )
        except Exception:
            pass
        acc["sum"], acc["n"] = 0.0, 0

    trainer.on_step_end = types.MethodType(_wrapped_step_end, trainer)


def _eval_sanity_check(
    *,
    val_parquet_path: str | Path,
    model: Any,
    tokenizer: Any,
    device: Any,
    max_new_tokens: int,
    global_step: int,
    diffusion_lm: bool = False,
) -> None:
    """Print first row input/target from val parquet and a short prediction.

    Causal LMs use greedy ``generate``. For ``diffusion_lm`` (e.g. ``llada_mini``), runs one parallel
    decode at inference timestep 0: prompt + trailing ``mask_token_id`` positions, argmax on the suffix
    (sanity logging only; not full iterative diffusion sampling).

    Uses ``_ddp_mp_safe_no_grad`` (which detaches DDP mixed-precision forward pre-hooks for the scope) so
    HuggingFace ``generate()`` (decorated with ``@torch.no_grad``) doesn't crash under DDP+MP.
    """
    import torch

    path = Path(val_parquet_path)
    if not path.is_file():
        return
    try:
        import pyarrow.parquet as pq

        tab = pq.read_table(path, columns=["text", "target"])
    except Exception as e:
        print(f"[eval] sanity skipped (parquet read): {e}", flush=True)
        return
    if tab.num_rows == 0:
        print("[eval] sanity skipped (empty val parquet)", flush=True)
        return
    row = tab.slice(0, 1).to_pylist()[0]
    src = str(row.get("text") or "")
    tgt = str(row.get("target") or "")
    system_msg = "You are a helpful, respectful and honest assistant."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": src},
    ]
    tok = tokenizer

    pred = ""
    if diffusion_lm:
        try:
            with _inference_inner(model) as inner:
                with torch.no_grad():
                    mask_id = getattr(getattr(inner, "config", None), "mask_token_id", None)
                    if mask_id is None:
                        pred = "<skipped: diffusion_lm model has no config.mask_token_id>"
                    else:
                        if getattr(tok, "chat_template", None):
                            ids = tok.apply_chat_template(
                                messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_tensors="pt",
                            )
                        else:
                            prompt_txt = f"[SYSTEM]\n{system_msg}\n[USER]\n{src}\n[ASSISTANT]\n"
                            ids = tok(prompt_txt, return_tensors="pt")["input_ids"]
                        ids = ids.to(device)
                        prompt_len = int(ids.shape[1])
                        max_pos = int(getattr(inner.config, "max_position_embeddings", 2048))
                        cap = max(32, min(int(max_new_tokens), 4096))
                        gen_len = min(cap, max_pos - prompt_len)
                        if gen_len <= 0:
                            pred = "<skipped: prompt fills context — no room for mask suffix>"
                        else:
                            suffix = torch.full((1, gen_len), int(mask_id), dtype=torch.long, device=device)
                            full = torch.cat([ids, suffix], dim=1)
                            attn = torch.ones_like(full, dtype=torch.long, device=device)
                            out = inner(input_ids=full, attention_mask=attn, labels=None)
                            logits = out.logits
                            pred_ids = logits[0, prompt_len:, :].argmax(dim=-1)
                            pred = tok.decode(pred_ids.cpu(), skip_special_tokens=True)
        except Exception as e:
            pred = f"<diffusion sanity decode failed: {e}>"
    else:
        pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else tok.eos_token_id
        try:
            with _inference_inner(model) as inner:
                with torch.no_grad():
                    if getattr(tok, "chat_template", None):
                        ids = tok.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        )
                    else:
                        prompt_txt = f"[SYSTEM]\n{system_msg}\n[USER]\n{src}\n[ASSISTANT]\n"
                        ids = tok(prompt_txt, return_tensors="pt")["input_ids"]
                    ids = ids.to(device)
                    attn = torch.ones_like(ids, dtype=torch.long, device=device)
                    cap = max(32, min(int(max_new_tokens), 4096))
                    gen_out = inner.generate(
                        ids,
                        attention_mask=attn,
                        max_new_tokens=cap,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=tok.eos_token_id,
                        use_cache=True,
                    )
                    new_tokens = gen_out[0][ids.shape[1] :]
                    pred = tok.decode(new_tokens.cpu(), skip_special_tokens=True)
        except Exception as e:
            pred = f"<generate failed: {e}>"

    print(
        f"[eval] sanity check (step={global_step}; first val row)\n"
        f"--- input ---\n{src}\n"
        f"--- target ---\n{tgt}\n"
        f"--- prediction ---\n{pred}",
        flush=True,
    )


def _build_parquet_eval_callback(
    trainer: Any, val_parquet: str | None, max_batches: int, diffusion_lm: bool = False
) -> Any:
    """Mean CE on a held-out pair parquet.

    Evaluation runs on **rank 0 only** (parquet I/O + short greedy sanity decode). Non-zero ranks wait at
    ``dist.barrier()`` before and after so they never advance to the next training step / ``on_train_end``
    while rank 0 is still in eval (avoids DDP deadlock).

    For ``diffusion_lm=True``, uses the model's scalar diffusion loss (masked CE); causal shifted CE is wrong for MDM.
    """

    from veomni.trainer.callbacks.evaluate_callback import EvaluateCallback

    class ParquetEvalCallback(EvaluateCallback):
        def __init__(self, t: Any, vp: str | None, mb: int, diffusion: bool) -> None:
            super().__init__(t)
            self.val_parquet = vp
            self.max_batches = mb
            self.diffusion_lm = diffusion
            self._built = False
            self._loader = None

        def on_train_end(self, state: Any) -> None:
            """If training stopped between eval intervals, run eval once (HF-style final eval)."""
            args = self.trainer.args
            es = getattr(args.train, "eval_steps", None)
            if not es:
                return
            if not self.val_parquet or not Path(self.val_parquet).is_file():
                return
            if state.global_step <= 0:
                return
            if state.global_step % es == 0:
                return
            self._evaluate(state)

        def _evaluate(self, state: Any) -> None:
            if not self.val_parquet or not Path(self.val_parquet).is_file():
                return
            import torch
            import torch.distributed as dist
            from torch.utils.data import DataLoader
            from tqdm.auto import tqdm

            from veomni.data.data_transform import build_data_transform
            from veomni.data.dataset import build_mapping_dataset
            from veomni.utils import helper

            args = self.trainer.args
            rank = dist.get_rank() if dist.is_initialized() else 0

            # Keep all ranks in lockstep: rank 0 runs I/O-heavy eval; others must wait here or they will
            # start the next optimizer step / hit train-end barriers first (classic DDP eval deadlock).
            if dist.is_initialized():
                dist.barrier()

            if rank != 0:
                if dist.is_initialized():
                    dist.barrier()
                self.trainer.model.train()
                return

            if not self._built:
                transform = build_data_transform(
                    args.data.data_type,
                    tokenizer=self.trainer.tokenizer,
                    chat_template=self.trainer.chat_template,
                    max_seq_len=args.data.max_seq_len,
                    text_keys=args.data.text_keys,
                )
                # Only rank 0 loads eval parquet; must not use main_process_first() (its barrier waits on all ranks).
                ds = build_mapping_dataset(
                    train_path=self.val_parquet,
                    transform=transform,
                    namespace="train",
                    distributed_sync=False,
                )
                try:
                    _nel = len(ds)
                except TypeError:
                    _nel = "?"
                helper.logger.info_rank0(f"[eval] val mapping dataset size={_nel} path={self.val_parquet}")

                def collate(batch: list) -> Any:
                    flat = []
                    for item in batch:
                        flat.append(item[0] if isinstance(item, list) else item)
                    return self.trainer.collate_fn(flat)

                self._loader = DataLoader(
                    ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
                )
                self._built = True

            from veomni.utils.constants import IGNORE_INDEX

            self.trainer.model.eval()
            total_ce_sum = 0.0
            total_ent_sum = 0.0
            n_tokens = 0
            n_batches = 0
            diffusion_loss_sum = 0.0
            n_diff_batches = 0
            hf_batch_losses: list[float] = []
            _cap = self.max_batches
            try:
                _loader_len = min(_cap, len(self._loader))  # type: ignore[arg-type]
            except Exception:
                _loader_len = None
            import sys as _sys
            import time as _time

            t0 = _time.time()
            # IMPORTANT: use inner unwrapped module so we bypass DDP `_pre_forward._sync_buffers` (a NCCL
            # collective) which would deadlock since only rank 0 enters this block.
            with _inference_inner(self.trainer.model) as inner:
                with torch.no_grad():
                    # ``leave=True``: ``leave=False`` clears the bar when eval finishes — easy to miss in logs.
                    # ``position=1``: nest under VeOmni's epoch ``trange`` (rank-0 training bar) so both stay readable.
                    bar = tqdm(
                        self._loader,
                        desc=f"[eval] CE batches (step {state.global_step})",
                        total=_loader_len,
                        leave=True,
                        position=1,
                        file=_sys.stdout,
                        mininterval=0.2,
                        dynamic_ncols=True,
                    )
                    for i, micro in enumerate(bar):
                        if i >= self.max_batches:
                            break
                        mb = micro[0] if isinstance(micro, list) else micro
                        mb = {
                            k: v.to(self.trainer.device, non_blocking=True) if torch.is_tensor(v) else v
                            for k, v in mb.items()
                        }
                        out = inner(**mb, use_cache=False)
                        if "labels" not in mb:
                            continue
                        if self.diffusion_lm:
                            if out.loss is None:
                                continue
                            diffusion_loss_sum += float(out.loss.item())
                            n_diff_batches += 1
                            n_batches += 1
                            continue
                        if getattr(out, "loss", None) is not None:
                            hf_batch_losses.append(float(out.loss.item()))
                        if out.logits is None:
                            continue
                        ce_sum, ent_sum, nt = _masked_lm_token_sums(out.logits, mb["labels"], IGNORE_INDEX)
                        if nt == 0:
                            continue
                        total_ce_sum += ce_sum
                        total_ent_sum += ent_sum
                        n_tokens += nt
                        n_batches += 1
                    bar.close()
            helper.logger.info_rank0(
                f"[eval] step={state.global_step} CE eval done in {(_time.time() - t0):.2f}s "
                f"({n_batches} batches)"
            )

            if self.diffusion_lm and n_diff_batches > 0:
                mean_dl = diffusion_loss_sum / n_diff_batches
                helper.logger.info_rank0(
                    f"[eval] step={state.global_step} diffusion_ce_mean_batch={mean_dl:.4f} "
                    f"(batches={n_diff_batches})"
                )
                try:
                    import wandb

                    if args.train.wandb.enable:
                        wandb.log({"eval/loss": mean_dl}, step=state.global_step)
                except Exception:
                    pass
            elif n_tokens > 0:
                mean_ce = total_ce_sum / n_tokens
                mean_ent = total_ent_sum / n_tokens
                ppl = math.exp(mean_ce)
                helper.logger.info_rank0(
                    f"[eval] step={state.global_step} mean_ce={mean_ce:.4f} ppl={ppl:.4f} "
                    f"mean_ent={mean_ent:.4f} (tokens={n_tokens}, batches={n_batches})"
                )
                try:
                    import wandb

                    if args.train.wandb.enable:
                        wandb.log(
                            {
                                "eval/loss": mean_ce,
                                "eval/perplexity": ppl,
                                "eval/entropy": mean_ent,
                            },
                            step=state.global_step,
                        )
                except Exception:
                    pass
            elif hf_batch_losses:
                mean_hf = sum(hf_batch_losses) / len(hf_batch_losses)
                helper.logger.info_rank0(
                    f"[eval] step={state.global_step} mean_loss={mean_hf:.4f} from model loss "
                    f"(batches={len(hf_batch_losses)}; token-level CE skipped — often logits/token mask mismatch)"
                )
                try:
                    import wandb

                    if args.train.wandb.enable:
                        wandb.log(
                            {
                                "eval/loss": mean_hf,
                                "eval/perplexity": math.exp(mean_hf),
                            },
                            step=state.global_step,
                        )
                except Exception:
                    pass
            else:
                helper.logger.warning_rank0(
                    "[eval] No loss or supervised tokens in validation batches "
                    "(check val parquet, labels, and model outputs). Sanity decode still runs below."
                )

            _eval_sanity_check(
                val_parquet_path=self.val_parquet,
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                device=self.trainer.device,
                max_new_tokens=min(512, int(args.data.max_seq_len)),
                global_step=int(state.global_step),
                diffusion_lm=self.diffusion_lm,
            )
            if dist.is_initialized():
                dist.barrier()

            self.trainer.model.train()

    return ParquetEvalCallback(trainer, val_parquet, max_batches, diffusion_lm)


def _build_embedding_eval_callback(trainer: Any, val_parquet: str | None, max_batches: int) -> Any:
    """Per-token binary keep/discard eval: reports BCE-loss + accuracy on the val parquet (rank 0 only)."""
    from veomni.trainer.callbacks.evaluate_callback import EvaluateCallback

    class EmbeddingEvalCallback(EvaluateCallback):
        def __init__(self, t: Any, vp: str | None, mb: int) -> None:
            super().__init__(t)
            self.val_parquet = vp
            self.max_batches = mb
            self._built = False
            self._loader = None

        def on_train_end(self, state: Any) -> None:
            args = self.trainer.args
            es = getattr(args.train, "eval_steps", None)
            if not es or not self.val_parquet or not Path(self.val_parquet).is_file():
                return
            if state.global_step <= 0 or state.global_step % es == 0:
                return
            self._evaluate(state)

        def _evaluate(self, state: Any) -> None:
            if not self.val_parquet or not Path(self.val_parquet).is_file():
                return
            import torch
            import torch.distributed as dist
            from torch.utils.data import DataLoader

            from veomni.data.data_transform import build_data_transform
            from veomni.data.dataset import build_mapping_dataset
            from veomni.utils import helper
            from veomni.utils.constants import IGNORE_INDEX

            args = self.trainer.args
            rank = dist.get_rank() if dist.is_initialized() else 0

            if dist.is_initialized():
                dist.barrier()

            if rank != 0:
                if dist.is_initialized():
                    dist.barrier()
                self.trainer.model.train()
                return

            if not self._built:
                transform = build_data_transform(
                    args.data.data_type,
                    tokenizer=self.trainer.tokenizer,
                    chat_template=self.trainer.chat_template,
                    max_seq_len=args.data.max_seq_len,
                    text_keys=args.data.text_keys,
                )
                ds = build_mapping_dataset(
                    train_path=self.val_parquet,
                    transform=transform,
                    namespace="train",
                    distributed_sync=False,
                )

                def collate(batch: list) -> Any:
                    flat = []
                    for item in batch:
                        flat.append(item[0] if isinstance(item, list) else item)
                    return self.trainer.collate_fn(flat)

                self._loader = DataLoader(
                    ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
                )
                self._built = True

            self.trainer.model.eval()
            total_loss = 0.0
            n_batches = 0
            n_correct = 0
            n_total = 0
            with _inference_inner(self.trainer.model) as inner:
                with torch.no_grad():
                    for i, micro in enumerate(self._loader):
                        if i >= self.max_batches:
                            break
                        mb = micro[0] if isinstance(micro, list) else micro
                        mb = {
                            k: v.to(self.trainer.device, non_blocking=True) if torch.is_tensor(v) else v
                            for k, v in mb.items()
                        }
                        out = inner(**mb, use_cache=False)
                        if out.loss is not None:
                            total_loss += float(out.loss.item())
                            n_batches += 1
                        if "labels" in mb and out.logits is not None:
                            labels = mb["labels"]
                            mask = labels.ne(IGNORE_INDEX)
                            if mask.any():
                                preds = (out.logits[mask] > 0).long()
                                n_correct += int((preds == labels[mask].long()).sum().item())
                                n_total += int(mask.sum().item())
            if n_batches > 0:
                mean_loss = total_loss / n_batches
                acc = (n_correct / n_total) if n_total > 0 else 0.0
                helper.logger.info_rank0(
                    f"[eval][embedding] step={state.global_step} bce={mean_loss:.4f} acc={acc:.4f} "
                    f"(batches={n_batches}, tokens={n_total})"
                )
                try:
                    import wandb

                    if args.train.wandb.enable:
                        wandb.log(
                            {"eval/loss": mean_loss, "eval/token_accuracy": acc},
                            step=int(state.global_step),
                        )
                except Exception:
                    pass

            if dist.is_initialized():
                dist.barrier()
            self.trainer.model.train()

    return EmbeddingEvalCallback(trainer, val_parquet, max_batches)


def _maybe_patch_diffusion_lm_collator(trainer: Any, diffusion_lm: bool) -> None:
    """Masked diffusion LMs need labels aligned with ``input_ids`` (no causal label shift in the collator)."""
    if not diffusion_lm:
        return
    from veomni.data.data_collator import MainCollator

    base_tr = trainer.base
    args = base_tr.args
    base_tr.collate_fn = MainCollator(
        pad_to_length=args.train.pad_to_length,
        seq_classification=True,
    )
    base_tr._build_dataloader()


def _maybe_patch_embedding_collator(trainer: Any, is_embedding_model: bool) -> None:
    """Per-token binary task: keep labels aligned with ``input_ids`` (no causal label shift)."""
    if not is_embedding_model:
        return
    from veomni.data.data_collator import MainCollator

    base_tr = trainer.base
    args = base_tr.args
    base_tr.collate_fn = MainCollator(
        pad_to_length=args.train.pad_to_length,
        seq_classification=True,
    )
    base_tr._build_dataloader()


def run_veomni(cfg: FinetuneConfig, train_parquet: str, eval_parquet: str | None) -> None:
    sys.path.insert(0, str(_VEOMNI_SRC))
    if cfg.include_loss_from_input:
        _register_full_loss_text_target()
    if cfg.is_embedding_model:
        _register_text_per_token_binary()

    from veomni.arguments import parser as ve_parser
    from veomni.arguments.arguments_types import VeOmniArguments
    from veomni.trainer.text_trainer import TextTrainer

    raw = _veomni_dict(cfg, train_parquet, eval_parquet)
    args = ve_parser._instantiate_recursive(VeOmniArguments, raw)
    import pyarrow.parquet as pq

    args.data.train_sample = max(1, pq.ParquetFile(train_parquet).metadata.num_rows)
    args.compute_train_steps(dataset_length=args.data.train_sample)

    trainer = TextTrainer(args)
    _maybe_patch_diffusion_lm_collator(trainer, cfg.diffusion_lm)
    _maybe_patch_embedding_collator(trainer, cfg.is_embedding_model)
    if not cfg.is_embedding_model:
        trainer.base.evaluate_callback = _build_parquet_eval_callback(
            trainer.base, eval_parquet, cfg.eval_max_batches, cfg.diffusion_lm
        )
        _attach_veomni_lm_entropy_wandb(trainer)
    else:
        trainer.base.evaluate_callback = _build_embedding_eval_callback(
            trainer.base, eval_parquet, cfg.eval_max_batches
        )
    trainer.train()

    # After train(), distributed is torn down; LOCAL_RANK still identifies the primary process under torchrun.
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        base = trainer.base
        gs = int(base.state.global_step)
        if gs > 0:
            ckpt_dir = Path(base.args.train.checkpoint.save_path) / f"global_step_{gs}"
            print(f"[finetune] Final checkpoint (DCP): {ckpt_dir.resolve()}", flush=True)
            if getattr(base.args.train.checkpoint, "save_hf_weights", False):
                print(f"[finetune] HuggingFace export: {(ckpt_dir / 'hf_ckpt').resolve()}", flush=True)
        else:
            print("[finetune] No checkpoint directory (global_step is 0).", flush=True)


def _hf_pair_tokenize_batch(cfg: FinetuneConfig, tok: Any, examples: dict[str, list]) -> dict[str, list]:
    system_msg = "You are a helpful, respectful and honest assistant."
    input_ids_l: list[list[int]] = []
    labels_l: list[list[int]] = []
    for src, tgt in zip(examples["text"], examples["target"]):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": str(src)},
            {"role": "assistant", "content": str(tgt)},
        ]
        if getattr(tok, "chat_template", None):
            full_ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors=None
            )
            prompt_ids = tok.apply_chat_template(
                messages[:2], tokenize=True, add_generation_prompt=True, return_tensors=None
            )
        else:
            prompt_txt = f"[SYSTEM]\n{system_msg}\n[USER]\n{src}\n[ASSISTANT]\n"
            full_txt = prompt_txt + str(tgt)
            full_ids = tok.encode(full_txt, add_special_tokens=True)
            prompt_ids = tok.encode(prompt_txt, add_special_tokens=True)
        if cfg.include_loss_from_input:
            labels = list(full_ids)
        else:
            labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
        max_len = cfg.max_seq_len
        if len(full_ids) > max_len:
            full_ids = full_ids[-max_len:]
            labels = labels[-max_len:]
        input_ids_l.append(full_ids)
        labels_l.append(labels)
    return {"input_ids": input_ids_l, "labels": labels_l}


@dataclass
class _PackCollator:
    cfg: FinetuneConfig
    tokenizer: Any
    max_len: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not self.cfg.packing:
            input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            pad_id = int(self.tokenizer.pad_token_id or 0)
            return {
                "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
                "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [torch.ones_like(x) for x in input_ids], batch_first=True, padding_value=0
                ),
            }
        pieces_ids: list[int] = []
        pieces_labels: list[int] = []
        eos = int(self.tokenizer.eos_token_id or 0)
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            if pieces_ids:
                pieces_ids.append(eos)
                pieces_labels.append(-100)
            pieces_ids.extend(ids)
            pieces_labels.extend(lab)
            if len(pieces_ids) >= self.max_len:
                break
        pieces_ids = pieces_ids[: self.max_len]
        pieces_labels = pieces_labels[: self.max_len]
        t_ids = torch.tensor([pieces_ids], dtype=torch.long)
        t_lab = torch.tensor([pieces_labels], dtype=torch.long)
        return {
            "input_ids": t_ids,
            "labels": t_lab,
            "attention_mask": torch.ones_like(t_ids),
        }


def run_huggingface(
    cfg: FinetuneConfig,
    train_rows: list[dict],
    val_rows: list[dict],
    val_parquet_path: str | None = None,
) -> None:
    import os

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    if not cfg.init_model_path:
        raise SystemExit("init_model_path is required for huggingface model_type")
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path or cfg.init_model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.init_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    try:
        mcls = f"{model.__class__.__module__}.{model.__class__.__name__}"
        mcfg = getattr(model, "config", None)
        print(
            f"[model] training_class={mcls} model_type={getattr(mcfg, 'model_type', None)} "
            f"architectures={getattr(mcfg, 'architectures', None)}",
            flush=True,
        )
    except Exception:
        pass

    use_gradient_checkpointing = bool(cfg.gradient_checkpointing)
    if use_gradient_checkpointing and not getattr(model, "_supports_gradient_checkpointing", True):
        warnings.warn(
            f"{type(model).__name__} does not support gradient checkpointing; "
            "using gradient_checkpointing=False (set gradient_checkpointing: false in YAML to silence)."
        )
        use_gradient_checkpointing = False

    def tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
        return _hf_pair_tokenize_batch(cfg, tok, examples)

    ds_train = Dataset.from_list(train_rows).map(
        tokenize_batch,
        batched=True,
        remove_columns=["text", "target"],
    )
    ds_val = Dataset.from_list(val_rows).map(
        tokenize_batch,
        batched=True,
        remove_columns=["text", "target"],
    )

    collator = _PackCollator(cfg, tok, cfg.max_seq_len)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    gas = max(1, cfg.global_batch_size // max(1, world) // max(1, cfg.micro_batch_size))

    eval_steps = cfg.eval_every_steps if cfg.eval_every_steps > 0 and len(val_rows) else None
    targs = TrainingArguments(
        output_dir=cfg.checkpoint_output_dir,
        per_device_train_batch_size=max(1, cfg.micro_batch_size),
        gradient_accumulation_steps=gas,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_steps=int(cfg.max_steps) if cfg.max_steps is not None else -1,
        num_train_epochs=cfg.num_train_epochs if cfg.max_steps is None else 1.0,
        logging_steps=10,
        save_steps=cfg.save_every_steps,
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        report_to=["wandb"],
        run_name=cfg.wandb_run_name or Path(cfg.checkpoint_output_dir).name,
        bf16=True,
        gradient_checkpointing=use_gradient_checkpointing,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine" if cfg.lr_decay_style == "cosine" else "linear",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val if val_rows else None,
        data_collator=collator,
        tokenizer=tok,
    )

    if val_parquet_path and eval_steps:

        class _HFEvalSanity(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None):
                if args.world_size > 1 and getattr(args, "process_index", 0) != 0:
                    return
                try:
                    import wandb

                    if metrics and "eval_loss" in metrics:
                        el = float(metrics["eval_loss"])
                        wandb.log({"eval/perplexity": math.exp(el)}, step=int(state.global_step))
                except Exception:
                    pass
                dev = next(trainer.model.parameters()).device
                _eval_sanity_check(
                    val_parquet_path=val_parquet_path,
                    model=trainer.model,
                    tokenizer=tok,
                    device=dev,
                    max_new_tokens=min(512, int(cfg.max_seq_len)),
                    global_step=int(state.global_step),
                    diffusion_lm=False,
                )

        trainer.add_callback(_HFEvalSanity())

    trainer.train()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: finetune.py path/to/config.yaml")
    yaml_path = Path(sys.argv[1]).expanduser().resolve()
    raw = _load_yaml(yaml_path)
    cfg = FinetuneConfig.from_dict(raw, str(yaml_path))
    cfg.config_path = str(yaml_path)
    _finalize_resume_checkpoint_path(cfg)

    _maybe_reexec_torchrun(cfg)

    if cfg.is_embedding_model:
        train_rows, val_rows = build_embedding_pair_rows(cfg)
    else:
        train_rows, val_rows = build_pair_rows(cfg)
    out_root = Path(cfg.checkpoint_output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_pq = out_root / "prepared_train_pairs.parquet"
    val_pq = out_root / "prepared_val_pairs.parquet"

    # Under torchrun every rank executes main(); parallel writes to the same parquet paths corrupt files.
    import time

    import pyarrow.parquet as pq

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if local_rank == 0:
            _write_pair_parquet(train_rows, train_pq)
            _write_pair_parquet(val_rows, val_pq)
        else:
            deadline = time.time() + 600.0
            while time.time() < deadline:
                if train_pq.is_file() and val_pq.is_file():
                    try:
                        pq.ParquetFile(str(train_pq))
                        pq.ParquetFile(str(val_pq))
                        break
                    except Exception:
                        pass
                time.sleep(0.05)
            else:
                raise SystemExit(
                    f"[finetune] rank {local_rank}: timed out waiting for prepared parquets under {out_root}"
                )
    else:
        _write_pair_parquet(train_rows, train_pq)
        _write_pair_parquet(val_rows, val_pq)

    if cfg.model_type == "veomni_supported":
        eval_pq = str(val_pq) if val_rows and cfg.eval_every_steps > 0 else None
        run_veomni(cfg, str(train_pq), eval_pq)
    elif cfg.model_type == "huggingface":
        run_huggingface(cfg, train_rows, val_rows, str(val_pq))
    elif cfg.model_type == "custom_local":
        raise SystemExit(
            "model_type custom_local has been removed. Register your model under "
            "finetuning/veomni/veomni/models/transformers/ (see VeOmni MODEL_CONFIG_REGISTRY / MODELING_REGISTRY) "
            "and use model_type veomni_supported with veomni_overrides.model.config_path pointing at your config.json."
        )
    else:
        raise SystemExit(f"Unknown model_type: {cfg.model_type}")


if __name__ == "__main__":
    main()
