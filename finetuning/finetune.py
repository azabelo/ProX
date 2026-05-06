#!/usr/bin/env python3
"""
Finetune on (input chunk, OpenRouter output chunk) pairs from parquet dirs.

Usage (multi-GPU, VeOmni path):
  torchrun --standalone --nproc_per_node=8 finetuning/finetune.py path/to/your_finetune.yaml

VeOmni loads ``torch.distributed`` even for one GPU. When ``LOCAL_RANK`` is unset, this script either
re-execs via ``torch.distributed.run`` (``num_gpus`` > 1) or sets single-process ``RANK``/``WORLD_SIZE`` env vars.

YAML schema (top-level keys):
  - model_arch: str (informational; also merged into ``veomni.model.foundation`` when provided)
  - model_type: veomni_supported | huggingface | custom_local
  - init_model_path: HuggingFace hub id or local path for base weights (maps to VeOmni ``model.model_path``)
  - resume_checkpoint_path: optional VeOmni DCP dir (e.g. ``.../checkpoints/global_step_1000``) for full resume
  - inputs_dir_path: directory of source shards (``.parquet`` or ``*.jsonl.zst``)
  - outputs_dir_path: OpenRouter output directory (copied YAML + ``*_openrouter.parquet``)
  - dataset_type: fineweb | dclm | redpajama | redpajama-v2 | passthrough (row text adapter, same as OpenRouter runner)
  - chunk_size: int (character chunks; must match OpenRouter run — mismatched input vs output chunk counts drop the whole doc)
  - is_code: bool — read ``programs_delimited`` vs ``text`` from output rows
  - packing: bool — VeOmni: enables ``dyn_bsz`` token packing; HF: packs examples in the collator up to ``max_seq_len``
  - include_loss_from_input: bool default false — false: CE only on assistant/output tokens (VeOmni ``text_target``)
  - num_gpus: int — used only for auto ``torchrun`` re-exec
  - num_validation_documents: int — documents held out entirely for validation (by ``source_parquet`` + ``source_row_index``)
  - max_seq_len, max_steps, global_batch_size, micro_batch_size, learning_rate, weight_decay, warmup_ratio,
    lr_decay_style, max_grad_norm, save_every_steps, eval_every_steps, checkpoint_output_dir, seed, wandb_project,
    wandb_run_name, gradient_checkpointing, tokenizer_path (optional override), custom_model_factory (for custom_local:
    ``"module.path:callable"`` returning a ``torch.nn.Module``)

Optional nested ``veomni:`` dict is deep-merged into the constructed VeOmni YAML dict before dataclass load.

``custom_local`` is experimental: you must set ``custom_model_factory``; training uses a minimal AdamW loop with
the same paired parquet dataloader as the HF path (single-process unless you extend it).
"""
from __future__ import annotations

import importlib
import math
import os
import random
import subprocess
import sys
import types
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    custom_model_factory: str | None = None
    veomni_overrides: dict[str, Any] = field(default_factory=dict)
    # If True, append tokenizer eos to each target string when encode(target) does not already end with eos_token_id
    # so the last supervised token in chat/text_target loss can be EOS (helps learned stopping in generate()).
    ensure_eos_on_target: bool = True

    @staticmethod
    def from_dict(raw: dict[str, Any], config_path: str) -> "FinetuneConfig":
        def g(key: str, default: Any = None) -> Any:
            return raw[key] if key in raw else default

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
            custom_model_factory=g("custom_model_factory", None),
            veomni_overrides=dict(g("veomni", {}) or {}),
            ensure_eos_on_target=_bool(g("ensure_eos_on_target", True)),
        )


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

    if "text_target_all" in DATA_TRANSFORM_REGISTRY:
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
    # When training a custom_local-style model through VeOmni, we still want the
    # factory to be able to read YAML `custom_arch` from the original finetune config.
    if cfg.custom_model_factory:
        foundation["finetune_config_path"] = str(cfg.config_path)

    dyn_bsz = bool(cfg.packing)
    data_type = "text_target_all" if cfg.include_loss_from_input else "text_target"

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
    if cfg.custom_model_factory:
        base["model"]["custom_model_factory"] = cfg.custom_model_factory
    return _deep_update(base, dict(cfg.veomni_overrides))


def _import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise SystemExit("custom_model_factory must look like 'package.module:fn'")
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if not callable(fn):
        raise SystemExit(f"{spec!r} is not callable")
    return fn


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
) -> None:
    """Print first row input/target from val parquet and a greedy ``generate`` prediction."""
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
    gen_m = _unwrap_for_generate(model)
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else tok.eos_token_id

    pred = ""
    try:
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
            gen_out = gen_m.generate(
                ids,
                attention_mask=attn,
                max_new_tokens=cap,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tok.eos_token_id,
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


def _build_parquet_eval_callback(trainer: Any, val_parquet: str | None, max_batches: int) -> Any:
    """Mean CE on a held-out pair parquet (rank 0 only)."""

    from veomni.trainer.callbacks.evaluate_callback import EvaluateCallback

    class ParquetEvalCallback(EvaluateCallback):
        def __init__(self, t: Any, vp: str | None, mb: int) -> None:
            super().__init__(t)
            self.val_parquet = vp
            self.max_batches = mb
            self._built = False
            self._loader = None

        def _evaluate(self, state: Any) -> None:
            if not self.val_parquet or not Path(self.val_parquet).is_file():
                return
            import torch
            from torch.utils.data import DataLoader
            from veomni.data.data_transform import build_data_transform
            from veomni.data.dataset import build_mapping_dataset
            from veomni.utils import helper

            args = self.trainer.args
            if args.train.global_rank != 0:
                return

            if not self._built:
                transform = build_data_transform(
                    args.data.data_type,
                    tokenizer=self.trainer.tokenizer,
                    chat_template=self.trainer.chat_template,
                    max_seq_len=args.data.max_seq_len,
                    text_keys=args.data.text_keys,
                )
                ds = build_mapping_dataset(train_path=self.val_parquet, transform=transform, namespace="train")

                def collate(batch: list) -> Any:
                    flat = []
                    for item in batch:
                        flat.append(item[0] if isinstance(item, list) else item)
                    return self.trainer.collate_fn(flat)

                self._loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
                self._built = True

            from veomni.utils.constants import IGNORE_INDEX

            self.trainer.model.eval()
            total_ce_sum = 0.0
            total_ent_sum = 0.0
            n_tokens = 0
            n_batches = 0
            with torch.no_grad():
                for i, micro in enumerate(self._loader):
                    if i >= self.max_batches:
                        break
                    mb = micro[0] if isinstance(micro, list) else micro
                    mb = {
                        k: v.to(self.trainer.device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in mb.items()
                    }
                    out = self.trainer.model(**mb, use_cache=False)
                    if out.logits is None or "labels" not in mb:
                        continue
                    ce_sum, ent_sum, nt = _masked_lm_token_sums(out.logits, mb["labels"], IGNORE_INDEX)
                    if nt == 0:
                        continue
                    total_ce_sum += ce_sum
                    total_ent_sum += ent_sum
                    n_tokens += nt
                    n_batches += 1

            if n_tokens > 0:
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

            _eval_sanity_check(
                val_parquet_path=self.val_parquet,
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                device=self.trainer.device,
                max_new_tokens=min(512, int(args.data.max_seq_len)),
                global_step=int(state.global_step),
            )
            self.trainer.model.train()

    return ParquetEvalCallback(trainer, val_parquet, max_batches)


def run_veomni(cfg: FinetuneConfig, train_parquet: str, eval_parquet: str | None) -> None:
    sys.path.insert(0, str(_VEOMNI_SRC))
    if cfg.include_loss_from_input:
        _register_full_loss_text_target()

    from veomni.arguments import parser as ve_parser
    from veomni.arguments.arguments_types import VeOmniArguments
    from veomni.trainer.text_trainer import TextTrainer

    raw = _veomni_dict(cfg, train_parquet, eval_parquet)
    args = ve_parser._instantiate_recursive(VeOmniArguments, raw)
    import pyarrow.parquet as pq

    args.data.train_sample = max(1, pq.ParquetFile(train_parquet).metadata.num_rows)
    args.compute_train_steps(dataset_length=args.data.train_sample)

    trainer = TextTrainer(args)
    trainer.base.evaluate_callback = _build_parquet_eval_callback(trainer.base, eval_parquet, cfg.eval_max_batches)
    _attach_veomni_lm_entropy_wandb(trainer)
    trainer.train()


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
                )

        trainer.add_callback(_HFEvalSanity())

    if cfg.resume_checkpoint_path and Path(cfg.resume_checkpoint_path).is_dir():
        trainer.train(resume_from_checkpoint=cfg.resume_checkpoint_path)
    else:
        trainer.train()


def run_custom_local(cfg: FinetuneConfig, train_rows: list[dict], val_rows: list[dict]) -> None:
    """
    Single-GPU minimal loop. ``custom_model_factory`` must be ``module:fn`` where
    ``fn(config_dict)`` returns a ``torch.nn.Module`` whose ``forward`` accepts
    ``input_ids``, ``attention_mask``, and ``labels`` like HuggingFace causal LMs.
    """
    if not cfg.custom_model_factory:
        raise SystemExit("custom_model_factory (e.g. 'mymod.build:build_model') is required for custom_local")
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    factory = _import_callable(cfg.custom_model_factory)
    tok_path = cfg.tokenizer_path or cfg.init_model_path
    if not tok_path:
        raise SystemExit("tokenizer_path or init_model_path is required for custom_local tokenization")
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
        return _hf_pair_tokenize_batch(cfg, tok, examples)

    ds_train = Dataset.from_list(train_rows).map(
        tokenize_batch,
        batched=True,
        remove_columns=["text", "target"],
    )
    collator = _PackCollator(cfg, tok, cfg.max_seq_len)
    loader = DataLoader(
        ds_train,
        batch_size=max(1, cfg.micro_batch_size),
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    model = factory(asdict(cfg))
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        mcls = f"{model.__class__.__module__}.{model.__class__.__name__}"
        mc = getattr(model, "cfg", None)
        print(f"[model] training_class={mcls} custom_cfg={mc}", flush=True)
    except Exception:
        pass
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    device = next(model.parameters()).device
    model.train()

    # Optional W&B logging (custom_local doesn't use HF Trainer / VeOmni callbacks).
    wandb = None
    wandb_active = False
    try:
        import os

        if os.environ.get("WANDB_MODE", "").lower() not in {"disabled", "offline"}:
            import wandb as _wandb  # type: ignore

            wandb = _wandb
            os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
            run_name = cfg.wandb_run_name or Path(cfg.checkpoint_output_dir).name
            wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))
            wandb_active = True
    except Exception:
        wandb = None
        wandb_active = False

    step = 0
    max_steps = cfg.max_steps if cfg.max_steps is not None else len(loader) * cfg.num_train_epochs
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            if torch.is_tensor(out):
                loss = out
            elif isinstance(out, tuple) and torch.is_tensor(out[0]):
                loss = out[0]
            else:
                loss = getattr(out, "loss", None)
            if loss is None:
                raise RuntimeError("custom model forward must return a tensor loss or object with .loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"[custom_local] step={step} loss={float(loss.item()):.4f}", flush=True)
                if wandb_active and wandb is not None:
                    try:
                        wandb.log(
                            {
                                "train/loss": float(loss.item()),
                                "train/grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                                "train/lr": float(opt.param_groups[0]["lr"]),
                            },
                            step=int(step),
                        )
                    except Exception:
                        pass
        if cfg.max_steps is not None:
            break
    if wandb_active and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: finetune.py path/to/config.yaml")
    yaml_path = Path(sys.argv[1]).expanduser().resolve()
    raw = _load_yaml(yaml_path)
    cfg = FinetuneConfig.from_dict(raw, str(yaml_path))
    cfg.config_path = str(yaml_path)

    _maybe_reexec_torchrun(cfg)

    train_rows, val_rows = build_pair_rows(cfg)
    out_root = Path(cfg.checkpoint_output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_pq = out_root / "prepared_train_pairs.parquet"
    val_pq = out_root / "prepared_val_pairs.parquet"
    _write_pair_parquet(train_rows, train_pq)
    _write_pair_parquet(val_rows, val_pq)

    if cfg.model_type == "veomni_supported":
        eval_pq = str(val_pq) if val_rows and cfg.eval_every_steps > 0 else None
        run_veomni(cfg, str(train_pq), eval_pq)
    elif cfg.model_type == "huggingface":
        run_huggingface(cfg, train_rows, val_rows, str(val_pq))
    elif cfg.model_type == "custom_local":
        run_custom_local(cfg, train_rows, val_rows)
    else:
        raise SystemExit(f"Unknown model_type: {cfg.model_type}")


if __name__ == "__main__":
    main()
