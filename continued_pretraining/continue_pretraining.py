#!/usr/bin/env python3
"""
Continued pretraining (next-token prediction) on raw documents from parquet / jsonl.zst shards.

Usage (multi-GPU, VeOmni path; auto re-execs via torchrun when ``num_gpus`` > 1):
  python continued_pretraining/continue_pretraining.py continued_pretraining/example_continue_pretraining.yaml

What this does
  1. Reads raw shards under ``inputs_dir_path`` (``dclm`` / ``fineweb`` / ``redpajama`` / … adapters).
  2. Holds out ``num_validation_documents`` docs for eval using **reservoir sampling** (uniform over the
     streamed corpus prefix; seeded by ``seed``). Documents evicted from the reservoir join the train stream.
  3. Streams training rows to ``prepared_train_text.parquet`` until the estimated token budget is met.
  4. Runs VeOmni ``plaintext`` LM training with optional dyn_bsz packing.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess  # noqa: F401
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

try:
    import site

    _USER_SITE = site.getusersitepackages()
    if isinstance(_USER_SITE, str):
        sys.path = [p for p in sys.path if Path(p).resolve() != Path(_USER_SITE).resolve()]
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openrouter_data_generation.run_openrouter_chunked import (  # noqa: E402
    get_row_text_fn,
    iter_shard_rows,
    sorted_shard_paths,
)

_VEOMNI_SRC = Path(__file__).resolve().parent / "veomni"


def _ensure_single_process_dist_env() -> None:
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


def _maybe_reexec_torchrun(cfg: "ContinuedPretrainingConfig") -> None:
    if os.environ.get("LOCAL_RANK") is not None:
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
        print("[continue_pretraining] re-exec:", " ".join(cmd), flush=True)
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


@dataclass
class ContinuedPretrainingConfig:
    config_path: str
    model_arch: str = "auto"
    init_model_path: str = "Qwen/Qwen2.5-0.5B"
    tokenizer_path: str | None = None
    resume_checkpoint_path: str | None = None
    inputs_dir_path: str = ""
    dataset_type: str = "dclm"
    num_validation_documents: int = 64
    max_input_documents: int | None = None
    prepared_data_dir: str | None = None
    train_token_oversample: float = 1.0
    chars_per_token: float = 4.0
    num_gpus: int = 8
    max_seq_len: int = 2048
    max_training_tokens: int = 1_000_000_000
    max_steps: int | None = None
    num_train_epochs: int = 1
    global_batch_size: int = 64
    micro_batch_size: int = 8
    learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    lr_decay_style: str = "cosine"
    max_grad_norm: float = 1.0
    save_every_steps: int = 500
    eval_every_steps: int = 500
    eval_max_batches: int = 32
    checkpoint_output_dir: str = "./continued_pretraining_outputs/run1"
    seed: int = 42
    wandb_project: str = "prox-continue-pretraining"
    wandb_run_name: str | None = None
    gradient_checkpointing: bool = False
    packing: bool = True
    fsdp_mode: str = "ddp"
    attn_implementation: str = "auto"
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 4
    veomni_overrides: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict[str, Any], config_path: str) -> "ContinuedPretrainingConfig":
        def g(key: str, default: Any = None) -> Any:
            return raw[key] if key in raw else default

        def optional_dir(v: Any) -> str | None:
            """YAML often uses unquoted ``none`` which parses as the string ``'none'`` — treat like null."""
            if v is None:
                return None
            if isinstance(v, str):
                s = v.strip()
                if not s or s.lower() in ("none", "null", "~"):
                    return None
                return s
            return str(v)

        return ContinuedPretrainingConfig(
            config_path=config_path,
            model_arch=str(g("model_arch", "auto")),
            init_model_path=str(g("init_model_path", g("model_path", ""))),
            tokenizer_path=g("tokenizer_path"),
            resume_checkpoint_path=g("resume_checkpoint_path", g("resume_checkpoint", None)),
            inputs_dir_path=str(g("inputs_dir_path", "")),
            dataset_type=str(g("dataset_type", "dclm")),
            num_validation_documents=int(g("num_validation_documents", 64)),
            max_input_documents=g("max_input_documents", None),
            prepared_data_dir=optional_dir(g("prepared_data_dir", None)),
            train_token_oversample=float(g("train_token_oversample", 1.0)),
            chars_per_token=float(g("chars_per_token", 4.0)),
            num_gpus=int(g("num_gpus", 1)),
            max_seq_len=int(g("max_seq_len", 2048)),
            max_training_tokens=int(g("max_training_tokens", 1_000_000_000)),
            max_steps=g("max_steps", None),
            num_train_epochs=int(g("num_train_epochs", 1)),
            global_batch_size=int(g("global_batch_size", 64)),
            micro_batch_size=int(g("micro_batch_size", 8)),
            learning_rate=float(g("learning_rate", 3e-5)),
            weight_decay=float(g("weight_decay", 0.1)),
            warmup_ratio=float(g("warmup_ratio", 0.01)),
            lr_decay_style=str(g("lr_decay_style", "cosine")),
            max_grad_norm=float(g("max_grad_norm", 1.0)),
            save_every_steps=int(g("save_every_steps", 500)),
            eval_every_steps=int(g("eval_every_steps", 500)),
            eval_max_batches=int(g("eval_max_batches", 32)),
            checkpoint_output_dir=str(g("checkpoint_output_dir", "./continued_pretraining_outputs/run1")),
            seed=int(g("seed", 42)),
            wandb_project=str(g("wandb_project", "prox-continue-pretraining")),
            wandb_run_name=g("wandb_run_name", None),
            gradient_checkpointing=_bool(g("gradient_checkpointing", False)),
            packing=_bool(g("packing", True)),
            fsdp_mode=str(g("fsdp_mode", "ddp")),
            attn_implementation=str(g("attn_implementation", "auto")),
            dataloader_num_workers=int(g("dataloader_num_workers", 8)),
            dataloader_prefetch_factor=int(g("dataloader_prefetch_factor", 4)),
            veomni_overrides=dict(g("veomni_overrides", g("veomni", {})) or {}),
        )


def _finalize_resume_checkpoint_path(cfg: ContinuedPretrainingConfig) -> None:
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
            p = cwd_p
    meta = p / ".metadata"
    if not p.is_dir():
        alt = ""
        if not rel.is_absolute():
            other = (yaml_p if p == cwd_p else cwd_p).resolve()
            if other != p:
                alt = f"\n  Also checked: {other}"
        raise SystemExit(
            f"[continue_pretraining] resume_checkpoint_path is not a directory:\n  {p}\n"
            f"  (YAML had {raw!r}; relative paths try cwd={Path.cwd().resolve()!s} then {cfg_yaml.parent!s}){alt}"
        )
    if not meta.is_file():
        raise SystemExit(
            f"[continue_pretraining] resume_checkpoint_path is missing .metadata (not a VeOmni DCP checkpoint):\n"
            f"  {meta}\n"
            f"  List checkpoints: ls {p.parent.resolve()!s}/global_step_* 2>/dev/null || true"
        )
    cfg.resume_checkpoint_path = str(p)


def _iter_documents(
    inputs_dir: Path, dataset_type: str
) -> Iterator[tuple[str, int, str]]:
    row_text_fn = get_row_text_fn(dataset_type)
    shards = sorted_shard_paths(inputs_dir)
    if not shards:
        raise SystemExit(f"No shards found under inputs_dir_path={inputs_dir}")
    for shard in shards:
        idx = 0
        for row in iter_shard_rows(shard):
            text = row_text_fn(row) if row else ""
            if text and text.strip():
                yield (shard.name, idx, text)
            idx += 1


def build_pretraining_parquets(
    cfg: ContinuedPretrainingConfig,
    out_root: Path,
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Stream shards to train (and optional val) parquets. Validation docs: reservoir sample (seed ``cfg.seed``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    inputs_dir = Path(cfg.inputs_dir_path).expanduser().resolve()
    if not inputs_dir.is_dir():
        raise SystemExit(f"inputs_dir_path is not a directory: {inputs_dir}")

    train_pq = out_root / "prepared_train_text.parquet"
    val_pq = out_root / "prepared_val_text.parquet"
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("source_shard", pa.string()),
            pa.field("source_row_index", pa.int64()),
        ]
    )

    target_train_tokens = max(1, int(cfg.train_token_oversample * cfg.max_training_tokens))
    target_train_chars = int(target_train_tokens * cfg.chars_per_token)

    n_val_target = max(0, int(cfg.num_validation_documents))
    rng = random.Random(int(cfg.seed))
    train_chars = 0
    train_docs = 0
    train_buf: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    BATCH = 1024

    def _flush_train(writer: Any) -> None:
        nonlocal train_buf
        if not train_buf:
            return
        tbl = pa.Table.from_pylist(train_buf, schema=schema)
        writer.write_table(tbl)
        train_buf = []

    train_pq.parent.mkdir(parents=True, exist_ok=True)
    val_desc = (
        f"{n_val_target} val docs (reservoir sampling, seed={cfg.seed})"
        if n_val_target
        else "no val holdout"
    )
    print(
        f"[continue_pretraining] streaming raw shards -> {train_pq.name} (target ~{target_train_tokens:,} tokens, "
        f"~{target_train_chars:,} chars; {val_desc}); inputs_dir={inputs_dir}",
        flush=True,
    )
    t0 = time.time()
    last_log = t0
    stream_i = 0

    def _maybe_log_progress() -> None:
        nonlocal last_log
        now = time.time()
        if now - last_log <= 5.0:
            return
        est_tokens = int(train_chars / max(1.0, cfg.chars_per_token))
        rate = est_tokens / max(1e-6, now - t0)
        print(
            f"[continue_pretraining]   prepared {train_docs:,} train docs / {est_tokens:,} est tokens "
            f"({rate / 1e6:.2f}M tok/s)",
            flush=True,
        )
        last_log = now

    def _emit_train(writer: Any, row: dict[str, Any]) -> bool:
        """Append one training document. Return True to stop outer iteration."""
        nonlocal train_chars, train_docs
        train_buf.append(row)
        train_chars += len(row["text"])
        train_docs += 1
        if len(train_buf) >= BATCH:
            _flush_train(writer)
        _maybe_log_progress()
        if cfg.max_input_documents and train_docs >= int(cfg.max_input_documents):
            print(
                f"[continue_pretraining] reached max_input_documents={cfg.max_input_documents}; stopping read.",
                flush=True,
            )
            return True
        if train_chars >= target_train_chars:
            print(
                f"[continue_pretraining] reached target ~{target_train_tokens:,} train tokens "
                f"(estimated from {train_chars:,} chars / {cfg.chars_per_token:.1f}); stopping read.",
                flush=True,
            )
            return True
        return False

    with pq.ParquetWriter(str(train_pq), schema=schema) as writer:
        for shard_name, row_idx, text in _iter_documents(inputs_dir, cfg.dataset_type):
            stream_i += 1
            row = {"text": text, "source_shard": shard_name, "source_row_index": int(row_idx)}

            if n_val_target == 0:
                if _emit_train(writer, row):
                    break
                continue

            if len(val_rows) < n_val_target:
                val_rows.append(row)
                continue

            j = rng.randint(1, stream_i)
            if j <= n_val_target:
                old = val_rows[j - 1]
                val_rows[j - 1] = row
                if _emit_train(writer, old):
                    break
            else:
                if _emit_train(writer, row):
                    break

        _flush_train(writer)

    val_docs = len(val_rows)
    if n_val_target > 0 and val_docs == 0:
        raise SystemExit(
            f"No validation rows produced (num_validation_documents={n_val_target}). "
            f"Increase num_validation_documents or check inputs_dir_path."
        )
    if train_docs == 0:
        raise SystemExit("No training rows produced; check inputs_dir_path and dataset_type.")

    if n_val_target > 0:
        val_table = pa.Table.from_pylist(val_rows, schema=schema)
        pq.write_table(val_table, str(val_pq))

    est_tokens = int(train_chars / max(1.0, cfg.chars_per_token))
    val_line = f"  val:   {val_docs:,} docs -> {val_pq}" if n_val_target > 0 else "  val:   (none)"
    print(
        f"[continue_pretraining] prepared parquets in {time.time() - t0:.1f}s\n"
        f"  train: {train_docs:,} docs, {train_chars:,} chars, ~{est_tokens:,} est tokens -> {train_pq}\n"
        f"{val_line}",
        flush=True,
    )
    stats = {
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_chars": train_chars,
        "estimated_train_tokens": est_tokens,
    }
    return train_pq, val_pq if n_val_target > 0 else None, stats


def _deep_update(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def _flash_attn_importable() -> bool:
    if _bool(os.environ.get("CONTINUED_PT_DISABLE_FLASH", ""), default=False):
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _infer_attn_from_yaml_only(cfg: ContinuedPretrainingConfig) -> str:
    raw = (getattr(cfg, "attn_implementation", "auto") or "auto").strip().lower()
    if raw in ("sdpa", "eager"):
        return "sdpa"
    if raw in ("flash_attention_2", "flash_attn", "flash"):
        if not _flash_attn_importable():
            raise SystemExit(
                f"attn_implementation={cfg.attn_implementation!r} requires a working ``flash_attn`` install "
                f"(``import flash_attn``). Install flash-attn, or set attn_implementation: auto or sdpa."
            )
        return "flash_attention_2"
    if raw != "auto":
        raise SystemExit(
            f"attn_implementation must be one of: auto, sdpa, flash_attention_2; got {cfg.attn_implementation!r}"
        )
    return "flash_attention_2" if _flash_attn_importable() else "sdpa"


def _effective_attn_implementation(cfg: ContinuedPretrainingConfig) -> str:
    if isinstance(cfg.veomni_overrides, dict):
        m = cfg.veomni_overrides.get("model", {})
        if isinstance(m, dict):
            oi = m.get("ops_implementation") or {}
            if isinstance(oi, dict):
                ai = oi.get("attn_implementation")
                if ai:
                    return str(ai).strip().lower()
    return _infer_attn_from_yaml_only(cfg)


def _auto_adjust_packing_for_sdpa(cfg: ContinuedPretrainingConfig) -> None:
    if not cfg.packing:
        return
    attn = _effective_attn_implementation(cfg)
    if attn in ("flash_attention_2", "flash_attention_3"):
        return
    packed_len = int(cfg.micro_batch_size) * int(cfg.max_seq_len)
    if packed_len <= 4096:
        return
    if _bool(os.environ.get("CONTINUED_PT_FORCE_PACKING", ""), default=False):
        print(
            "[continue_pretraining][warn] CONTINUED_PT_FORCE_PACKING=1: keeping packing=true under SDPA — "
            "likely CUDA OOM; install flash-attn or lower micro_batch_size.",
            flush=True,
        )
        return
    print(
        f"[continue_pretraining] flash_attention_2 not in use but packing would create packed length {packed_len} "
        f"(micro_batch_size × max_seq_len) — SDPA tends to OOM. Disabling packing (fixed rectangular micro-batching).\n"
        f"  To use packing again: install flash-attn and set attn_implementation: auto or flash_attention_2.",
        flush=True,
    )
    cfg.packing = False


def _validate_batch_geometry(cfg: ContinuedPretrainingConfig) -> None:
    denom = int(cfg.micro_batch_size) * int(cfg.num_gpus)
    if int(cfg.global_batch_size) % denom != 0:
        raise SystemExit(
            f"global_batch_size={cfg.global_batch_size} must be divisible by "
            f"(micro_batch_size × num_gpus)={denom} "
            f"(got remainder {cfg.global_batch_size % denom})."
        )


def _enable_torch_throughput_knobs() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass


def _veomni_dict(
    cfg: ContinuedPretrainingConfig,
    train_parquet: str,
    eval_parquet: str | None,
    estimated_train_tokens: int,
    derived_max_steps: int,
) -> dict[str, Any]:
    foundation: dict[str, str] = {}
    if cfg.model_arch and cfg.model_arch != "auto":
        foundation["architecture"] = cfg.model_arch

    nw = max(0, int(cfg.dataloader_num_workers))
    pf: int | None
    if nw > 0:
        pf = max(2, int(cfg.dataloader_prefetch_factor))
    else:
        pf = None

    base_attn = _infer_attn_from_yaml_only(cfg)

    base: dict[str, Any] = {
        "model": {
            "model_path": cfg.init_model_path,
            "tokenizer_path": cfg.tokenizer_path or cfg.init_model_path,
            "foundation": foundation,
            "ops_implementation": {"attn_implementation": base_attn},
        },
        "data": {
            "train_path": train_parquet,
            "eval_path": eval_parquet,
            "datasets_type": "mapping",
            "data_type": "plaintext",
            "text_keys": "text",
            "max_seq_len": cfg.max_seq_len,
            "train_size": max(1, int(estimated_train_tokens)),
            "train_sample": 1_000_000,
            "dataloader": {
                "type": "native",
                "num_workers": nw,
                "prefetch_factor": pf,
                "drop_last": True,
                "pin_memory": True,
            },
        },
        "train": {
            "dyn_bsz": bool(cfg.packing),
            "micro_batch_size": cfg.micro_batch_size,
            "global_batch_size": cfg.global_batch_size,
            "num_train_epochs": cfg.num_train_epochs,
            "max_steps": derived_max_steps,
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
                "save_hf_weights": True,
                "hf_save_steps": 0,
                "hf_save_epochs": 0,
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
                    "fsdp_mode": cfg.fsdp_mode,
                    "full_shard": cfg.fsdp_mode != "ddp",
                    "offload": False,
                    "mixed_precision": {"enable": cfg.fsdp_mode != "ddp"},
                },
            },
        },
    }
    return _deep_update(base, dict(cfg.veomni_overrides))


def _unwrap_for_generate(model: Any) -> Any:
    m = model
    if hasattr(m, "module"):
        m = m.module
    while hasattr(m, "_fsdp_wrapped_module"):
        m = m._fsdp_wrapped_module
    return m


@contextmanager
def _inference_inner(model: Any):
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


def _masked_lm_token_sums(
    logits: Any, labels: Any, ignore_index: int
) -> tuple[float, float, int]:
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


def _aligned_lm_token_sums(
    logits: Any, labels: Any, ignore_index: int
) -> tuple[float, float, int]:
    import torch
    import torch.nn.functional as F

    v = logits.size(-1)
    flat_l = logits.reshape(-1, v)
    flat_y = labels.reshape(-1)
    m = flat_y != ignore_index
    if not m.any():
        return 0.0, 0.0, 0
    ce_sum = F.cross_entropy(flat_l[m], flat_y[m], reduction="sum")
    probs = F.softmax(flat_l[m].float(), dim=-1)
    ent_sum = (-(probs * probs.clamp_min(1e-12).log()).sum(-1)).sum()
    n = int(m.sum().item())
    return float(ce_sum.item()), float(ent_sum.item()), n


def _build_pretraining_eval_callback(trainer: Any, val_parquet: str | None, max_batches: int) -> Any:
    from veomni.trainer.callbacks.evaluate_callback import EvaluateCallback

    class PretrainingEvalCallback(EvaluateCallback):
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
            from tqdm.auto import tqdm

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
                try:
                    _nel = len(ds)
                except TypeError:
                    _nel = "?"
                helper.logger.info_rank0(f"[eval] val mapping dataset size={_nel} path={self.val_parquet}")

                def collate(batch: list) -> Any:
                    # One dataset row -> list of plaintext chunks; evaluate every chunk (not only the first).
                    row = batch[0]
                    chunks = row if isinstance(row, list) else [row]
                    return chunks

                self._loader = DataLoader(
                    ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
                )
                self._built = True

            self.trainer.model.eval()
            total_ce_sum = 0.0
            total_ent_sum = 0.0
            n_tokens = 0
            n_tokens_ent = 0
            n_batches = 0
            doc_iters = 0
            try:
                _loader_len = min(self.max_batches, len(self._loader))  # type: ignore[arg-type]
            except Exception:
                _loader_len = None

            t0 = time.time()
            with _inference_inner(self.trainer.model) as inner:
                with torch.no_grad():
                    bar = tqdm(
                        self._loader,
                        desc=f"[eval] CE batches (step {state.global_step})",
                        total=_loader_len,
                        leave=True,
                        position=1,
                        file=sys.stdout,
                        mininterval=0.2,
                        dynamic_ncols=True,
                    )
                    for micro in bar:
                        if doc_iters >= self.max_batches:
                            break
                        doc_iters += 1
                        chunks = micro if isinstance(micro, list) else [micro]
                        for ch in chunks:
                            mb = self.trainer.collate_fn([ch])
                            mb = {
                                k: v.to(self.trainer.device, non_blocking=True) if torch.is_tensor(v) else v
                                for k, v in mb.items()
                            }
                            out = inner(**mb, use_cache=False)
                            if "labels" not in mb:
                                continue
                            labels = mb["labels"]
                            nt = int((labels != IGNORE_INDEX).sum().item())
                            if nt <= 0:
                                continue

                            if getattr(out, "loss", None) is not None:
                                ce_mean = float(out.loss.item())
                                ce_sum = ce_mean * nt
                                ent_part = getattr(out, "entropy", None)
                                if ent_part is not None:
                                    ent_sum = float(ent_part.item()) * nt
                                    n_tokens_ent += nt
                                else:
                                    ent_sum = 0.0
                            elif getattr(out, "logits", None) is not None:
                                logits = out.logits
                                if logits.shape[:2] == labels.shape:
                                    ce_sum, ent_sum, nt = _aligned_lm_token_sums(
                                        logits, labels, IGNORE_INDEX
                                    )
                                else:
                                    ce_sum, ent_sum, nt = _masked_lm_token_sums(
                                        logits, labels, IGNORE_INDEX
                                    )
                                if nt == 0:
                                    continue
                                n_tokens_ent += nt
                            else:
                                continue
                            total_ce_sum += ce_sum
                            total_ent_sum += ent_sum
                            n_tokens += nt
                            n_batches += 1
                    bar.close()

            helper.logger.info_rank0(
                f"[eval] step={state.global_step} CE eval done in {(time.time() - t0):.2f}s "
                f"({n_batches} chunk-forwards, tokens={n_tokens})"
            )

            if n_tokens > 0:
                mean_ce = total_ce_sum / n_tokens
                ppl = math.exp(mean_ce)
                mean_ent = (total_ent_sum / n_tokens_ent) if n_tokens_ent > 0 else None
                ent_str = f" mean_ent={mean_ent:.4f}" if mean_ent is not None else ""
                helper.logger.info_rank0(
                    f"[eval] step={state.global_step} mean_ce={mean_ce:.4f} ppl={ppl:.4f}"
                    f"{ent_str} (tokens={n_tokens}, chunk_forwards={n_batches})"
                )
                try:
                    import wandb

                    if args.train.wandb.enable:
                        payload = {
                            "eval/loss": mean_ce,
                            "eval/perplexity": ppl,
                            "eval/tokens": n_tokens,
                        }
                        if mean_ent is not None:
                            payload["eval/entropy"] = mean_ent
                        wandb.log(payload, step=int(state.global_step))
                except Exception:
                    pass
            else:
                helper.logger.warning_rank0(
                    "[eval] No supervised tokens in validation batches (check val parquet / tokenizer)."
                )

            if dist.is_initialized():
                dist.barrier()
            self.trainer.model.train()

    return PretrainingEvalCallback(trainer, val_parquet, max_batches)


def _install_checkpoint_post_hook(trainer: Any, cfg: ContinuedPretrainingConfig) -> None:
    import torch.distributed as dist

    cb = getattr(trainer.base, "checkpointer_callback", None)
    if cb is None:
        return
    orig_save = cb._save_checkpoint
    tokens_per_step = int(cfg.global_batch_size) * int(cfg.max_seq_len)
    save_path = Path(trainer.base.args.train.checkpoint.save_path)
    yaml_src = Path(cfg.config_path)

    def _wrapped_save(state: Any) -> None:
        orig_save(state)
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        gs = int(state.global_step)
        ckpt_dir = save_path / f"global_step_{gs}"
        if not ckpt_dir.is_dir():
            return
        toks_upper = gs * tokens_per_step
        link_name = f"step_{gs}_tokens_{toks_upper}"
        link_path = save_path / link_name
        try:
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(ckpt_dir.name, target_is_directory=True)
        except OSError:
            pass
        try:
            shutil.copy2(yaml_src, ckpt_dir / "run_config.yaml")
        except OSError:
            pass

    cb._save_checkpoint = _wrapped_save  # type: ignore[assignment]


def _attach_pretraining_metrics(trainer: Any, cfg: ContinuedPretrainingConfig) -> None:
    import torch
    import torch.nn.functional as F

    args = trainer.base.args
    if not args.train.wandb.enable:
        return

    inner = _unwrap_for_generate(trainer.base.model)
    head = getattr(inner, "lm_head", None)
    ent_acc = {"sum": 0.0, "n": 0}
    log_lm_entropy = _bool(os.environ.get("CONTINUED_PT_LOG_LM_ENTROPY", ""), default=False)

    if head is not None and log_lm_entropy:

        def _ent_hook(_mod: Any, _inp: Any, out: Any) -> None:
            if _mod.training or out is None:
                return
            with torch.no_grad():
                p = F.softmax(out.float(), dim=-1)
                e = -(p * p.clamp_min(1e-12).log()).sum(-1)
                ent_acc["sum"] += float(e.mean().item())
                ent_acc["n"] += 1

        head.register_forward_hook(_ent_hook)

    tokens_per_step = int(cfg.global_batch_size) * int(cfg.max_seq_len)
    state: dict[str, Any] = {"start_time": None, "start_step": None}
    _orig = trainer.on_step_end

    def _wrapped_step_end(self: Any, loss: Any = None, loss_dict: Any = None, grad_norm: Any = None) -> None:
        _orig(self, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        if self.base.args.train.global_rank != 0 or not self.base.args.train.wandb.enable:
            ent_acc["sum"], ent_acc["n"] = 0.0, 0
            return

        gs = int(self.base.state.global_step)
        if state["start_time"] is None:
            state["start_time"] = time.time()
            state["start_step"] = gs

        elapsed = max(1e-6, time.time() - float(state["start_time"]))
        steps_done = max(1, gs - int(state["start_step"]))
        toks_total = gs * tokens_per_step
        toks_since_start = steps_done * tokens_per_step

        payload: dict[str, Any] = {
            "train/tokens_total": float(toks_total),
            "train/tokens_per_sec": float(toks_since_start / elapsed),
            "train/seconds_elapsed": float(elapsed),
        }
        if ent_acc["n"] > 0:
            payload["train/entropy_lm_mean"] = float(ent_acc["sum"] / ent_acc["n"])

        try:
            import wandb

            wandb.log(payload, step=gs)
        except Exception:
            pass

        ent_acc["sum"], ent_acc["n"] = 0.0, 0

    trainer.on_step_end = types.MethodType(_wrapped_step_end, trainer)


def _warn_about_packed_attention_oom_risk(cfg: ContinuedPretrainingConfig) -> None:
    if not cfg.packing:
        return
    packed_len = int(cfg.micro_batch_size) * int(cfg.max_seq_len)
    if packed_len <= 4096:
        return
    attn = _effective_attn_implementation(cfg)
    if attn in ("flash_attention_2", "flash_attention_3"):
        return
    print(
        f"[continue_pretraining][warn] packing=true with micro_batch_size={cfg.micro_batch_size} "
        f"and max_seq_len={cfg.max_seq_len} packs each micro batch into a sequence of {packed_len} tokens. "
        "Under SDPA this often forces a math attention kernel and OOMs.\n"
        "  Fixes: install flash-attn (``attn_implementation: auto``), or set ``packing: false``, "
        "or lower ``micro_batch_size`` so micro_batch_size × max_seq_len ≤ 4096.",
        flush=True,
    )


def _patch_compute_train_steps_for_prepared_plaintext_parquet(args: Any, max_steps_budget: int) -> None:
    if max_steps_budget <= 0:
        return
    orig = args.compute_train_steps

    def wrapped(self: Any, dataset_length: int | None = None) -> None:
        orig(self, dataset_length)
        cap = int(max_steps_budget)
        if self.train.dyn_bsz:
            if self._train_steps < cap:
                need_tokens = cap * int(self.train.global_batch_size) * int(self.data.max_seq_len)
                self.data.train_size = max(int(self.data.train_size), need_tokens)
                orig(self, None)
            return
        if self._train_steps < cap:
            dbl = max(1, int(self.train.dataloader_batch_size))
            self.data.train_sample = max(int(self.data.train_sample), cap * dbl)
            orig(self, None)

    args.compute_train_steps = types.MethodType(wrapped, args)


def run_veomni(
    cfg: ContinuedPretrainingConfig,
    train_parquet: str,
    val_parquet: str | None,
    estimated_train_tokens: int,
    derived_max_steps: int,
) -> None:
    attn = _effective_attn_implementation(cfg)
    if attn in ("flash_attention_2", "flash_attention_3") and not _flash_attn_importable():
        raise SystemExit(
            f"Resolved attention backend is {attn!r} (from YAML / veomni_overrides) but ``import flash_attn`` "
            "failed. Remove the override or install flash-attn."
        )

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            f"[continue_pretraining] attention_backend={attn!r} packing={cfg.packing} "
            f"(flash_attn installed: {_flash_attn_importable()})",
            flush=True,
        )

    _warn_about_packed_attention_oom_risk(cfg)
    _enable_torch_throughput_knobs()
    sys.path.insert(0, str(_VEOMNI_SRC))

    from veomni.arguments import parser as ve_parser
    from veomni.arguments.arguments_types import VeOmniArguments
    from veomni.trainer.text_trainer import TextTrainer

    raw = _veomni_dict(cfg, train_parquet, val_parquet, estimated_train_tokens, derived_max_steps)
    args = ve_parser._instantiate_recursive(VeOmniArguments, raw)

    import pyarrow.parquet as pq

    num_rows = max(1, pq.ParquetFile(train_parquet).metadata.num_rows)
    if args.train.dyn_bsz:
        args.data.train_sample = num_rows
    _patch_compute_train_steps_for_prepared_plaintext_parquet(args, int(derived_max_steps))

    trainer = TextTrainer(args)
    if val_parquet:
        trainer.base.evaluate_callback = _build_pretraining_eval_callback(
            trainer.base, val_parquet, cfg.eval_max_batches
        )
    _attach_pretraining_metrics(trainer, cfg)
    _install_checkpoint_post_hook(trainer, cfg)
    trainer.train()

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        base = trainer.base
        gs = int(base.state.global_step)
        if gs > 0:
            ckpt_dir = Path(base.args.train.checkpoint.save_path) / f"global_step_{gs}"
            print(f"[continue_pretraining] Final checkpoint (DCP): {ckpt_dir.resolve()}", flush=True)
            if getattr(base.args.train.checkpoint, "save_hf_weights", False):
                print(f"[continue_pretraining] HuggingFace export: {(ckpt_dir / 'hf_ckpt').resolve()}", flush=True)
        else:
            print("[continue_pretraining] No checkpoint directory (global_step is 0).", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: continue_pretraining.py path/to/config.yaml")
    yaml_path = Path(sys.argv[1]).expanduser().resolve()
    raw = _load_yaml(yaml_path)
    cfg = ContinuedPretrainingConfig.from_dict(raw, str(yaml_path))
    cfg.config_path = str(yaml_path)
    _finalize_resume_checkpoint_path(cfg)
    _validate_batch_geometry(cfg)
    _auto_adjust_packing_for_sdpa(cfg)
    _maybe_reexec_torchrun(cfg)

    out_root = Path(cfg.checkpoint_output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg.prepared_data_dir).expanduser().resolve() if cfg.prepared_data_dir else out_root
    data_root.mkdir(parents=True, exist_ok=True)

    train_pq = data_root / "prepared_train_text.parquet"
    val_pq = data_root / "prepared_val_text.parquet"
    stats_path = data_root / "prepared_text_stats.json"

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        try:
            shutil.copy2(yaml_path, out_root / yaml_path.name)
        except Exception as e:
            print(f"[continue_pretraining] warn: could not copy YAML to {out_root}: {e}", flush=True)

    need_val = int(cfg.num_validation_documents) > 0

    if world > 1 and local_rank != 0:
        deadline = time.time() + 3600.0
        while time.time() < deadline:
            if train_pq.is_file() and stats_path.is_file() and (not need_val or val_pq.is_file()):
                try:
                    import pyarrow.parquet as pq

                    pq.ParquetFile(str(train_pq))
                    if need_val:
                        pq.ParquetFile(str(val_pq))
                    break
                except Exception:
                    pass
            time.sleep(0.5)
        else:
            raise SystemExit(
                f"[continue_pretraining] rank {local_rank}: timed out waiting for prepared parquets under {data_root}"
            )
        with stats_path.open() as f:
            stats = json.load(f)
    else:
        reuse_ok = (
            train_pq.is_file()
            and stats_path.is_file()
            and (not need_val or val_pq.is_file())
        )
        if reuse_ok:
            with stats_path.open() as f:
                stats = json.load(f)
            print(
                f"[continue_pretraining] Reusing existing prepared parquets:\n"
                f"  train: {train_pq}\n"
                f"  val:   {val_pq if need_val else '(none)'}\n"
                f"  stats: {stats}",
                flush=True,
            )
        else:
            train_pq, val_pq_out, stats = build_pretraining_parquets(cfg, data_root)
            if val_pq_out is not None:
                val_pq = val_pq_out
            with stats_path.open("w") as f:
                json.dump(stats, f, indent=2)

    estimated_train_tokens = int(stats.get("estimated_train_tokens", cfg.max_training_tokens))
    tokens_per_step = int(cfg.global_batch_size) * int(cfg.max_seq_len)
    derived_max_steps = max(
        1, math.ceil(int(cfg.max_training_tokens) / max(1, tokens_per_step))
    )
    if cfg.max_steps is not None:
        derived_max_steps = int(cfg.max_steps)

    if local_rank == 0:
        attn = _effective_attn_implementation(cfg)
        print(
            f"[continue_pretraining] training plan:\n"
            f"  model = {cfg.init_model_path}\n"
            f"  training token budget = {cfg.max_training_tokens:,} (max_steps = {derived_max_steps:,}; "
            f"per step ≈ {tokens_per_step:,} = global_batch_size × max_seq_len)\n"
            f"  global_batch_size={cfg.global_batch_size}, micro_batch_size={cfg.micro_batch_size}, "
            f"max_seq_len={cfg.max_seq_len}, packing={cfg.packing}, fsdp_mode={cfg.fsdp_mode}\n"
            f"  attention={attn}, dataloader_workers={cfg.dataloader_num_workers}\n"
            f"  num_gpus={cfg.num_gpus} (world={world})\n"
            f"  resume_checkpoint_path={cfg.resume_checkpoint_path}",
            flush=True,
        )

    eval_parquet = (
        str(val_pq)
        if cfg.eval_every_steps > 0 and need_val and val_pq.is_file()
        else None
    )
    run_veomni(cfg, str(train_pq), eval_parquet, estimated_train_tokens, derived_max_steps)


if __name__ == "__main__":
    main()
