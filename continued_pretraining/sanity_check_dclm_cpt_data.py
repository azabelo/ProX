#!/usr/bin/env python3
"""Sanity-check DCLM CPT plaintext pipeline: transforms, packing labels, padded batch, GPU forward.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=5 python continued_pretraining/sanity_check_dclm_cpt_data.py \\
      continued_pretraining/continue_pretraining_dclm_raw.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "yaml_path",
        nargs="?",
        default=str(_repo_root() / "continued_pretraining/continue_pretraining_dclm_raw.yaml"),
        help="CPT YAML (for paths + model + max_seq_len)",
    )
    args = ap.parse_args()
    root = _repo_root()
    yaml_path = Path(args.yaml_path).expanduser().resolve()
    raw_cfg = _load_yaml(yaml_path)

    init_model = raw_cfg["init_model_path"]
    max_seq_len = int(raw_cfg["max_seq_len"])
    prep = raw_cfg.get("prepared_data_dir")
    ckpt_out = raw_cfg.get("checkpoint_output_dir", "./continued_pretraining_outputs/run1")
    if prep:
        data_root = (root / str(prep).lstrip("./")).resolve() if not Path(prep).is_absolute() else Path(prep).resolve()
    else:
        data_root = (root / str(ckpt_out).lstrip("./")).resolve() if not Path(ckpt_out).is_absolute() else Path(
            ckpt_out
        ).resolve()
    train_pq = data_root / "prepared_train_text.parquet"
    if not train_pq.is_file():
        raise SystemExit(f"Missing train parquet: {train_pq}")

    veomni_src = root / "continued_pretraining/veomni"
    sys.path.insert(0, str(veomni_src))

    from veomni.data.data_collator import MainCollator
    from veomni.data.data_transform import build_data_transform
    from veomni.utils.constants import IGNORE_INDEX
    from veomni.utils.loss_utils import count_loss_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("This check expects a GPU (CUDA_VISIBLE_DEVICES=...)")

    print(f"[sanity] yaml={yaml_path}", flush=True)
    print(f"[sanity] train_parquet={train_pq}", flush=True)
    print(f"[sanity] device={device} ({torch.cuda.get_device_name(device)})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(init_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    transform = build_data_transform(
        "plaintext",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_keys="text",
    )

    table = pq.read_table(train_pq, columns=["text"])
    col = table.column(0)
    n = len(col)
    # Spread across the *train* parquet (already excludes val rows in CPT prep).
    if n < 4:
        raise SystemExit(f"Train parquet only has {n} rows; need >= 4 for packing check.")
    stride = max(1, n // 16)
    indices = sorted({min(n - 1, i * stride) for i in range(16)} | {0, n // 2, n - 1})
    features: list[dict] = []
    for idx in indices:
        if idx >= len(col):
            break
        text = col[idx].as_py()
        if not text or not str(text).strip():
            continue
        ex = transform({"text": str(text)})
        if not ex:
            continue
        # keep one chunk per row for packing test
        features.append({k: v.clone() for k, v in ex[0].items()})

    if len(features) < 4:
        raise SystemExit("Too few transformed chunks; widen indices or check parquet.")

    print(f"[sanity] built {len(features)} plaintext chunks from parquet rows", flush=True)

    # --- Per-chunk invariants (matches process_plaintext_example) ---
    for i, f in enumerate(features[:3]):
        ii, am, lb = f["input_ids"], f["attention_mask"], f["labels"]
        assert am.min() == 1 and am.max() == 1, f"chunk{i}: attention_mask must be all 1"
        assert ii.shape == lb.shape == am.shape, f"chunk{i}: shape mismatch"
        assert torch.equal(ii, lb), f"chunk{i}: plaintext labels must equal input_ids (shift is in loss/collator)"
        assert len(ii) <= max_seq_len, f"chunk{i}: length {len(ii)} > max_seq_len"

    print("[sanity] per-chunk: attention_mask all 1; labels == input_ids; len <= max_seq_len", flush=True)

    # --- Packed batch (same MainCollator path as training with dyn_bsz) ---
    collator = MainCollator()
    packed = collator(features)
    for k in ("input_ids", "labels", "attention_mask", "position_ids", "cu_seq_lens_q"):
        assert k in packed, f"missing key {k} in packed batch"

    # First token of each *concatenated* segment after the first should be masked in labels
    # (PackingCollator sets labels of segments 1..n-1 first token to IGNORE_INDEX)
    lb = packed["labels"][0]
    ignored = (lb == IGNORE_INDEX).sum().item()
    total = lb.numel()
    print(f"[sanity] packed: seq_len={total}, IGNORE_INDEX positions={ignored}", flush=True)
    assert ignored >= len(features) - 1, "expected at least (num_segments - 1) boundary-masked label positions"

    ct = count_loss_token(packed)
    n_supervised = int(ct["foundation_tokens"].item())
    print(f"[sanity] count_loss_token foundation_tokens={n_supervised} (supervised label positions)", flush=True)
    assert n_supervised > 0 and n_supervised < total, "supervised token count should be strictly between 0 and all"

    # --- HF rectangular batch (padding): mask 0 on pad, labels -100 on pad ---
    a = features[0]
    b = features[1]
    L = max(len(a["input_ids"]), len(b["input_ids"]))
    pad_id = tokenizer.pad_token_id or 0

    def pad_row(f: dict, length: int) -> dict:
        ids = f["input_ids"]
        pad_len = length - len(ids)
        assert pad_len >= 0
        pi = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
        pm = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        pl = torch.cat([f["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=f["labels"].dtype)])
        return {"input_ids": pi, "attention_mask": pm, "labels": pl}

    ra = pad_row(a, L)
    rb = pad_row(b, L)
    rect = {
        "input_ids": torch.stack([ra["input_ids"], rb["input_ids"]]).to(device),
        "attention_mask": torch.stack([ra["attention_mask"], rb["attention_mask"]]).to(device),
        "labels": torch.stack([ra["labels"], rb["labels"]]).to(device),
    }

    model = AutoModelForCausalLM.from_pretrained(
        init_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(**rect)
    loss = float(out.loss.item())
    print(f"[sanity] rectangular 2×{L} padded batch: loss={loss:.4f} (finite CE)", flush=True)
    assert loss == loss and loss < 100.0, "loss should be finite and not explode on real text"

    print("[sanity] OK — masking/packing invariants and a real GPU forward look sane.", flush=True)


if __name__ == "__main__":
    main()
