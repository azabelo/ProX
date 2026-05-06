#!/usr/bin/env python3
"""
Run gair-prox/web-doc-refining-lm on a handful of FineWeb documents (Transformers only; no vLLM).

Use this when CUDA drivers and the installed PyTorch build disagree (torch.cuda.is_available()
is False) or you want a small offline smoke test. Mirrors the prompt wiring in
`data_gen.tasks.apply_doc_refining` (Llama-2 chat template + same system string).

Outputs per document under --out-dir:
  doc_{i:03d}_input.txt          raw FineWeb `text`
  doc_{i:03d}_model_output.txt   decoded model continuation (the generated "program" / answer)
  doc_{i:03d}_refined.txt        text after `utils.doc_utils.execute_meta_operations`
manifest.json                   index of files and source shard / row
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.doc_utils import execute_meta_operations  # noqa: E402


def _default_fineweb_dir() -> str:
    return os.path.normpath(
        os.path.join(_REPO_ROOT, "data", "raw", "HuggingFaceFW", "fineweb", "sample", "10BT")
    )


def _iter_fineweb_texts(data_dir: str, max_docs: int):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet shards under {data_dir!r}")
    got = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=512):
            col = batch.column(0)
            for i in range(batch.num_rows):
                txt = col[i].as_py()
                if not isinstance(txt, str):
                    continue
                yield path, got, txt
                got += 1
                if got >= max_docs:
                    return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fineweb-dir", default=os.environ.get("FINEWEB_DATA_DIR", _default_fineweb_dir()))
    p.add_argument(
        "--out-dir",
        default=os.path.join(_REPO_ROOT, "data", "refining_fineweb_demo"),
        help="Directory to write paired inputs/outputs",
    )
    p.add_argument("--n-docs", type=int, default=10)
    p.add_argument("--model-path", default="gair-prox/web-doc-refining-lm")
    p.add_argument(
        "--token-template",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Tokenizer used for chat template (same default as apply_doc_refining.py)",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--prompt-max-length", type=int, default=2000)
    args = p.parse_args()

    fineweb_dir = os.path.expanduser(args.fineweb_dir)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    chat_tok = AutoTokenizer.from_pretrained(args.token_template, use_fast=False)
    base_tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    chat_tok.bos_token = base_tok.bos_token
    chat_tok.eos_token = base_tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=False,
    )
    model.eval()
    model.to(device)

    manifest: list[dict] = []
    system_msg = "You are a helpful, respectful and honest assistant."

    for idx, (shard_path, row_idx, user_msg) in enumerate(
        tqdm(list(_iter_fineweb_texts(fineweb_dir, args.n_docs)), desc="documents")
    ):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        input_ids = chat_tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=args.prompt_max_length,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(device)
        pad_id = base_tok.pad_token_id or base_tok.eos_token_id
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=base_tok.eos_token_id,
            )
        gen_ids = out[0, input_ids.shape[1] :]
        gen_text = chat_tok.decode(gen_ids, skip_special_tokens=True).strip()
        refined = execute_meta_operations(user_msg, gen_text)

        stem = f"doc_{idx:03d}"
        paths = {
            "input": os.path.join(out_dir, f"{stem}_input.txt"),
            "model_output": os.path.join(out_dir, f"{stem}_model_output.txt"),
            "refined": os.path.join(out_dir, f"{stem}_refined.txt"),
        }
        with open(paths["input"], "w", encoding="utf-8") as f:
            f.write(user_msg)
        with open(paths["model_output"], "w", encoding="utf-8") as f:
            f.write(gen_text)
        with open(paths["refined"], "w", encoding="utf-8") as f:
            f.write(refined)

        manifest.append(
            {
                "index": idx,
                "fineweb_shard": os.path.basename(shard_path),
                "fineweb_row_in_run": row_idx,
                "files": paths,
            }
        )

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "token_template": args.token_template,
                "device": str(device),
                "n_docs": len(manifest),
                "fineweb_dir": fineweb_dir,
                "entries": manifest,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(manifest)} document(s) under {out_dir}")


if __name__ == "__main__":
    main()
