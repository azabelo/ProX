"""Hugging Face tokenizer loading shared by vLLM inference tasks."""

from __future__ import annotations

from typing import Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_auto_tokenizer_fast_fallback(model_path: str) -> Tuple[PreTrainedTokenizerBase, bool]:
    """
    Prefer the fast (Rust) tokenizer; fall back to slow if ``tokenizer.json`` / tokenizers fails.

    VeOmni HF exports occasionally ship a ``tokenizer.json`` that does not parse under the
    installed ``tokenizers`` crate (Rust ``ModelWrapper`` error); slow uses vocab/merges instead.

    Returns:
        (tokenizer, used_slow_fallback). When ``used_slow_fallback`` is True, pass
        ``tokenizer_mode=\"slow\"`` to vLLM; otherwise ``tokenizer_mode=\"auto\"``.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
        return tok, False
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        return tok, True
