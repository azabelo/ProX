"""
Qwen2.5~0.5B-shaped decoder-only LM: GQA, RoPE, RMSNorm, SwiGLU — one block uses sliding-window attention.

Use with finetune.py ``model_type: custom_local`` and
``custom_model_factory: my_architectures.qwen_like_one_swa_layer:build_model``.

Extra knobs live under ``custom_arch`` in the same YAML (loaded via ``FinetuneConfig.config_path``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Defaults aligned with Qwen2.5-0.5B (see HuggingFace ``config.json``).
_DEFAULT_ARCH: dict[str, Any] = {
    "vocab_size": 151936,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    # One layer (default: middle) uses sliding-window self-attention instead of full causal.
    "swa_layer_index": 12,
    "sliding_window": 128,
}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, H, L, Dh]; cos,sin broadcast to Dh."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def _rope_cos_sin(
    max_positions: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_positions, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_h = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin_h = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos_h[None, None, :, :], sin_h[None, None, :, :]  # [1,1,L,Dh]


def build_attn_bias(
    bsz: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Additive bias [B,1,L,L] for attention logits (0 attend, -inf mask).
    key_padding_mask: [B,L] True = keep token.
    """
    bias = torch.zeros(bsz, 1, seq_len, seq_len, device=device, dtype=dtype)
    # Causal: no j > i
    tri = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    bias.masked_fill_(tri.view(1, 1, seq_len, seq_len), float("-inf"))
    if sliding_window is not None and sliding_window > 0:
        # Mask if distance i - j >= window (j left of i). Only past keys.
        for i in range(seq_len):
            lo = max(0, i - sliding_window + 1)
            if lo > 0:
                bias[:, :, i, :lo] = float("-inf")
    if key_padding_mask is not None:
        # Mask keys where padding
        pad = ~key_padding_mask  # [B,L]
        bias.masked_fill_(pad[:, None, None, :].expand_as(bias), float("-inf"))
    return bias


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        max_positions: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.repeats = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.register_buffer("_rope_cos", torch.zeros(1, 1, 1, 1), persistent=False)
        self.register_buffer("_rope_sin", torch.zeros(1, 1, 1, 1), persistent=False)
        self._max_pos = max_positions
        self._rope_theta = rope_theta

    def _ensure_rope(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._rope_cos.shape[2] >= self._max_pos and self._rope_cos.device == device:
            return
        cos, sin = _rope_cos_sin(self._max_pos, self.head_dim, self._rope_theta, device, dtype)
        self._rope_cos = cos
        self._rope_sin = sin

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype
        self._ensure_rope(device, dtype)
        cos = self._rope_cos[:, :, :seq_len, :]
        sin = self._rope_sin[:, :, :seq_len, :]

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        # Repeat KV for GQA
        if self.repeats > 1:
            k = k[:, :, None, :, :].expand(bsz, self.num_kv_heads, self.repeats, seq_len, self.head_dim)
            k = k.reshape(bsz, self.num_heads, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(bsz, self.num_kv_heads, self.repeats, seq_len, self.head_dim)
            v = v.reshape(bsz, self.num_heads, seq_len, self.head_dim)

        key_padding = attention_mask.bool() if attention_mask is not None else None
        attn_bias = build_attn_bias(
            bsz,
            seq_len,
            device,
            dtype,
            sliding_window=self.sliding_window,
            key_padding_mask=key_padding,
        )
        q2 = q * (self.head_dim**-0.5)
        # [B,H,L,L]
        scores = torch.matmul(q2, k.transpose(-1, -2))
        scores = scores + attn_bias
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(ctx)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: "_ArchConfig", layer_idx: int) -> None:
        super().__init__()
        swa = cfg.swa_layer_index == layer_idx
        window = cfg.sliding_window if swa else None
        self.self_attn = GroupedQueryAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.max_seq_len,
            sliding_window=window,
        )
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@dataclass
class _ArchConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_seq_len: int
    swa_layer_index: int
    sliding_window: int


class QwenLikeOneSWALM(nn.Module):
    """Decoder LM with exactly one SWA layer; logits use tied input embeddings."""

    def __init__(self, cfg: _ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        _ = position_ids  # accepted for VeOmni/HF compatibility; RoPE is computed internally
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        # Weight-tied LM head
        logits = F.linear(x, self.embed_tokens.weight)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        out = type("Out", (), {})()
        out.logits = logits
        out.loss = loss
        return out


def _load_yaml_arch(config_path: str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    return dict(raw.get("custom_arch") or {})


def build_model(ft_cfg_dict: dict[str, Any]) -> nn.Module:
    """
    Factory for ``finetune.py`` custom_local. ``ft_cfg_dict`` is ``dataclasses.asdict(FinetuneConfig)``.
    Reads ``custom_arch`` from the YAML at ``config_path``; merges with ``_DEFAULT_ARCH``.
    ``vocab_size`` defaults from HuggingFace config of ``init_model_path`` when present.
    """
    yaml_extra = _load_yaml_arch(str(ft_cfg_dict.get("config_path", "")))
    arch = {**_DEFAULT_ARCH, **yaml_extra}

    tok_path = ft_cfg_dict.get("tokenizer_path") or ft_cfg_dict.get("init_model_path")
    if not tok_path:
        raise SystemExit("init_model_path (or tokenizer_path) required for vocab size")

    vocab_size = arch.get("vocab_size")
    if vocab_size is None or int(vocab_size) <= 0:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(str(tok_path), trust_remote_code=True)
        vocab_size = int(getattr(hf_cfg, "vocab_size", 151936))
    else:
        vocab_size = int(vocab_size)

    max_seq_len = int(ft_cfg_dict.get("max_seq_len", 2048))
    swa_idx = int(arch["swa_layer_index"])
    n_layers = int(arch["num_hidden_layers"])
    if not (0 <= swa_idx < n_layers):
        raise SystemExit(f"custom_arch.swa_layer_index must be in [0, {n_layers - 1}], got {swa_idx}")

    cfg = _ArchConfig(
        vocab_size=vocab_size,
        hidden_size=int(arch["hidden_size"]),
        intermediate_size=int(arch["intermediate_size"]),
        num_hidden_layers=n_layers,
        num_attention_heads=int(arch["num_attention_heads"]),
        num_key_value_heads=int(arch["num_key_value_heads"]),
        head_dim=int(arch["head_dim"]),
        rms_norm_eps=float(arch["rms_norm_eps"]),
        rope_theta=float(arch["rope_theta"]),
        max_seq_len=max_seq_len,
        swa_layer_index=swa_idx,
        sliding_window=int(arch["sliding_window"]),
    )
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    assert cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim

    model = QwenLikeOneSWALM(cfg)
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
    return model
