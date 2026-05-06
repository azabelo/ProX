"""
Qwen-like decoder LM where *one* block uses a selective SSM (Mamba-style) residual instead of GQA.

The inner SSM runs in ``mamba_d_inner`` dims (typically < hidden) to keep projections tractable.

Factory: ``my_architectures.qwen_like_one_mamba_layer:build_model``
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from my_architectures.qwen_like_one_swa_layer import (
    GroupedQueryAttention,
    RMSNorm,
    SwiGLU,
)

_DEFAULT_ARCH: dict[str, Any] = {
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "vocab_size": 151936,
    "mamba_layer_index": 12,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_d_inner": 512,
}


def _yaml_arch_extra(config_path: str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text()) or {}
    return dict(raw.get("custom_arch") or {})


class MambaBlock(nn.Module):
    """Gated causal conv + selective discrete SSM scan; projects to/from ``d_model``."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        d_conv: int,
        dt_rank: int | None = None,
        dt_scale: float = 0.03,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(16, d_model // 16)
        self.dt_scale = dt_scale

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_inner * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        Bm: torch.Tensor,
        Cm: torch.Tensor,
    ) -> torch.Tensor:
        """u, delta [B,L,DI]; Bm, Cm [B,L,DI,N] → [B,L,DI]."""
        bsz, seq_len, d_inner = u.shape
        n = self.d_state
        A = -torch.exp(self.A_log.to(dtype=u.dtype).clamp(min=-20.0, max=2.0))
        h = torch.zeros(bsz, d_inner, n, dtype=u.dtype, device=u.device)
        ys: list[torch.Tensor] = []
        for i in range(seq_len):
            dti = delta[:, i, :, None].clamp(min=1e-5, max=1.0)
            exp_fact = torch.exp(dti * A.unsqueeze(0))
            inp = u[:, i, :, None] * Bm[:, i]
            h = exp_fact * h + dti * inp
            y_i = (h * Cm[:, i]).sum(dim=-1)
            ys.append(y_i)
        return torch.stack(ys, dim=1)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1).to(x.dtype)

        bsz, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_gate, z_gate = xz.chunk(2, dim=-1)
        x_c = self.conv1d(x_gate.transpose(1, 2))
        x_c = x_c[..., :seq_len].transpose(1, 2).contiguous()
        x_c = F.silu(x_c)

        xdbl = self.x_proj(x_c)
        dt_r, b_flat, c_flat = torch.split(
            xdbl,
            [self.dt_rank, self.d_inner * self.d_state, self.d_inner * self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_r)) * self.dt_scale + 1e-4
        bm = b_flat.view(bsz, seq_len, self.d_inner, self.d_state)
        cm = c_flat.view(bsz, seq_len, self.d_inner, self.d_state)

        y = self._selective_scan(x_c, delta, bm, cm)
        y = y + x_c * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z_gate)
        return self.out_proj(y)


class AttentionDecoderLayer(nn.Module):
    """Standard full-causal GQA + SwiGLU."""

    def __init__(self, cfg: "_MambaArchConfig", rope_max: int) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.rope_theta,
            rope_max,
            sliding_window=None,
        )
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MambaDecoderLayer(nn.Module):
    """Residual Mamba SS SSM branch + SwiGLU."""

    def __init__(self, cfg: "_MambaArchConfig", rope_unused: int) -> None:
        super().__init__()
        _ = rope_unused
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mamba = MambaBlock(
            d_model=cfg.hidden_size,
            d_inner=cfg.mamba_d_inner,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            dt_rank=max(16, cfg.hidden_size // 16),
        )
        self.post_mamba_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        mask = attention_mask.bool() if attention_mask is not None else None
        x = x + self.mamba(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_mamba_layernorm(x))
        return x


@dataclass
class _MambaArchConfig:
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
    mamba_layer_index: int
    mamba_d_inner: int
    mamba_d_state: int
    mamba_d_conv: int


class QwenLikeOneMambaLM(nn.Module):
    def __init__(self, cfg: _MambaArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        layers: list[nn.Module] = []
        for i in range(cfg.num_hidden_layers):
            if i == cfg.mamba_layer_index:
                layers.append(MambaDecoderLayer(cfg, cfg.max_seq_len))
            else:
                layers.append(AttentionDecoderLayer(cfg, cfg.max_seq_len))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        # VeOmni / HF-style callers may provide `position_ids`, `use_cache`, etc.
        # This minimal model ignores them.
        _ = position_ids, use_cache, kwargs
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
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


def build_model(ft_cfg_dict: dict[str, Any]) -> nn.Module:
    yaml_extra = _yaml_arch_extra(str(ft_cfg_dict.get("config_path", "")))
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
    n_layers = int(arch["num_hidden_layers"])
    mamba_idx = int(arch["mamba_layer_index"])
    if not (0 <= mamba_idx < n_layers):
        raise SystemExit(f"custom_arch.mamba_layer_index must be in [0, {n_layers - 1}], got {mamba_idx}")

    cfg = _MambaArchConfig(
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
        mamba_layer_index=mamba_idx,
        mamba_d_inner=int(arch["mamba_d_inner"]),
        mamba_d_state=int(arch["mamba_d_state"]),
        mamba_d_conv=int(arch["mamba_d_conv"]),
    )
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    assert cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim

    model = QwenLikeOneMambaLM(cfg)
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
    return model
