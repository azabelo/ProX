# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
LLaDA-style **masked diffusion language model** (MDM) — bidirectional self-attention + RoPE, inspired by
`GSAI-ML/LLaDA-8B-Base <https://huggingface.co/GSAI-ML/LLaDA-8B-Base>`_ (no copied weights; tiny 3-layer stack).

Training: sample a discrete timestep, mask a random fraction of (non-pad) tokens, predict the clean token at
masked positions (cross-entropy). This replaces causal next-token loss for this architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from veomni.utils.constants import IGNORE_INDEX

from .configuration_llada_mini import LladaMiniConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _rope_cos_sin(
    max_positions: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_positions, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LladaMiniAttention(nn.Module):
    """Bidirectional multi-head attention + RoPE (MDM / LLaDA-style)."""

    def __init__(self, config: LladaMiniConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        nh = config.num_attention_heads
        assert h % nh == 0
        self.head_dim = h // nh
        self.n_heads = nh
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.attn_drop = nn.Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor, attn_padding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype
        cos, sin = _rope_cos_sin(
            self.config.max_position_embeddings,
            self.head_dim,
            self.config.rope_theta,
            device,
            dtype,
        )
        cos = cos[:, :, :seq_len, :]
        sin = sin[:, :, :seq_len, :]

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attn_padding_bias is not None:
            scores = scores + attn_padding_bias
        p = F.softmax(scores, dim=-1).to(dtype=dtype)
        p = self.attn_drop(p)
        out = torch.matmul(p, v)
        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.config.hidden_size)
        return self.o_proj(out)


class LladaMiniMLP(nn.Module):
    def __init__(self, config: LladaMiniConfig) -> None:
        super().__init__()
        h = config.hidden_size
        inter = config.intermediate_size
        self.gate = nn.Linear(h, inter, bias=False)
        self.up = nn.Linear(h, inter, bias=False)
        self.down = nn.Linear(inter, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LladaMiniBlock(nn.Module):
    def __init__(self, config: LladaMiniConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LladaMiniAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LladaMiniMLP(config)
        self.drop = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor, attn_padding_bias: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x), attn_padding_bias))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class LladaMiniPreTrainedModel(PreTrainedModel):
    config_class = LladaMiniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LladaMiniBlock", "RMSNorm"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)


class LladaMiniModel(LladaMiniPreTrainedModel):
    """Bidirectional stack (default MDM attention bias = zeros → fully visible within sequence)."""

    def __init__(self, config: LladaMiniConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.time_emb = nn.Embedding(config.num_diffusion_timesteps, config.hidden_size)
        self.layers = nn.ModuleList([LladaMiniBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Keep embedding LUTs in FP32; transformer weights stay in model dtype (typically BF16).
        self.embed_tokens.to(dtype=torch.float32)
        self.time_emb.to(dtype=torch.float32)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def _bidirectional_padding_bias(
        self, attention_mask: Optional[torch.Tensor], seq_len: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """[B,1,L,L] additive mask: mask keys where pad (HF mask 1 = keep)."""
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            return None
        bsz = attention_mask.shape[0]
        # Key mask: columns where pad → -inf
        km = (1.0 - attention_mask.to(dtype=dtype)).view(bsz, 1, 1, seq_len)
        km = km * torch.finfo(dtype).min
        return km.expand(bsz, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor,]:
        del kwargs
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        # Blocks run in reduced precision (e.g. BF16); embeddings computed/stored in FP32.
        compute_dtype = self.layers[0].self_attn.q_proj.weight.dtype
        x = x.to(compute_dtype)
        t_emb = self.time_emb(timestep).unsqueeze(1).to(dtype=compute_dtype)
        x = x + t_emb

        seq_len = x.shape[1]
        pad_bias = self._bidirectional_padding_bias(attention_mask, seq_len, x.dtype)

        for layer in self.layers:
            x = layer(x, pad_bias)

        x = self.norm_f(x)
        return (x,)


class LladaMiniForCausalLM(LladaMiniPreTrainedModel, GenerationMixin):
    """HF causal LM API shape; **training loss is diffusion masking CE**, not shifted causal LM."""

    config_class = LladaMiniConfig

    def __init__(self, config: LladaMiniConfig):
        super().__init__(config)
        self.model = LladaMiniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del use_cache, kwargs

        if input_ids is None and inputs_embeds is None:
            raise ValueError("LladaMini requires input_ids or inputs_embeds")

        # ---------- Inference / scoring: no masking objective ----------
        if labels is None:
            if input_ids is None:
                raise ValueError("Provide input_ids when labels is None")
            ts = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            hidden = self.model(input_ids=input_ids, timestep=ts, attention_mask=attention_mask)[0]
            logits = self.lm_head(hidden)
            if self.config.scale_logits:
                logits = logits * (1.0 / math.sqrt(self.config.hidden_size))
            return CausalLMOutputWithPast(loss=None, logits=logits)

        # ---------- Diffusion objective (train & eval when labels present): ----------
        clean = input_ids
        assert clean is not None
        device = clean.device
        bsz, seq_len = clean.shape

        valid = attention_mask.bool() if attention_mask is not None else torch.ones_like(clean, dtype=torch.bool)
        # Only denoise supervised positions (same mask semantics as ``text_target``: prompt = IGNORE_INDEX).
        supervised = labels.ne(IGNORE_INDEX)

        # timestep per sequence
        T = self.config.num_diffusion_timesteps
        t = torch.randint(0, T, (bsz,), device=device, dtype=torch.long)
        # Linear mask rate ↑ with t (simple MDM schedule)
        frac = (t.float() + 1.0) / float(T)

        rand = torch.rand(bsz, seq_len, device=device)
        mask_positions = rand < frac.unsqueeze(1)

        mask_positions = mask_positions & valid & supervised

        # Ensure at least one supervised token per row when possible
        for i in range(bsz):
            row_mask = mask_positions[i]
            sup_i = supervised[i] & valid[i]
            if not row_mask.any() and sup_i.any():
                idx = torch.nonzero(sup_i, as_tuple=False).squeeze(-1)
                pick = idx[torch.randint(0, idx.numel(), (1,), device=device)]
                mask_positions[i, pick] = True

        noisy = clean.clone()
        mid = self.config.mask_token_id
        noisy[mask_positions] = mid

        hidden = self.model(input_ids=noisy, timestep=t, attention_mask=attention_mask)[0]
        logits = self.lm_head(hidden)
        if self.config.scale_logits:
            logits = logits * (1.0 / math.sqrt(self.config.hidden_size))

        # CE only where we masked; targets are **clean** tokens
        if mask_positions.any():
            loss = F.cross_entropy(
                logits[mask_positions],
                clean[mask_positions],
                reduction="mean",
            )
        else:
            loss = logits.sum() * 0.0

        return CausalLMOutputWithPast(loss=loss, logits=logits)
