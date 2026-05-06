# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Test harness decoder: attention → SWA → MoE → Mamba-like → DeltaNet-like → attention (shared).

Registered as ``test_everything`` so VeOmni loads via ``build_foundation_model`` (not ``custom_model_factory``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func

from .configuration_test_everything import TestEverythingConfig


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


def _causal_bias(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """[1,1,L,L] additive mask: -inf above diagonal."""
    b = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    tri = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    b.masked_fill_(tri, float("-inf"))
    return b


def _swa_bias(seq_len: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Causal + sliding window: mask keys farther than ``window-1`` left of query."""
    b = _causal_bias(seq_len, device, dtype)
    if window <= 0:
        return b
    for i in range(seq_len):
        lo = max(0, i - window + 1)
        if lo > 0:
            b[:, :, i, :lo] = float("-inf")
    return b


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class _GQAAttention(nn.Module):
    """GQA self-attention with optional sliding-window mask (full vs SWA)."""

    def __init__(self, config: TestEverythingConfig, sliding_window: Optional[int]) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv = max(1, config.num_attention_heads // 4)
        self.head_dim = h // self.n_heads
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)
        self.register_buffer("_cos", torch.zeros(1), persistent=False)
        self.register_buffer("_sin", torch.zeros(1), persistent=False)
        self._rope_len = 0

    def _ensure_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._rope_len:
            return
        max_p = max(seq_len, self.config.max_position_embeddings)
        c, s = _rope_cos_sin(max_p, self.head_dim, self.config.rope_theta, device, dtype)
        self._cos = c
        self._sin = s
        self._rope_len = max_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        self._ensure_rope(seq_len, x.device, x.dtype)
        cos = self._cos[:, :, :seq_len, :].to(dtype=x.dtype)
        sin = self._sin[:, :, :seq_len, :].to(dtype=x.dtype)

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv, self.head_dim).transpose(1, 2)

        # RoPE on q,k (broadcast cos/sin to heads)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = (
                k[:, :, None, :, :]
                .expand(bsz, self.n_kv, rep, seq_len, self.head_dim)
                .reshape(bsz, self.n_heads, seq_len, self.head_dim)
            )
            v = (
                v[:, :, None, :, :]
                .expand(bsz, self.n_kv, rep, seq_len, self.head_dim)
                .reshape(bsz, self.n_heads, seq_len, self.head_dim)
            )

        if self.sliding_window is None:
            attn_bias = _causal_bias(seq_len, x.device, x.dtype)
        else:
            attn_bias = _swa_bias(seq_len, self.sliding_window, x.device, x.dtype)

        # [B,H,L,L]
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores + attn_bias
        p = F.softmax(scores, dim=-1)
        p = F.dropout(p, p=self.config.attention_dropout, training=self.training)
        out = torch.matmul(p, v)
        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MoELayer(nn.Module):
    """Top-``k`` sparse MoE (same routing pattern as ``qwen3_moe``).

    Router: softmax over experts, then ``torch.topk`` selects ``k`` experts per token.
    Pre-softmax ``router`` logits are returned when ``return_router_logits=True`` so HF
    ``load_balancing_loss_func`` matches ``qwen3_moe`` (softmax applied inside the loss).
    Set ``num_experts_per_tok == num_experts`` for dense mixture (``k == n``).
    With ``k < n``, some experts may be unused on a step (DDP may need ``find_unused_parameters=True``).
    """

    def __init__(self, config: TestEverythingConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.num_experts = config.num_experts
        raw_k = config.num_experts_per_tok
        if raw_k is None:
            raw_k = self.num_experts
        self.top_k = max(1, min(int(raw_k), self.num_experts))
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        inter = config.intermediate_size
        self.router = nn.Linear(h, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(h, inter, bias=False),
                nn.GELU(),
                nn.Linear(inter, h, bias=False),
            )
            for _ in range(self.num_experts)
        )

    def forward(self, x: torch.Tensor, return_router_logits: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = x.shape
        input_dtype = x.dtype
        x_flat = x.reshape(-1, h)
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(input_dtype)

        final = torch.zeros_like(x_flat)
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = x_flat[token_idx]
            cur = self.experts[expert_idx](current_state)
            cur = cur * router_top_value[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))

        out = final.view(bsz, seq_len, h)
        if return_router_logits:
            return out, router_logits
        return out


class MambaMini(nn.Module):
    """Compact selective SSM-style block (conv + shallow recurrence), not official Mamba-2."""

    def __init__(self, config: TestEverythingConfig) -> None:
        super().__init__()
        d = config.hidden_size
        di = config.mamba_d_inner
        self.in_proj = nn.Linear(d, di * 2, bias=False)
        self.conv1d = nn.Conv1d(di, di, config.mamba_d_conv, groups=di, padding=config.mamba_d_conv - 1)
        self.out_proj = nn.Linear(di, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, d = x.shape
        xz = self.in_proj(x)
        x1, z = xz.chunk(2, dim=-1)
        xc = x1.transpose(1, 2)
        xc = self.conv1d(xc)[:, :, :seq].transpose(1, 2)
        y = xc * F.silu(z)
        return self.out_proj(y)


class DeltaNetMini(nn.Module):
    """Depthwise conv + gated linear mix (DeltaNet/GatedDeltaNet-style *test stub*, not FLA kernels)."""

    def __init__(self, config: TestEverythingConfig) -> None:
        super().__init__()
        d = config.hidden_size
        k = config.delta_kernel_size
        self.dw = nn.Conv1d(d, d, k, padding=k - 1, groups=d, bias=True)
        self.gate = nn.Linear(d, d, bias=True)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, d = x.shape
        h = x.transpose(1, 2)
        h = self.dw(h)[:, :, :seq].transpose(1, 2)
        g = torch.sigmoid(self.gate(x))
        return self.proj(g * h + (1 - g) * x)


class TestEverythingPreTrainedModel(PreTrainedModel):
    config_class = TestEverythingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["RMSNorm"]
    # VeOmni passes ``attn_implementation="sdpa"`` from ``build_foundation_model``; PreTrainedModel
    # otherwise raises in ``_sdpa_can_dispatch`` for unknown classes. This stack does not use HF
    # ``ALL_ATTENTION_FUNCTIONS``—attention is hand-rolled matmul + mask—so SDPA gating is satisfied
    # without using ``scaled_dot_product_attention`` in this file.
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


class TestEverythingModel(TestEverythingPreTrainedModel):
    """Six sublayers with shared attention at positions 0 and 5."""

    def __init__(self, config: TestEverythingConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.shared_attn = _GQAAttention(config, sliding_window=None)
        self.norms = nn.ModuleList([RMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(6)])
        self.swa = _GQAAttention(config, sliding_window=config.sliding_window)
        self.moe = MoELayer(config)
        self.mamba = MambaMini(config)
        self.delta = DeltaNetMini(config)
        # Stack: shared_attn → SWA → MoE → Mamba-like → DeltaNet-like → shared_attn (same module as first).

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[tuple] = None,
        output_router_logits: Optional[bool] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
        del past_key_values, kwargs
        collect_r = output_router_logits if output_router_logits is not None else bool(
            getattr(self.config, "output_router_logits", False)
        )
        del use_cache
        if input_ids is not None:
            x = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Need input_ids or inputs_embeds")

        router_layers: list[torch.Tensor] = []
        modules = [self.shared_attn, self.swa, self.moe, self.mamba, self.delta, self.shared_attn]
        for i, mod in enumerate(modules):
            r = self.norms[i](x)
            if mod is self.moe and collect_r:
                y, rl = self.moe(r, return_router_logits=True)
                router_layers.append(rl)
            else:
                y = mod(r)
            x = x + y
            if attention_mask is not None:
                # zero out padded positions (broadcast [B,L,1])
                x = x * attention_mask.to(dtype=x.dtype).unsqueeze(-1)

        if collect_r:
            return (x, tuple(router_layers))
        return (x, None)


class TestEverythingForCausalLM(TestEverythingPreTrainedModel, GenerationMixin):
    """Causal head + ``GenerationMixin`` so ``generate()`` works (eval sanity / HF parity)."""

    config_class = TestEverythingConfig

    def __init__(self, config: TestEverythingConfig):
        super().__init__(config)
        self.model = TestEverythingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = float(getattr(config, "router_aux_loss_coef", 0.001))
        self.num_experts = config.num_experts
        raw_k = config.num_experts_per_tok
        if raw_k is None:
            raw_k = config.num_experts
        self.num_experts_per_tok = max(1, min(int(raw_k), config.num_experts))
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
        output_router_logits: Optional[bool] = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        del kwargs
        cfg_r = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        collect_router = bool(cfg_r) or (labels is not None and self.router_aux_loss_coef != 0)
        hidden, router_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=collect_router,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        aux_loss: Optional[torch.Tensor] = None
        if collect_router and router_logits is not None:
            lb = load_balancing_loss_func(
                router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if isinstance(lb, torch.Tensor):
                aux_loss = lb
            if labels is not None and isinstance(aux_loss, torch.Tensor):
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            router_logits=router_logits if collect_router else None,
        )
