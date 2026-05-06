# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Qwen2-0.5B–sized **depth/width/FFN** stack with **Mamba-style** sublayers in place of self-attention.

The Mamba block is the same compact conv + gated pattern as ``MambaMini`` in ``test_everything`` (not mamba_ssm).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_qwen2_mamba import Qwen2MambaConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Qwen2MambaBlock(nn.Module):
    """Compact selective SSM-style block (depthwise conv + gated linear), causal via conv trim."""

    def __init__(self, config: Qwen2MambaConfig) -> None:
        super().__init__()
        d = config.hidden_size
        di = config.mamba_d_inner
        dc = config.mamba_d_conv
        self.in_proj = nn.Linear(d, di * 2, bias=False)
        self.conv1d = nn.Conv1d(di, di, dc, groups=di, padding=dc - 1)
        self.out_proj = nn.Linear(di, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        xz = self.in_proj(x)
        x1, z = xz.chunk(2, dim=-1)
        xc = x1.transpose(1, 2)
        xc = self.conv1d(xc)[:, :, :seq].transpose(1, 2)
        y = xc * F.silu(z)
        return self.out_proj(y)


class Qwen2MambaMLP(nn.Module):
    """SiLU-gated FFN (same pattern as Qwen2)."""

    def __init__(self, config: Qwen2MambaConfig) -> None:
        super().__init__()
        h = config.hidden_size
        inter = config.intermediate_size
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MambaDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2MambaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = Qwen2MambaBlock(config)
        self.mlp = Qwen2MambaMLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.mamba(x)
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        hidden_states = residual + x

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        return hidden_states


class Qwen2MambaPreTrainedModel(PreTrainedModel):
    config_class = Qwen2MambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Qwen2MambaDecoderLayer", "RMSNorm"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class Qwen2MambaModel(Qwen2MambaPreTrainedModel):
    def __init__(self, config: Qwen2MambaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2MambaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        **kwargs,
    ) -> tuple:
        del use_cache, past_key_values, kwargs
        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("Need input_ids or inputs_embeds")

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        return (hidden_states,)


class Qwen2MambaForCausalLM(Qwen2MambaPreTrainedModel, GenerationMixin):
    config_class = Qwen2MambaConfig

    def __init__(self, config: Qwen2MambaConfig):
        super().__init__(config)
        self.model = Qwen2MambaModel(config)
        # Tied embeddings: do **not** keep a separate ``lm_head`` Parameter tied to ``embed_tokens`` —
        # that duplicates the same Parameter in ``named_parameters()`` and breaks DDP ("marked ready twice").
        # Use ``F.linear(..., embed_tokens.weight)`` in forward instead.
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        if getattr(self.config, "tie_word_embeddings", False):
            return self.model.embed_tokens
        assert self.lm_head is not None
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if getattr(self.config, "tie_word_embeddings", False):
            self.model.embed_tokens = new_embeddings
        else:
            self.lm_head = new_embeddings

    def tie_weights(self) -> None:
        # Tied case: no separate ``lm_head`` module (see ``_logits``). Untied: ``lm_head`` already distinct.
        pass

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "tie_word_embeddings", False):
            return F.linear(hidden_states, self.model.embed_tokens.weight)
        assert self.lm_head is not None
        return self.lm_head(hidden_states)

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]
        logits = self._logits(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)
