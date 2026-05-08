# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Qwen2-backbone token-level binary classifier (drops the last decoder block).

Architecture:
- Inner ``Qwen2Model`` backbone (HF) with ``num_hidden_layers = original - drop_last_n_layers``.
- ``score`` linear head: ``hidden_size -> 1`` over per-token hidden states.
- ``forward(input_ids, attention_mask, labels)`` returns a ``TokenClassifierOutput`` with logits
  ``[B, L]`` and (when ``labels`` is provided) per-token ``BCEWithLogitsLoss`` against the same
  ``[B, L]`` labels (``-100`` positions are ignored, matching ``IGNORE_INDEX``).

Pretrained weight loading (Qwen/Qwen2.5-0.5B etc.):
- All matched keys (``model.embed_tokens.*``, ``model.layers.{0..L-2}.*``, ``model.norm.*``) are
  copied into the backbone.
- Unmatched keys (``lm_head.*``, ``model.layers.{L-1}.*``) are dropped silently by VeOmni's
  ``load_model_weights`` (logged as ``Unexpected key`` info messages).
- The new ``score.weight`` (and ``score.bias`` if present) are listed as missing parameters and
  initialized via ``post_process_after_weight_loading`` -> ``_init_parameter``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_qwen2_embedding import Qwen2EmbeddingConfig


def _import_qwen2_model() -> type[nn.Module]:
    """VeOmni's qwen2 register may apply patches; reuse the canonical class for both v4 and v5."""
    from ....utils.import_utils import is_transformers_version_greater_or_equal_to

    if is_transformers_version_greater_or_equal_to("5.0.0"):
        from ..qwen2.generated.patched_modeling_qwen2_gpu import Qwen2Model
        return Qwen2Model
    from transformers import Qwen2Model

    from ..qwen2.modeling_qwen2 import apply_veomni_qwen2_patch

    apply_veomni_qwen2_patch()
    return Qwen2Model


class Qwen2EmbeddingPreTrainedModel(PreTrainedModel):
    config_class = Qwen2EmbeddingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: nn.Module) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2EmbeddingModel(Qwen2EmbeddingPreTrainedModel):
    """Thin wrapper around the (truncated) Qwen2 backbone used by ``Qwen2EmbeddingForTokenClassification``."""

    def __init__(self, config: Qwen2EmbeddingConfig):
        super().__init__(config)
        Qwen2Model = _import_qwen2_model()
        self.model = Qwen2Model(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("use_cache", None)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


class Qwen2EmbeddingForTokenClassification(Qwen2EmbeddingPreTrainedModel):
    """Per-token binary classifier on top of a (truncated) Qwen2 backbone."""

    config_class = Qwen2EmbeddingConfig

    def __init__(self, config: Qwen2EmbeddingConfig):
        super().__init__(config)
        Qwen2Model = _import_qwen2_model()
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=True)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        kwargs.pop("use_cache", None)
        backbone_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )
        hidden = backbone_out.last_hidden_state if hasattr(backbone_out, "last_hidden_state") else backbone_out[0]
        logits = self.score(hidden).squeeze(-1)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            from veomni.utils.constants import IGNORE_INDEX

            labels = labels.to(device=logits.device)
            mask = labels.ne(IGNORE_INDEX)
            if mask.any():
                t = labels[mask].to(logits.dtype).clamp_(0.0, 1.0)
                loss = F.binary_cross_entropy_with_logits(logits[mask], t, reduction="mean")
            else:
                loss = logits.sum() * 0.0

        return TokenClassifierOutput(loss=loss, logits=logits)
