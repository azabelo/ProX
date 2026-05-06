# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tiny configurable LM used to exercise VeOmni ``MODEL_CONFIG_REGISTRY`` / ``MODELING_REGISTRY``."""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class TestEverythingConfig(PretrainedConfig):
    """
    Decoder LM with a fixed stack:
    full attention → sliding-window attention → MoE FFN → Mamba-like SSM → DeltaNet-like recurrence
    → full attention (weight-tied with the first attention block).

    Intended for integration tests only (random init, small ``hidden_size``).
    """

    model_type = "test_everything"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 256,
        # ``GenerationMixin`` / cache code paths expect a layer count on causal LMs.
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 512,
        sliding_window: int = 32,
        num_experts: int = 4,
        num_experts_per_tok: Optional[int] = None,
        norm_topk_prob: bool = False,
        mamba_d_inner: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        delta_kernel_size: int = 3,
        rope_theta: float = 10_000.0,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        # Same MoE training signals as ``Qwen3MoeConfig`` (Switch-style load balancing aux loss).
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.mamba_d_inner = mamba_d_inner
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.delta_kernel_size = delta_kernel_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
