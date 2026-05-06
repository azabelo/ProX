# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Mini LLaDA-style MDM config: bidirectional transformer + discrete masking diffusion (few layers)."""

from transformers.configuration_utils import PretrainedConfig


class LladaMiniConfig(PretrainedConfig):
    """
    Similar role to `LLaDA <https://huggingface.co/GSAI-ML/LLaDA-8B-Base>`_ (encoder-style MDM with RoPE),
    reduced to a tiny stack for integration tests.

    Training uses a **masking diffusion** objective (random timestep, replace subset of tokens with
    ``mask_token_id``, predict original ids at masked positions).
    """

    model_type = "llada_mini"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        intermediate_size: int = 512,
        num_hidden_layers: int = 3,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        num_diffusion_timesteps: int = 32,
        mask_token_id: int = 151935,
        # False: tied embed/lm_head breaks DDP (same parameter used twice per forward).
        tie_word_embeddings: bool = False,
        scale_logits: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.mask_token_id = mask_token_id
        self.scale_logits = scale_logits

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
