# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Qwen2-shaped causal LM where each decoder layer uses a Mamba-style SSM block instead of attention."""

from __future__ import annotations

from typing import Optional

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class Qwen2MambaConfig(Qwen2Config):
    """
    Hyperparameters match ``Qwen2Config`` (e.g. Qwen2 / Qwen2.5-0.5B width/depth/FFN), but the model
    implementation replaces multi-head self-attention with a compact Mamba-like block (see modeling).
    """

    model_type = "qwen2_mamba"

    def __init__(
        self,
        mamba_d_inner: Optional[int] = None,
        mamba_d_conv: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        h = int(self.hidden_size)
        self.mamba_d_inner = int(mamba_d_inner) if mamba_d_inner is not None else h * 2
        self.mamba_d_conv = int(mamba_d_conv)
