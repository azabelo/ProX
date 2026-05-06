# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Qwen2 variant: explicit per-layer sliding-window attention via ``layer_types``."""

from __future__ import annotations

from typing import List, Optional, Union

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class Qwen2SWAConfig(Qwen2Config):
    """
    Same architecture as ``Qwen2Config``, but layers listed in ``swa_layer_indices`` use
    sliding-window attention (``sliding_attention``); all others use full causal attention.

    Hugging Face ``Qwen2Attention`` picks SWA vs full using ``config.layer_types[layer_idx]``
    and ``config.sliding_window`` (see ``transformers`` Qwen2 modeling).
    """

    model_type = "qwen2_swa"

    def __init__(
        self,
        swa_layer_indices: Optional[Union[List[int], tuple]] = None,
        swa_sliding_window: int = 4096,
        **kwargs,
    ):
        swa_layer_indices = list(swa_layer_indices or [])
        swa_layer_indices = sorted({int(i) for i in swa_layer_indices})

        num_hidden_layers = int(kwargs.get("num_hidden_layers", 32))
        for idx in swa_layer_indices:
            if idx < 0 or idx >= num_hidden_layers:
                raise ValueError(
                    f"Each index in swa_layer_indices must be in [0, {num_hidden_layers - 1}], got {idx}"
                )

        layer_types = [
            "sliding_attention" if i in swa_layer_indices else "full_attention"
            for i in range(num_hidden_layers)
        ]

        use_sw = len(swa_layer_indices) > 0
        kwargs["layer_types"] = layer_types
        kwargs["use_sliding_window"] = use_sw
        if use_sw:
            kwargs["sliding_window"] = int(swa_sliding_window)

        super().__init__(**kwargs)

        self.swa_layer_indices = swa_layer_indices
        self.swa_sliding_window = int(swa_sliding_window)
