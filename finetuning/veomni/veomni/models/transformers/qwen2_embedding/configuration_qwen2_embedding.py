# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Qwen2 variant with the last decoder block removed and a per-token binary-classification head.

Same hyperparameters as ``Qwen2Config`` (so the standard pretrained checkpoints load cleanly),
plus:

- ``drop_last_n_layers``: number of trailing decoder blocks to drop. Decreases ``num_hidden_layers``
  before the inner ``Qwen2Model`` is constructed so the backbone has ``L - drop_last_n_layers``
  blocks. The dropped block weights in the pretrained state dict are reported as unexpected and
  skipped by VeOmni's ``load_model_weights``; the new ``score`` head is randomly initialized via
  ``post_process_after_weight_loading``.
- ``tie_word_embeddings`` defaults to ``False`` — there is no ``lm_head`` to tie to, and VeOmni's
  post-load tie path raises if ``get_output_embeddings()`` is None and tying is requested.
"""

from __future__ import annotations

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class Qwen2EmbeddingConfig(Qwen2Config):
    model_type = "qwen2_embedding"

    def __init__(
        self,
        drop_last_n_layers: int = 1,
        **kwargs,
    ):
        # No ``lm_head`` to tie ``embed_tokens`` to — VeOmni's ``post_process_after_weight_loading``
        # raises if tying is requested but ``get_output_embeddings()`` is None.
        kwargs.pop("tie_word_embeddings", None)
        tie_word_embeddings = False

        drop_last_n_layers = max(0, int(drop_last_n_layers))
        original_num_hidden_layers = int(kwargs.get("num_hidden_layers", 24))
        if drop_last_n_layers >= original_num_hidden_layers:
            raise ValueError(
                f"drop_last_n_layers={drop_last_n_layers} must be < num_hidden_layers={original_num_hidden_layers}"
            )

        new_num_hidden_layers = original_num_hidden_layers - drop_last_n_layers
        kwargs["num_hidden_layers"] = new_num_hidden_layers

        # Qwen2Config validates ``layer_types`` length against ``num_hidden_layers`` in
        # ``transformers>=4.56`` (``layer_type_validation``); shrink any inherited list to match.
        lt = kwargs.get("layer_types")
        if isinstance(lt, (list, tuple)) and len(lt) != new_num_hidden_layers:
            kwargs["layer_types"] = list(lt)[:new_num_hidden_layers]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.drop_last_n_layers = drop_last_n_layers
        self.original_num_hidden_layers = original_num_hidden_layers
