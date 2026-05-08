# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_embedding")
def register_qwen2_embedding_config():
    from .configuration_qwen2_embedding import Qwen2EmbeddingConfig

    return Qwen2EmbeddingConfig


@MODELING_REGISTRY.register("qwen2_embedding")
def register_qwen2_embedding_modeling(architecture: str):
    """Per-token binary classifier on top of a Qwen2 backbone with the last decoder block dropped.

    Reuses the upstream HF ``Qwen2Model`` backbone so pretrained Qwen2 weights load cleanly
    (matched keys: ``model.embed_tokens.*`` / ``model.layers.{0..L-2}.*`` / ``model.norm.*``;
    skipped keys: ``lm_head.*`` and ``model.layers.{L-1}.*`` — the dropped block).
    """
    from .modeling_qwen2_embedding import Qwen2EmbeddingForTokenClassification, Qwen2EmbeddingModel

    if "ForTokenClassification" in architecture or "ForBinary" in architecture:
        return Qwen2EmbeddingForTokenClassification
    if "Model" in architecture:
        return Qwen2EmbeddingModel
    return Qwen2EmbeddingForTokenClassification
