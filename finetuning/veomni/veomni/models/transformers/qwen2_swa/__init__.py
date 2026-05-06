# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_swa")
def register_qwen2_swa_config():
    from .configuration_qwen2_swa import Qwen2SWAConfig

    return Qwen2SWAConfig


@MODELING_REGISTRY.register("qwen2_swa")
def register_qwen2_swa_modeling(architecture: str):
    """Reuse stock Qwen2 weights and VeOmni Qwen2 patches; only config differs (layer_types / SWA)."""
    if is_transformers_version_greater_or_equal_to("5.0.0"):
        from ..qwen2.generated.patched_modeling_qwen2_gpu import (
            Qwen2ForCausalLM,
            Qwen2ForSequenceClassification,
            Qwen2Model,
        )
    else:
        from transformers import Qwen2ForCausalLM, Qwen2ForSequenceClassification, Qwen2Model

        from ..qwen2.modeling_qwen2 import apply_veomni_qwen2_patch

        apply_veomni_qwen2_patch()

    if "ForCausalLM" in architecture:
        return Qwen2ForCausalLM
    if "ForSequenceClassification" in architecture:
        return Qwen2ForSequenceClassification
    if "Model" in architecture:
        return Qwen2Model
    return Qwen2ForCausalLM
