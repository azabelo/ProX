# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("qwen2_mamba")
def register_qwen2_mamba_config():
    from .configuration_qwen2_mamba import Qwen2MambaConfig

    return Qwen2MambaConfig


@MODELING_REGISTRY.register("qwen2_mamba")
def register_qwen2_mamba_modeling(architecture: str):
    from .modeling_qwen2_mamba import Qwen2MambaForCausalLM, Qwen2MambaModel

    if "ForCausalLM" in architecture:
        return Qwen2MambaForCausalLM
    if "Model" in architecture:
        return Qwen2MambaModel
    return Qwen2MambaForCausalLM
