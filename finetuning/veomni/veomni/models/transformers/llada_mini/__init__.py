# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("llada_mini")
def register_llada_mini_config():
    from .configuration_llada_mini import LladaMiniConfig

    return LladaMiniConfig


@MODELING_REGISTRY.register("llada_mini")
def register_llada_mini_modeling(architecture: str):
    from .modeling_llada_mini import LladaMiniForCausalLM, LladaMiniModel

    if "ForCausalLM" in architecture:
        return LladaMiniForCausalLM
    if "Model" in architecture:
        return LladaMiniModel
    return LladaMiniForCausalLM
