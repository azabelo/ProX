# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("test_everything")
def register_test_everything_config():
    from .configuration_test_everything import TestEverythingConfig

    return TestEverythingConfig


@MODELING_REGISTRY.register("test_everything")
def register_test_everything_modeling(architecture: str):
    from .modeling_test_everything import TestEverythingForCausalLM, TestEverythingModel

    if "ForCausalLM" in architecture:
        return TestEverythingForCausalLM
    if "Model" in architecture:
        return TestEverythingModel
    return TestEverythingForCausalLM
