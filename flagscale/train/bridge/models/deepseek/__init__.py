# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

from flagscale.train.bridge.models.deepseek.deepseek_provider import (
    DeepSeekModelProvider,
    DeepSeekProvider,
    DeepSeekV2LiteModelProvider,
    DeepSeekV2LiteProvider,
    DeepSeekV2ModelProvider,
    DeepSeekV2Provider,
    DeepSeekV3ModelProvider,
    DeepSeekV3Provider,
    MoonlightModelProvider16B,
    MoonlightProvider,
)
from flagscale.train.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge  # noqa: F401
from flagscale.train.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge  # noqa: F401

__all__ = [
    "DeepSeekModelProvider",
    "DeepSeekV2LiteModelProvider",
    "DeepSeekV2ModelProvider",
    "DeepSeekV3ModelProvider",
    "MoonlightModelProvider16B",
    "DeepSeekProvider",
    "DeepSeekV2LiteProvider",
    "DeepSeekV2Provider",
    "DeepSeekV3Provider",
    "MoonlightProvider",
]
