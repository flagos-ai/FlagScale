# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

# Import model providers for easy access
from flagscale.train.bridge.models.conversion.auto_bridge import AutoBridge
from flagscale.train.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from flagscale.train.bridge.models.conversion.model_bridge import MegatronModelBridge
from flagscale.train.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from flagscale.train.bridge.models.conversion.utils import weights_verification_table

__all__ = [
    "AutoBridge",
    "MegatronMappingRegistry",
    "MegatronModelBridge",
    "ColumnParallelMapping",
    "GatedMLPMapping",
    "MegatronParamMapping",
    "QKVMapping",
    "ReplicatedMapping",
    "RowParallelMapping",
    "AutoMapping",
    "weights_verification_table",
]
