# Copyright (c) 2025, BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

from .attention import Qwen35SelfAttention
from .language_model import Qwen35LanguageModule
from .layer_specs import get_qwen35_language_model_spec
from .qwen35_model import Qwen35Model
from .rope import Qwen35LanguageRotaryEmbedding, get_rope_index
from .transformer_config import (
    Qwen35TransformerConfig,
    get_vision_model_config,
    get_vision_projection_config,
)

# Re-export vision model spec from qwen3_vl (identical vision encoder)
from flagscale.models.megatron.qwen3_vl.layer_specs import get_qwen3vl_vision_model_spec

__all__ = [
    "Qwen35Model",
    "Qwen35LanguageModule",
    "Qwen35TransformerConfig",
    "get_vision_model_config",
    "get_vision_projection_config",
    "get_qwen35_language_model_spec",
    "get_qwen3vl_vision_model_spec",
    "Qwen35SelfAttention",
    "Qwen35LanguageRotaryEmbedding",
    "get_rope_index",
]
