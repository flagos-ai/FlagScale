# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Configuration utilities for the Bagel VLM (Vision-Language Model with MoT).

Bagel uses Qwen2.5-7B as the language backbone with Mixture-of-Transformers (MoT)
layers for unified understanding and generation. The vision encoder is SigLIP ViT.
"""

from typing import Optional
from functools import partial

import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.mot import get_mot_layer_with_transformer_engine_spec


def get_qwen_language_model_config(
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig tuned for Qwen2.5-7B.

    The hyper-parameters follow the Qwen2.5-7B architecture used as the
    language backbone in Bagel.
    """
    cfg = TransformerConfig(num_layers=28, hidden_size=3584, num_attention_heads=28)

    # Feed-forward / MLP hidden size.
    cfg.ffn_hidden_size = 18944

    # SwiGLU (SiLU-gate) activation.
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    # Normalisation – RMSNorm.
    cfg.normalization = "RMSNorm"
    cfg.layernorm_epsilon = 1e-6

    # Positional embeddings – RoPE.
    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 1000000
    cfg.rotary_percent = 1.0

    # Sequence length.
    cfg.seq_length = 30720
    cfg.max_position_embeddings = 32768

    # Attention / dropout.
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    # GQA: 4 KV heads.
    cfg.num_query_groups = 4

    # Bias usage – Qwen2.5 uses QKV bias but no bias in other linear layers.
    cfg.add_bias_linear = False
    cfg.add_qkv_bias = True

    # Weight sharing.
    cfg.untie_embeddings_and_output_weights = True

    # Kernel / TE fusions.
    cfg.bias_activation_fusion = False
    cfg.masked_softmax_fusion = False
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = False
    cfg.apply_rope_fusion = False

    # QK layernorm for MoT.
    cfg.qk_layernorm = True

    # Attention precision.
    cfg.attention_softmax_in_fp32 = True

    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


def get_vision_model_config(
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for SigLIP ViT vision encoder.

    Uses the penultimate layer output (27 - 1 = 26 layers).
    """
    cfg = TransformerConfig(num_layers=26, hidden_size=1152, num_attention_heads=16)

    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.ffn_hidden_size = 4304
    cfg.gated_linear_unit = False
    cfg.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    cfg.kv_channels = 72
    cfg.num_query_groups = 16
    cfg.layernorm_zero_centered_gamma = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    # Original Bagel ViT checkpoint uses learned absolute positions, not 2D RoPE.
    cfg.rope = False
    cfg.apply_rope_fusion = False
    cfg.qk_layernorm = False
    cfg.layernorm_epsilon = 1e-6

    # Disable recompute for vision encoder by default.
    cfg.recompute_method = None
    cfg.recompute_granularity = None
    cfg.recompute_num_layers = None

    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


def get_bagel_projection_config(
    hidden_size: int = 3584,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP.

    Projects from ViT hidden size (1152) to language model hidden size (3584).
    The projection uses a 2-layer MLP with GELU_tanh activation.
    """
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = hidden_size
    cfg.bias_activation_fusion = False
    # The released BAGEL connector checkpoint contains both FC biases.
    cfg.add_bias_linear = True
    cfg.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    cfg.gated_linear_unit = False

    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg


def get_bagel_language_layer_spec(qk_layernorm: bool = True) -> ModuleSpec:
    """Layer spec for the Bagel MoT language model (Transformer-Engine).

    Uses the MoT (Mixture-of-Transformers) layer spec with separate
    understanding and generation branches.
    """
    return get_mot_layer_with_transformer_engine_spec(
        branch_names=["und", "gen"],
        qk_layernorm=qk_layernorm,
    )


def get_bagel_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision-projection MLP."""
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
