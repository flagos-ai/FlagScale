# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
""" Configuration for the vision tower, the llm tower and the projector. """
import torch

from megatron.core.activations import quick_gelu, squared_relu, fast_gelu


def get_language_model_config(config):
    """Get the language model config."""
    # qwen2.5 is the same as qwen2 on the model side
    if config.language_model_type == "qwen2.5_7b":
        config.language_model_type = "qwen2_7b"
    elif config.language_model_type == "qwen2.5_1.5b":
        config.language_model_type = "qwen2_1.5b"

    # Add qwen2_7b config
    if config.language_model_type == "qwen2_7b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = (
            False  # linear_qkv has bias but linear_proj and MLP has no bias
        )
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True

    # Add qwen2_1.5b config
    elif config.language_model_type == "qwen2_1.5b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = (
            False  # linear_qkv has bias but linear_proj and MLP has no bias
        )
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
    else:
        raise ValueError(f"Unknown language model type: {config.language_model_type}")

    return config


def get_vision_model_config(config, apply_query_key_layer_scaling):
    """Get the vision model config."""
    if config.vision_model_type == "clip":
        config.num_layers = 24
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1024
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 4096
        config.gated_linear_unit = False
        config.activation_func = quick_gelu
        config.kv_channels = 64
        config.num_attention_heads = 16
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = "LayerNorm"
        config.apply_rope_fusion = False
    # Add siglip config
    elif config.vision_model_type == "siglip":
        # Select the output of the penultimate layer, not the last.
        # So the num layer is the raw number 27 - 1
        config.num_layers = 26
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1152
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 4304
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.kv_channels = 72
        config.num_attention_heads = 16
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = "LayerNorm"
        config.apply_rope_fusion = False
        config.layernorm_epsilon = 1e-6
        # This is the temporary setting of recompute for the siglip model
        config.recompute_method = None
        config.recompute_granularity = None
        config.recompute_num_layers = None
    else:
        raise ValueError(f"Unknown vision model type: {config.vision_model_type}")
    return config


def get_vision_projection_config(config, hidden_size):
    """Get the vision projection config."""
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.hidden_size = hidden_size  # Used as the vision projection output size, i.e., the input to the language model.
    if config.language_model_type == "qwen2.5_7b":
        config.language_model_type = "qwen2_7b"
    elif config.language_model_type == "qwen2.5_1.5b":
        config.language_model_type = "qwen2_1.5b"

    if config.language_model_type == "qwen2_7b":
        config.ffn_hidden_size = 3584
        config.add_bias_linear = True
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "qwen2_1.5b":
        config.ffn_hidden_size = 1536
        config.add_bias_linear = True
        config.activation_func = torch.nn.functional.gelu
    else:
        raise ValueError(f"Unknown language model type: {config.language_model_type}")
    return config
