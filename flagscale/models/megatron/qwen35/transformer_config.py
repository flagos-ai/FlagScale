# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from flagscale.models.megatron.qwen3_vl.transformer_config
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from functools import partial

import torch

from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig


@dataclass
class Qwen35TransformerConfig(TransformerConfig):
    """
    Transformer config for Qwen3.5
    """

    # =========================================================================
    # Hybrid GDN + Attention architecture
    # =========================================================================
    experimental_attention_variant: str = "gated_delta_net"
    linear_attention_freq: int | list[int] = 4
    layernorm_zero_centered_gamma: bool = True
    attention_output_gate: bool = True

    # --- Gated DeltaNet (GDN) parameters ---
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48

    # =========================================================================
    # Common LLM parameters
    # =========================================================================
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    qk_layernorm_hidden_dim: bool = False
    peft_type: str = None
    kv_channels: int | None = 256
    num_query_groups: int = 4
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25
    seq_length: int = 262144

    # =========================================================================
    # VL-specific parameters
    # =========================================================================
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    apply_rotary_pos_emb_in_fp32: bool = False
    apply_rope_fusion: bool = False

    # Vision-Language token IDs (248xxx series for Qwen3.5)
    vision_start_token_id: int = 248053
    image_token_id: int = 248056
    video_token_id: int = 248057
    bos_token_id: int = 248045
    eos_token_id: int = 248044

    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    patch_size: int = 16
    in_channels: int = 3
    num_position_embeddings: int = 2304
    language_max_sequence_length: int = 2048
    scatter_embedding_sequence_parallel: bool = False

    # Vision-specific (shared with Qwen3-VL)
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False


def get_vision_model_config(args, config):
    """Build vision encoder config from language transformer config."""
    assert parallel_state.get_virtual_pipeline_model_parallel_world_size() is None, "NotSupported"

    # Qwen3.5 vision encoder params from args (varies by model size)
    config.num_layers = args.vision_num_layers
    config.hidden_size = args.vision_hidden_size
    config.ffn_hidden_size = args.vision_ffn_hidden_size
    config.deepstack_visual_indexes = []

    config.num_attention_heads = args.vision_num_attention_heads
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    config.gated_linear_unit = False
    config.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = "LayerNorm"
    config.seq_length = args.seq_length

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = 16
    config.in_channels = 3
    config.spatial_merge_size = 2
    config.num_position_embeddings = 2304

    # Disable pipeline parallelism in vision model
    config.pipeline_model_parallel_size = 1
    config.first_pipeline_num_layers = None
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None

    # Reset recompute settings for vision encoder
    if args.vision_recompute_activations:
        config.recompute_granularity = "full"
        config.recompute_method = "uniform"
        config.recompute_num_layers = 1

    return config


def get_vision_projection_config(config, embed_dim, spatial_merge_size):
    """Build vision projection (patch merger) config."""
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = embed_dim * (spatial_merge_size**2)
    config.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    return config
