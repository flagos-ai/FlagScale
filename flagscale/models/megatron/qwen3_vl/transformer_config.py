
# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/main/megatron_patch/model/qwen3_vl/transformer_config.py
#
# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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
import torch

import functools
from typing import List
from dataclasses import dataclass, field
from megatron.core.transformer import TransformerConfig
from megatron.core import parallel_state


@dataclass
class Qwen3VLTransformerConfig(TransformerConfig):

    transformer_impl: str = 'transformer_engine'
    rotary_base: int = None
    rotary_scaling_factor: int = None
    max_position_embeddings: int = None
    moe_aux_loss_coeff: float = 0.0
    num_position_embeddings: int = 0
    deepstack_visual_indexes: List = field(default_factory=lambda: [8, 16, 24])

    vision_start_token_id: int = 151652
    image_token_id: int = 151655
    video_token_id: int = 151656
    spatial_merge_size: int = 2

    # The following options are set with --disable-bias-linear --add-qkv-bias
    # in the script
    # add_bias_linear = False
    # add_qkv_bias = True



def get_vision_model_config(args, config):
    # Given a Transformer Config from decoder, build vision encoder config
    # diff: out_hidden_size & intermediate_size
    # mlp: hidden_size -> intermediate_size -> embed_dim, silu

    assert parallel_state.get_virtual_pipeline_model_parallel_world_size() is None, "NotSupported"
    if args.num_layers == 36 and args.hidden_size == 2560:
        # 4B
        config.num_layers = 24 # depth
        config.hidden_size = 1024 # hidden_size
        config.ffn_hidden_size = 4096
        config.deepstack_visual_indexes=[5, 11, 17]

    else:
        # 8B, 32B, 30BA3B, 235BA22B
        config.num_layers = 27 # depth
        config.hidden_size = 1152 # hidden_size
        config.ffn_hidden_size = 4304
        config.deepstack_visual_indexes=[8, 16, 24]

    config.num_attention_heads = 16 # num_heads
    config.add_bias_linear = True # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True # qkv_proj in attn has bias
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    config.gated_linear_unit = False # no gated
    # NOTE(lizhyu): different from Qwen2.5-VL, use GELU here vs "gelu_pytorch_tanh" in huggingface implementation.
    config.activation_func = functools.partial(torch.nn.functional.gelu, approximate="tanh") # Using GELU activation not SiLU
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads # no GQA
    config.layernorm_zero_centered_gamma = False # False
    config.apply_query_key_layer_scaling = False # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False # no swiglu, set false
    config.bias_dropout_fusion = False # no dropout, set false
    config.attention_softmax_in_fp32 = True # use True
    config.normalization = 'LayerNorm' # use LayerNorm
    config.seq_length = args.seq_length

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = 16
    config.in_channels = 3
    config.spatial_merge_size = 2
    config.num_position_embeddings = 2304

    # disable pipeline parallelism in vision model
    config.pipeline_model_parallel_size = 1
    config.first_pipeline_num_layers = None
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None

    # Reset recompute settings for vision encoder, otherwise it may use the language's settings.
    if args.vision_recompute_activations:
        config.recompute_granularity = 'full'
        config.recompute_method = 'uniform'
        config.recompute_num_layers = 1

    return config


def get_vision_projection_config(config, embed_dim, spatial_merge_size):
    # NOTE: vision projection MLP not for deepstack projector.
    # merger:
    # context_dim = hidden_size * merge_size**2
    # out_hidden_size = hidden_size
    # context_dim -> context_dim -> out_hidden_size
    # MLP:
    # input_size -> ffn_hidden_size -> hidden_size
    # spec: LN -> Linear(bias=True) -> GELU -> Linear(bias=True)
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = embed_dim * (spatial_merge_size ** 2)
    config.activation_func = torch.nn.GELU() # Using GELU activation not SiLU
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    return config
