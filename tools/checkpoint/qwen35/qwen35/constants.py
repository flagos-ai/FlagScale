# Copyright 2026 FlagOS Contributors
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

"""Shared constants for Qwen3.5 checkpoint conversion."""

# Layer norm adjustment for zero-centered gamma.
# Qwen3.5's HF RMSNorm stores weights in zero-centered form (scale = 1 + weight),
# which matches Megatron's layernorm_zero_centered_gamma=True format. Therefore
# no adjustment is needed by default. The old behavior can be re-enabled with
# --adjust-ln for models that store raw gamma values.
LN_ADJUSTMENT = False


# Prefix patterns that need _extra_state tensors for TE layers.
EXTRA_STATE_KEYS = {
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "mlp.experts.linear_fc1",
    "mlp.experts.linear_fc2",
    "mlp.shared_experts.linear_fc1",
    "mlp.shared_experts.linear_fc2",
    "self_attention.in_proj",
    "self_attention.out_proj",
    "self_attention.out_norm",
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "self_attention.core_attention",
    "self_attention.q_layernorm",
    "self_attention.k_layernorm",
    "pre_mlp_layernorm",
    "decoder.final_layernorm",
    "vision_model.projection.encoder.linear_fc1",
    "vision_model.projection.encoder.linear_fc2",
    "mtp.layers",
}
