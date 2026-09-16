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

"""KERV-specific runtime operators.

The package deliberately keeps the public surface small.  Every operator has
an explicit native fallback and can be selected by the FlagScale runtime
without changing the model's numerical control flow.
"""

from .kerv_ops import (
    kerv_action_projection_select,
    kerv_add_rms_norm,
    kerv_kv_commit,
    kerv_silu_mul,
    kerv_static_tree_pack,
    kerv_value_cache_store,
    kerv_verify_accept_control,
    register_kerv_ops,
)
from .phase2_ops import (
    kerv_action_verify_accept,
    kerv_down_proj_residual_rms_norm,
    kerv_draft_action_topk,
    kerv_kv_accept_commit,
    kerv_logical_kv_commit,
    kerv_o_proj_residual_rms_norm,
    kerv_rope_kv_store,
    kerv_tree_embed_pack,
    kerv_vision_add_layer_norm,
    kerv_vision_bias_gelu,
)
from .runtime import configure_kerv_ops, kerv_ops_manifest
from .static_tree_attention import (
    install_static_tree_attention,
    static_tree_attention,
    static_tree_attention_reference,
)

# Python-level spelling mirrors the public torch operator name.
kerv_static_tree_attention = static_tree_attention

__all__ = [
    "configure_kerv_ops",
    "kerv_action_projection_select",
    "kerv_action_verify_accept",
    "kerv_add_rms_norm",
    "kerv_draft_action_topk",
    "kerv_down_proj_residual_rms_norm",
    "kerv_kv_commit",
    "kerv_kv_accept_commit",
    "kerv_logical_kv_commit",
    "kerv_ops_manifest",
    "kerv_o_proj_residual_rms_norm",
    "kerv_rope_kv_store",
    "kerv_silu_mul",
    "kerv_static_tree_pack",
    "kerv_tree_embed_pack",
    "kerv_vision_add_layer_norm",
    "kerv_vision_bias_gelu",
    "static_tree_attention",
    "kerv_static_tree_attention",
    "install_static_tree_attention",
    "kerv_value_cache_store",
    "kerv_verify_accept_control",
    "register_kerv_ops",
    "static_tree_attention_reference",
]
