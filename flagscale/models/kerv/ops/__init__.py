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

"""Bundled KERV runtime optimizations.

KERV imports its optimized runtime as the top-level package
``KERVRuntimeOptimization``. FlagScale keeps that package below this module
so the model integration remains self-contained while preserving KERV's
public import contract.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def bundled_runtime_path() -> Path:
    """Return the directory that contains ``KERVRuntimeOptimization``."""

    return Path(__file__).resolve().parent


def load_embodied_ops() -> ModuleType:
    """Load the bundled operator module through KERV's canonical name."""

    runtime_path = str(bundled_runtime_path())
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    return importlib.import_module("KERVRuntimeOptimization.embodied_ops")


_EMBODIED_EXPORTS = (
    "configure_kerv_ops",
    "install_static_tree_attention",
    "kerv_action_projection_select",
    "kerv_action_verify_accept",
    "kerv_add_rms_norm",
    "kerv_down_proj_residual_rms_norm",
    "kerv_draft_action_topk",
    "kerv_kv_accept_commit",
    "kerv_kv_commit",
    "kerv_logical_kv_commit",
    "kerv_o_proj_residual_rms_norm",
    "kerv_ops_manifest",
    "kerv_rope_kv_store",
    "kerv_silu_mul",
    "kerv_static_tree_attention",
    "kerv_static_tree_pack",
    "kerv_tree_embed_pack",
    "kerv_value_cache_store",
    "kerv_verify_accept_control",
    "kerv_vision_add_layer_norm",
    "kerv_vision_bias_gelu",
    "register_kerv_ops",
    "static_tree_attention",
    "static_tree_attention_reference",
)


def __getattr__(name: str):
    if name in _EMBODIED_EXPORTS:
        return getattr(load_embodied_ops(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["bundled_runtime_path", "load_embodied_ops", *_EMBODIED_EXPORTS]
