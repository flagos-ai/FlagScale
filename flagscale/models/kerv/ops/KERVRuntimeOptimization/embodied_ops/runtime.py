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

"""Configuration and hit accounting for KERV FlagOS operators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from .kerv_ops import (
    configure_backend,
    configure_enabled_operators,
    configure_recording,
    register_kerv_ops,
)
from .static_tree_attention import (
    configure_backend as configure_attention_backend,
    configure_enabled as configure_attention_enabled,
    register_static_tree_attention,
)

_PHASE2_MAINLINE = (
    "kerv_tree_embed_pack",
    "kerv_rope_kv_store",
    "kerv_action_verify_accept",
    "kerv_draft_action_topk",
    "kerv_kv_accept_commit",
    "kerv_vision_add_layer_norm",
    "kerv_vision_bias_gelu",
)

_PHASE2_EXPERIMENTAL = (
    "kerv_o_proj_residual_rms_norm",
    "kerv_down_proj_residual_rms_norm",
    "kerv_logical_kv_commit",
)


_DEFAULT_OPERATORS = (
    "kerv_silu_mul",
    "kerv_add_rms_norm",
    "kerv_verify_accept_control",
    "kerv_static_tree_pack",
    "kerv_static_tree_attention",
    "kerv_action_projection_select",
    "kerv_kv_commit",
    *_PHASE2_MAINLINE,
)


def _normalise_include(include: Iterable[str] | str | None) -> list[str]:
    if include is None:
        return list(_DEFAULT_OPERATORS)
    if isinstance(include, str):
        values = [item.strip() for item in include.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in include if str(item).strip()]
    return list(dict.fromkeys(values))


def kerv_ops_manifest(
    *,
    enabled: bool,
    include: Iterable[str] | str | None = None,
    backend: str = "auto",
    record: bool = False,
    record_once: bool = True,
    log_path: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    operators = _normalise_include(include)
    known = set(_DEFAULT_OPERATORS) | {"kerv_value_cache_store"} | set(_PHASE2_EXPERIMENTAL)
    unknown = sorted(set(operators) - known)
    if strict and unknown:
        raise ValueError(f"unknown KERV embodied operators: {unknown}")
    return {
        "schema_version": 1,
        "enabled": bool(enabled),
        "backend": str(backend),
        "requested_operators": operators,
        "record": bool(record),
        "record_once": bool(record_once),
        "log_path": log_path,
        "strict": bool(strict),
        "native_fallback": True,
        "operator_namespace": "flagos_embodied",
        "mainline_operators": list(_PHASE2_MAINLINE),
        "experimental_operators": list(_PHASE2_EXPERIMENTAL),
        "performance_gate": {"min_local_speedup": 1.05, "e2e_target": 1.08},
    }


def configure_kerv_ops(
    *,
    enabled: bool = False,
    include: Iterable[str] | str | None = None,
    backend: str = "auto",
    record: bool = False,
    record_once: bool = True,
    log_path: str | None = None,
    manifest_path: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Install the public namespace and write an auditable configuration."""

    manifest = kerv_ops_manifest(
        enabled=enabled,
        include=include,
        backend=backend,
        record=record,
        record_once=record_once,
        log_path=log_path,
        strict=strict,
    )
    configure_recording(log_path, bool(enabled and record), bool(record_once))
    configure_backend(backend)
    configure_attention_backend(backend)
    requested = set(manifest["requested_operators"]) if enabled else set()
    configure_enabled_operators(requested)
    configure_attention_enabled("kerv_static_tree_attention" in requested)
    if enabled:
        register_kerv_ops()
        # register_kerv_ops extends the namespace with phase2_ops.  Importing
        # explicitly here also covers applications that import runtime before
        # the kerv_ops module has been eagerly loaded.
        from .phase2_ops import register_phase2_ops

        register_phase2_ops()
        register_static_tree_attention()
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if enabled:
        print(
            "[FlagOS/KERV] embodied operators enabled: "
            + ", ".join(manifest["requested_operators"]),
            flush=True,
        )
    return manifest
