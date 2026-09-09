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

"""Per-forward RoPE cache for transformer stacks with identical rotary modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn


class _ForwardRotaryCache:
    def __init__(self, log_path: str | None, record: bool, record_once: bool):
        self.log_path = Path(log_path) if record and log_path else None
        self.record_once = bool(record_once)
        self.recorded = set()
        self.value = None
        self.position_ids = None
        self.dtype = None
        self.misses = 0
        self.hits = 0
        self.layer_count = 0
        self.validated = set()
        self.rejected = set()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch(exist_ok=True)

    def begin(self) -> None:
        self.value = None
        self.position_ids = None
        self.dtype = None

    def get(self, original, value: torch.Tensor, position_ids: torch.Tensor):
        if self.value is None or self.position_ids is not position_ids or self.dtype != value.dtype:
            self.value = original(value, position_ids)
            self.position_ids = position_ids
            self.dtype = value.dtype
            self.misses += 1
            return self.value

        owner = getattr(original, "__self__", None)
        validation_key = (
            id(owner),
            tuple(position_ids.shape),
            str(value.dtype),
            str(value.device),
        )
        if validation_key in self.rejected:
            return original(value, position_ids)
        if validation_key not in self.validated:
            reference = original(value, position_ids)
            exact = bool(
                len(reference) == len(self.value)
                and all(
                    torch.equal(candidate, cached)
                    for candidate, cached in zip(reference, self.value)
                )
            )
            if not exact:
                self.rejected.add(validation_key)
                return reference
            self.validated.add(validation_key)

        self.hits += 1
        if self.log_path is not None:
            key = (tuple(position_ids.shape), str(value.dtype), int(position_ids.shape[-1]))
            if not self.record_once or key not in self.recorded:
                self.recorded.add(key)
                record = {
                    "operator": "rotary_embedding_cache",
                    "implementation": "flagos_per_forward_rope_cache",
                    "position_shape": list(position_ids.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "reused_layers": max(0, self.layer_count - 1),
                    "bitwise_validated": True,
                }
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        return self.value


def _install_on_language_model(
    language_model: nn.Module,
    cache: _ForwardRotaryCache,
) -> int:
    if getattr(language_model, "_flagos_rope_cache_installed", False):
        raise RuntimeError("FlagOS RoPE cache is already installed")
    layers = list(getattr(getattr(language_model, "model", None), "layers", ()))
    rotary_modules = [
        getattr(getattr(layer, "self_attn", None), "rotary_emb", None) for layer in layers
    ]
    if len(rotary_modules) < 2 or any(module is None for module in rotary_modules):
        return 0

    reference = rotary_modules[0].inv_freq
    reference_signature = (
        tuple(reference.shape),
        reference.dtype,
        reference.device,
        getattr(rotary_modules[0], "dim", None),
        getattr(rotary_modules[0], "base", None),
        getattr(rotary_modules[0], "max_position_embeddings", None),
        getattr(rotary_modules[0], "scaling_factor", None),
    )
    for module in rotary_modules[1:]:
        inv_freq = getattr(module, "inv_freq", None)
        signature = (
            tuple(inv_freq.shape) if isinstance(inv_freq, torch.Tensor) else None,
            getattr(inv_freq, "dtype", None),
            getattr(inv_freq, "device", None),
            getattr(module, "dim", None),
            getattr(module, "base", None),
            getattr(module, "max_position_embeddings", None),
            getattr(module, "scaling_factor", None),
        )
        if not isinstance(inv_freq, torch.Tensor) or signature != reference_signature:
            return 0

    original_model_forward = language_model.model_forward

    def cached_model_forward(self, *args, **kwargs):
        cache.begin()
        return original_model_forward(*args, **kwargs)

    language_model.model_forward = MethodType(cached_model_forward, language_model)

    for rotary in rotary_modules:
        original_rotary_forward = rotary.forward

        def cached_rotary_forward(self, value, position_ids, _original=original_rotary_forward):
            return cache.get(_original, value, position_ids)

        rotary.forward = MethodType(cached_rotary_forward, rotary)

    language_model._flagos_rope_cache_installed = True
    language_model._flagos_rope_cache = cache
    cache.layer_count = max(cache.layer_count, len(rotary_modules))
    return len(rotary_modules)


def enable_rotary_cache(
    model: nn.Module,
    enabled: bool,
    record: bool,
    record_once: bool,
    log_path: str | None,
    manifest_path: str | None,
) -> dict[str, Any]:
    """Reuse identical RoPE cos/sin tensors across decoder layers in one forward."""
    if record and enabled and not log_path:
        raise ValueError("RoPE cache recording requires a log path")
    cache = _ForwardRotaryCache(log_path, record and enabled, record_once)
    targets = []
    if enabled:
        for name, module in model.named_modules():
            if type(module).__name__ == "LlamaSpecForCausalLM" and hasattr(module, "model_forward"):
                layer_count = _install_on_language_model(module, cache)
                if layer_count:
                    targets.append({"module": name, "layer_count": layer_count})
        if not targets:
            raise RuntimeError("FlagOS RoPE cache found no compatible language model")

    manifest: dict[str, Any] = {
        "enabled": bool(enabled),
        "implementation": "flagos_per_forward_rope_cache",
        "model_class": type(model).__name__,
        "target_count": len(targets),
        "targets": targets,
        "native_fallback": True,
        "inference_only": True,
    }
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[FlagOS/RoPECache] {'enabled' if enabled else 'disabled'}; targets={targets}",
        flush=True,
    )
    return manifest
