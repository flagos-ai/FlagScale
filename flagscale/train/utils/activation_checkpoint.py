# Adapted from https://github.com/pytorch/torchtitan/blob/54aeddf5/torchtitan/distributed/activation_checkpoint.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation checkpointing (recomputation) utilities.

Supports:
- full: checkpoint every matched module
- selective (layer-wise): checkpoint every Nth matched module
- selective (per-op SAC): save expensive ops, recompute cheap ones
- memory_budget: compiler-driven (requires torch.compile)
"""

import re
from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as dist_checkpoint_wrapper,
)

from flagscale.train.train_config import ActivationCheckpointConfig

# Spelling matches PyTorch's internal op registration (not a typo).
_FUSED_ATTN_OP_NAME = "_scaled_dot_product_fused_attention_override" + "able"

DEFAULT_OP_SAC_SAVE_LIST: set = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    getattr(torch.ops.aten, _FUSED_ATTN_OP_NAME).default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}


def _build_fqn_map(model: nn.Module) -> dict[int, str]:
    return {id(mod): name for name, mod in model.named_modules()}


def _replace_module_by_fqn(model: nn.Module, fqn: str, new_module: nn.Module) -> None:
    parts = fqn.rsplit(".", 1)
    if len(parts) == 1:
        setattr(model, parts[0], new_module)
    else:
        parent = model.get_submodule(parts[0])
        setattr(parent, parts[1], new_module)


_KNOWN_OPAQUE_MODULE_PATTERNS = (
    "GatedDeltaNet",
    "DeltaNet",
    "LinearAttention",
    "RetNet",
    "HGRN",
    "Mamba",
)


def _warn_opaque_autograd_functions(targets: list[tuple[str, nn.Module]]) -> None:
    """Warn about modules containing ops that are opaque to per-op SAC.

    Per-op SAC operates at the ATen dispatcher level and cannot see inside
    torch.autograd.Function subclasses (e.g. FLA's chunk_gated_delta_rule,
    Mamba's selective_scan). These ops will be fully recomputed during backward,
    which may cause unexpected compute overhead.
    """
    opaque_fqns = []
    for fqn, module in targets:
        for _, child in module.named_modules():
            class_name = type(child).__name__
            if any(pat in class_name for pat in _KNOWN_OPAQUE_MODULE_PATTERNS):
                opaque_fqns.append(fqn)
                break


def apply_activation_checkpointing(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    units: list[nn.Module] | None = None,
    op_sac_save_list: set[torch._ops.OpOverload] | None = None,
    model_compile_enabled: bool = False,
) -> None:
    """Apply activation checkpointing to model layers.

    Args:
        model: The top-level model (needed for FQN resolution and module replacement).
        ac_config: ActivationCheckpointConfig with mode, patterns, etc.
        units: Default candidate modules (from fsdp_units()). Used when checkpoint_patterns is None.
        op_sac_save_list: Ops to save for per-op SAC mode. Defaults to DEFAULT_OP_SAC_SAVE_LIST.
        model_compile_enabled: Whether torch.compile is active (required for memory_budget mode).
    """
    if ac_config.mode == "none":
        return

    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires torch.compile"
        import torch._functorch.config

        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        return

    if ac_config.checkpoint_patterns is not None:
        patterns = [re.compile(p) for p in ac_config.checkpoint_patterns]
        targets = [
            (fqn, mod) for fqn, mod in model.named_modules() if any(p.search(fqn) for p in patterns)
        ]
    else:
        fqn_map = _build_fqn_map(model)
        targets = []
        for unit in units or []:
            fqn = fqn_map.get(id(unit))
            if fqn is not None:
                targets.append((fqn, unit))

    if not targets:
        return

    if op_sac_save_list is None:
        op_sac_save_list = DEFAULT_OP_SAC_SAVE_LIST

    if ac_config.mode == "selective" and ac_config.selective_ac_option == "op":
        _warn_opaque_autograd_functions(targets)

    layer_count = 0
    wrapped_count = 0
    for fqn, module in targets:
        wrapped, layer_count = _apply_ac_to_unit(
            module, ac_config, op_sac_save_list=op_sac_save_list, layer_count=layer_count
        )
        if wrapped is not module:
            _replace_module_by_fqn(model, fqn, wrapped)
            wrapped_count += 1


def _apply_ac_to_unit(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    op_sac_save_list: set[torch._ops.OpOverload],
    layer_count: int,
) -> tuple[nn.Module, int]:
    valid_modes = ("full", "selective")
    if ac_config.mode not in valid_modes:
        raise ValueError(
            f"Invalid activation checkpoint mode: '{ac_config.mode}'. "
            f"Expected one of {valid_modes} (memory_budget and none are handled earlier)."
        )

    if ac_config.mode == "full":
        return dist_checkpoint_wrapper(
            module, preserve_rng_state=ac_config.preserve_rng_state
        ), layer_count

    if ac_config.selective_ac_option == "op":
        return _apply_op_sac(module, ac_config, op_sac_save_list=op_sac_save_list), layer_count

    return _apply_layer_sac(module, ac_config, layer_count=layer_count)


def _apply_layer_sac(
    module: nn.Module, ac_config: ActivationCheckpointConfig, *, layer_count: int
) -> tuple[nn.Module, int]:
    layer_count += 1
    freq = int(ac_config.selective_ac_option)
    if not freq or layer_count % freq == 0:
        return dist_checkpoint_wrapper(
            module, preserve_rng_state=ac_config.preserve_rng_state
        ), layer_count
    return module, layer_count


def _apply_op_sac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
    *,
    op_sac_save_list: set[torch._ops.OpOverload],
) -> nn.Module:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)

        def policy_fn(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            if func == torch.ops.aten.mm.default:
                meta[f"{mode}_mm_count"] += 1
            to_save = func in op_sac_save_list and not (
                func == torch.ops.aten.mm.default and meta[f"{mode}_mm_count"] % 2 == 0
            )
            return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

        return create_selective_checkpoint_contexts(policy_fn)

    return dist_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
    )
