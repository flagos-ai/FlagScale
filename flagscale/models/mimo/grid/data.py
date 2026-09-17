# Copyright (c) 2026, BAAI. All rights reserved.

"""Rank-aware data-loading utilities for MIMO (FlagScale-native).

Port of the Megatron-Bridge ``data/megatron_mimo/dp_utils.py`` concepts with
no Bridge or distributed dependency: sampling info, data-needed-by-role
decisions, and DP slicing of a global micro-batch into module-local shards.
Pure functions of explicit dataclasses and tensors, unit-testable on CPU
without process groups.

Conventions:

- Every data-loading rank samples the same global micro-batch
  (``sampler_dp_size=1``); per-module DP sub-sharding is deferred to
  :func:`slice_batch_for_module_dp` in the forward step, matching the
  MIMO bridge's contiguous batch-dimension split/concatenate routing.
- Multimodal MRoPE ``position_ids`` are ``[3, batch, seq]``; their
  batch dimension is 1 (:func:`_batch_dim_for_tensor`).
- Patch-packed visual inputs (``{hidden_states, grid_thw, ...}``) use dim 0
  for different units across fields (patches vs images) and are sliced
  jointly (:func:`is_patch_packed_visual_dict`).
- Language-only ranks (non-colocated layouts) consume encoder outputs from
  the MIMO bridge; raw modality inputs are dropped before DP slicing
  (:func:`should_drop_modality_inputs`).
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

from .parallelism import (
    LANGUAGE_MODULE_NAME,
    MIMOLayout,
    MIMOParallelismConfig,
    classify_layout,
)


@dataclass(frozen=True)
class ModuleDataRole:
    """Rank-aware role describing what data a rank must load.

    ``module_name`` names the module whose data this rank loads; ``pp_rank`` /
    ``pp_size`` locate the rank within that module's pipeline group (modality
    modules are never pipelined: ``pp_size=1, pp_rank=0``).
    """

    module_name: str
    pp_rank: int = 0
    pp_size: int = 1

    def __post_init__(self) -> None:
        assert isinstance(self.module_name, str) and self.module_name, (
            f"module_name must be a non-empty string, got {self.module_name!r}."
        )
        assert isinstance(self.pp_size, int) and not isinstance(self.pp_size, bool), (
            f"pp_size must be an integer, got {self.pp_size!r}."
        )
        assert self.pp_size >= 1, f"pp_size must be >= 1, got {self.pp_size}."
        assert isinstance(self.pp_rank, int) and not isinstance(self.pp_rank, bool), (
            f"pp_rank must be an integer, got {self.pp_rank!r}."
        )
        assert 0 <= self.pp_rank < self.pp_size, (
            f"pp_rank {self.pp_rank} out of range [0, {self.pp_size})."
        )

    @property
    def is_language(self) -> bool:
        """Whether this role belongs to the (mandatory) language module."""
        return self.module_name == LANGUAGE_MODULE_NAME

    @property
    def is_first_stage(self) -> bool:
        """Whether this role sits on the module's first PP stage."""
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        """Whether this role sits on the module's last PP stage."""
        return self.pp_rank == self.pp_size - 1


def needs_data_for_role(role: ModuleDataRole) -> bool:
    """Decide whether a rank must load a data batch for its module.

    Language: every PP stage needs batch metadata - first stages consume
    ``input_ids``, last stages consume ``labels``/``loss_mask``, and MRoPE-style
    models also need ``position_ids`` on intermediate PP stages. Modality
    modules: only the first PP stage needs raw modality inputs.
    """
    if role.is_language:
        return True
    return role.is_first_stage


@dataclass(frozen=True)
class SamplingInfo:
    """Sampler-level DP settings for a data-loading rank.

    Every data-loading rank samples with ``sampler_dp_size=1`` so all ranks
    load identical global micro-batches; module-local DP slicing is deferred
    to :func:`slice_batch_for_module_dp` in the forward step. Do **not** use
    these values to construct a ``DistributedSampler`` for per-module
    sharding - they are deliberately un-sharded at the sampler level.
    """

    sampler_dp_rank: int = 0
    sampler_dp_size: int = 1
    needs_data: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.sampler_dp_size, int) and not isinstance(
            self.sampler_dp_size, bool
        ), f"sampler_dp_size must be an integer, got {self.sampler_dp_size!r}."
        assert self.sampler_dp_size >= 1, (
            f"sampler_dp_size must be >= 1, got {self.sampler_dp_size}."
        )
        assert isinstance(self.sampler_dp_rank, int) and not isinstance(
            self.sampler_dp_rank, bool
        ), f"sampler_dp_rank must be an integer, got {self.sampler_dp_rank!r}."
        assert 0 <= self.sampler_dp_rank < self.sampler_dp_size, (
            f"sampler_dp_rank {self.sampler_dp_rank} out of range [0, {self.sampler_dp_size})."
        )


def get_sampling_info(role: ModuleDataRole) -> SamplingInfo:
    """Return the sampler settings for a rank's module role.

    All data-loading ranks share ``sampler_dp_rank=0, sampler_dp_size=1`` so
    they stay synchronised on the same sample order; the per-module
    sub-sharding happens later in :func:`slice_batch_for_module_dp`.
    """
    return SamplingInfo(
        sampler_dp_rank=0,
        sampler_dp_size=1,
        needs_data=needs_data_for_role(role),
    )


def _batch_dim_for_tensor(key: str, value: torch.Tensor) -> int:
    """Return the batch dimension for a known MIMO batch tensor.

    Multimodal MRoPE ``position_ids`` are ``[3, batch, seq]``, so their
    batch dimension is 1.
    """
    if key == "position_ids" and value.dim() >= 3 and value.size(0) == 3:
        return 1
    return 0


def is_patch_packed_visual_dict(value: Any) -> bool:
    """Detect a patch-packed visual encoder input layout.

    Some VLM adapters pack all image patches of a microbatch into one flat
    tensor plus per-image grid metadata: ``hidden_states`` is
    ``[sum(patches_across_images), patch_feature_dim]`` and ``grid_thw`` is
    ``[num_images, 3]``. Dim 0 of the two tensors means different things
    (patches vs images), so they cannot be DP-sliced independently; they must
    be sliced jointly along per-image boundaries.
    """
    return (
        isinstance(value, dict)
        and isinstance(value.get("hidden_states"), torch.Tensor)
        and isinstance(value.get("grid_thw"), torch.Tensor)
        and value["hidden_states"].dim() >= 1
        and value["grid_thw"].dim() == 2
        and value["grid_thw"].size(-1) == 3
    )


def _slice_patch_packed_visual_dict(
    value: dict[str, Any], dp_rank: int, dp_size: int
) -> dict[str, Any]:
    """Joint-slice a patch-packed ``{hidden_states, grid_thw, ...}`` dict.

    Shards by image count, then derives the patch range from
    ``cumsum(grid_thw.prod(dim=-1))``; other keys pass through unchanged as
    global encoder metadata.

    Constraints:

    - ``num_images`` (rows of ``grid_thw``) must be divisible by ``dp_size``.
      With single-image-per-sample data and the MIMO MBS divisible by the
      encoder DP, this always holds.
    - ``hidden_states`` dim 0 must equal ``sum(grid_thw.prod(dim=-1))``.
    """
    assert dp_size >= 1, f"dp_size must be >= 1, got {dp_size}."
    g = value["grid_thw"]  # [num_images, 3]
    hs = value["hidden_states"]  # [sum(patches), feat]
    n_images = int(g.size(0))
    assert n_images % dp_size == 0, (
        f"Patch-packed visual input has num_images_in_microbatch ({n_images}) "
        f"not divisible by encoder DP ({dp_size}). Set the MIMO "
        f"MICRO_BATCH_SIZE so that each encoder DP shard receives a whole "
        f"number of images."
    )
    imgs_per_shard = n_images // dp_size
    img_lo = dp_rank * imgs_per_shard
    img_hi = img_lo + imgs_per_shard

    patches_per_image = g.prod(dim=-1).to(torch.long)  # [num_images]
    total_patches = int(patches_per_image.sum().item())
    assert int(hs.size(0)) == total_patches, (
        f"Patch-packed visual input expected hidden_states dim 0 ({hs.size(0)}) "
        f"to equal sum(grid_thw products) ({total_patches})."
    )
    patch_offsets = torch.zeros(n_images + 1, dtype=torch.long, device=patches_per_image.device)
    patch_offsets[1:] = patches_per_image.cumsum(0)
    patch_lo = int(patch_offsets[img_lo].item())
    patch_hi = int(patch_offsets[img_hi].item())

    out: dict[str, Any] = {}
    for key, sub_value in value.items():
        if key == "grid_thw":
            out[key] = g[img_lo:img_hi].contiguous()
        elif key == "hidden_states":
            out[key] = hs[patch_lo:patch_hi].contiguous()
        else:
            # Other entries pass through as global metadata.
            out[key] = sub_value
    return out


def slice_batch_for_module_dp(
    batch: Mapping[str, Any],
    dp_rank: int,
    dp_size: int,
) -> dict[str, Any]:
    """Slice a global micro-batch for this rank's module-local DP shard.

    All data-loading ranks receive the same global micro-batch (the sampler
    uses ``sampler_dp_size=1``); the slicing is contiguous to match the MIMO
    bridge's batch-dimension split/concatenate fan-out and fan-in routing.
    Nested dicts (e.g. ``modality_inputs``) are recursed into; patch-packed
    ``{hidden_states, grid_thw}`` visual inputs are joint-sliced (see
    :func:`is_patch_packed_visual_dict`).

    Args:
        batch: Global batch dict with tensors of shape ``[global_batch, ...]``,
            except known layouts such as multimodal MRoPE ``position_ids``
            shaped ``[3, global_batch, seq]``.
        dp_rank: This rank's position in its module-local DP group.
        dp_size: Size of the module-local DP group.

    Returns:
        Dict with tensors sliced to shape ``[global_batch // dp_size, ...]``.
    """
    assert isinstance(dp_size, int) and not isinstance(dp_size, bool) and dp_size >= 1, (
        f"dp_size must be a positive integer, got {dp_size!r}."
    )
    assert isinstance(dp_rank, int) and not isinstance(dp_rank, bool), (
        f"dp_rank must be an integer, got {dp_rank!r}."
    )
    assert 0 <= dp_rank < dp_size, f"dp_rank {dp_rank} out of range [0, {dp_size})."
    if dp_size == 1:
        # No sharding: return a copy so the function never aliases its input.
        return dict(batch)

    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_dim = _batch_dim_for_tensor(key, value)
            batch_size = value.size(batch_dim)
            assert batch_size % dp_size == 0, (
                f"Batch size {batch_size} for key '{key}' is not divisible by "
                f"DP size {dp_size}. Ensure micro_batch_size is divisible by "
                f"every module's data_parallel_size."
            )
            local_batch_size = batch_size // dp_size
            start_idx = dp_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            index = [builtins.slice(None)] * value.dim()
            index[batch_dim] = builtins.slice(start_idx, end_idx)
            sliced[key] = value[tuple(index)]
        elif isinstance(value, dict):
            # Patch-packed visual inputs use dim 0 for different units
            # across fields (patches vs images), so slice jointly.
            if is_patch_packed_visual_dict(value):
                sliced[key] = _slice_patch_packed_visual_dict(value, dp_rank, dp_size)
            else:
                # Recurse into nested dicts (e.g. modality_inputs).
                sliced[key] = slice_batch_for_module_dp(value, dp_rank, dp_size)
        elif isinstance(value, list) and len(value) > 0:
            list_len = len(value)
            if list_len % dp_size == 0:
                local_len = list_len // dp_size
                start_idx = dp_rank * local_len
                end_idx = start_idx + local_len
                sliced[key] = value[start_idx:end_idx]
            else:
                # Keep as-is if not evenly divisible (global metadata).
                sliced[key] = value
        else:
            # Keep non-tensor, non-list values as-is.
            sliced[key] = value

    return sliced


def should_drop_modality_inputs(
    role: ModuleDataRole,
    config: MIMOParallelismConfig,
    world_size: int,
) -> bool:
    """Decide whether this rank must drop raw modality inputs from its batch.

    In a non-colocated layout, language-only ranks consume encoder outputs
    from the MIMO bridge instead of raw modality inputs, so the raw inputs
    are dropped before DP slicing (their leading dimension is not the
    language sample batch). Colocated language ranks also host the vision
    module and keep the raw inputs.
    """
    assert role.is_language, (
        f"should_drop_modality_inputs is only meaningful for language ranks, "
        f"got role for module {role.module_name!r}."
    )
    if config.layout is MIMOLayout.AUTO:
        layout = classify_layout(config.module_parallelisms, world_size)
    else:
        layout = config.layout
    return layout is MIMOLayout.NON_COLOCATED


def drop_modality_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``batch`` with the raw modality inputs set to None.

    Language-only ranks (non-colocated layouts) receive encoder outputs via
    the MIMO bridge. This covers the nested ``modality_inputs`` key and the
    raw visual batch keys (``imgs`` / ``videos`` / ``image_thw_grids`` /
    ``video_thw_grids``): they are patch-packed - ``imgs`` dim 0 is the total
    patch count across the batch's images, not the sample count - so the
    generic sample-DP slicer must never see them. The batch dict is
    shallow-copied; tensors are shared.
    """
    out = dict(batch)
    out["modality_inputs"] = None
    for key in ("imgs", "videos", "image_thw_grids", "video_thw_grids"):
        out[key] = None
    return out


def prepare_batch_for_module(
    batch: Mapping[str, Any],
    dp_rank: int,
    dp_size: int,
    role: ModuleDataRole,
    config: MIMOParallelismConfig,
    world_size: int,
) -> dict[str, Any]:
    """Prepare a global micro-batch for a rank's module-local DP shard.

    1. Drop modality inputs on language-only ranks (non-colocated layout):
       they consume encoder outputs from the bridge, and the raw modality
       tensors' leading dimension is not the language sample batch.
    2. Contiguously slice the batch for the module-local DP shard (see
       :func:`slice_batch_for_module_dp`).

    Args:
        batch: Global batch dict (same object on every data-loading rank).
        dp_rank: This rank's position in its module-local DP group.
        dp_size: Size of the module-local DP group.
        role: This rank's module role.
        config: Finalized MIMO parallelism configuration.
        world_size: Total number of ranks (used to resolve AUTO layout).

    Returns:
        The module-local batch dict ready for the forward step.
    """
    if role.is_language and should_drop_modality_inputs(role, config, world_size):
        batch = drop_modality_inputs(batch)
    return slice_batch_for_module_dp(batch, dp_rank, dp_size)
