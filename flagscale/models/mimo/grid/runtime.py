# Copyright (c) 2025, BAAI. All rights reserved.

"""Multi-module process group utilities for NON-colocated MIMO heterogeneous parallel training.

Non-colocated MIMO assigns each rank to exactly one module; every module's
``ProcessGroupCollection`` is nullable on ranks that do not participate in it.
All functions accept plain ``module_to_grid_map`` / ``pg_collections`` dicts
(duck-typed) so they work with any infra object, including FlagScale's
non-colocated ``grid.infra``, without importing it.

``HyperCommGrid`` exposes no ``get_pg_size()``; DP sizes are derived from
``grid.shape`` / ``grid.dim_names`` (works on all ranks, including those
outside the grid).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch.distributed as dist

from megatron.core.distributed.finalize_model_grads import (
    finalize_model_grads as _finalize_model_grads,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import MultiModuleProcessGroupCollection

if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.process_groups_config import ProcessGroupCollection


def _get_dp_size_from_grid(grid: HyperCommGrid) -> int:
    """Get the DP dimension size from a grid's shape metadata.

    Uses ``grid.shape`` / ``grid.dim_names`` rather than process groups so it
    works on all ranks, including those outside the grid.
    """
    dp_idx = grid.dim_names.index("dp")
    return grid.shape[dp_idx]


def is_current_rank_in_grid(grid: HyperCommGrid) -> bool:
    """Check if the current rank participates in the given grid."""
    current_rank = dist.get_rank()
    return grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)


def get_active_module_pg(
    pg_collections: dict[str, ProcessGroupCollection | None],
) -> tuple[str, ProcessGroupCollection]:
    """Return the (module_name, pg_collection) for the single active module on this rank.

    Non-colocated MIMO assigns each rank to exactly one module; entries for
    non-participating modules are ``None``.

    Raises:
        AssertionError: If more or fewer than one module is active on this rank.
    """
    active = [(name, pg) for name, pg in pg_collections.items() if pg is not None]
    assert len(active) == 1, (
        f"Non-colocated MIMO requires exactly one active ProcessGroupCollection per rank, "
        f"got {len(active)}. Colocated MIMO is not supported by this code path."
    )
    return active[0]


def build_pg_collection_for_schedule(
    pg_collections: dict[str, ProcessGroupCollection | None],
):
    """Build pg_collection compatible with the schedule.

    Uses ``MultiModuleProcessGroupCollection`` (it allows missing LLM PG on
    encoder-only ranks), built directly from the filtered non-None
    collections.  Configuration errors must propagate: do NOT add a
    try/except fallback that silently downgrades to a plain list of
    collections (the schedule consumes that with different semantics).
    IMPORTANT: uses pg_collections directly; do NOT rebuild PGs.

    Raises:
        ValueError: If no module has a non-None collection on this rank (a
            stub rank - the setup-time ``validate_no_stub_ranks`` guard
            should have rejected this configuration already).
    """
    module_pgs = {k: v for k, v in pg_collections.items() if v is not None}
    if not module_pgs:
        raise ValueError("module_pgs dict cannot be empty")
    language_model_module_name = (
        MIMO_LANGUAGE_MODULE_KEY if MIMO_LANGUAGE_MODULE_KEY in module_pgs else None
    )
    return MultiModuleProcessGroupCollection(
        module_pgs=module_pgs,
        language_model_module_name=language_model_module_name,
    )


@contextmanager
def multimodule_no_sync(*, module_to_grid_tuple: list[tuple]):
    """Context manager to disable gradient sync for all modules during microbatch accumulation.

    Designed for functools.partial() pre-binding of ``module_to_grid_tuple``,
    since the schedule calls no_sync_func() with no arguments.

    Args:
        module_to_grid_tuple: List of (module, grid) tuples (keyword-only,
            bound via partial).

    Raises:
        AttributeError: If a participating module is not DDP-wrapped
            (missing ``no_sync``); failing fast beats silently running
            without gradient-sync control.
    """
    contexts = []
    for module, grid in module_to_grid_tuple:
        if module is not None and is_current_rank_in_grid(grid):
            if not hasattr(module, "no_sync"):
                raise AttributeError(
                    f"module {type(module).__name__} has no 'no_sync' method; "
                    "multimodule_no_sync requires DDP-wrapped modules"
                )
            contexts.append(module.no_sync())

    # Enter all contexts.
    for ctx in contexts:
        ctx.__enter__()

    try:
        yield
    finally:
        # Exit all contexts in reverse order.
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)


def finalize_model_grads_multimodule(
    model,
    num_tokens=None,
    pg_collection=None,
    force_all_reduce=None,
    *,
    module_to_grid_map: dict[str, HyperCommGrid],
    pg_collections: dict[str, ProcessGroupCollection | None],
    module_to_grid_tuple: list[tuple],
):
    """Finalize gradients for each module using module-local pg_collections.

    Signature matches the schedule's call pattern:
        config.finalize_model_grads_func([model], num_tokens, pg_collection, force_all_reduce=flag)

    ``module_to_grid_map`` / ``pg_collections`` / ``module_to_grid_tuple`` are
    pre-bound via partial().  The schedule-provided ``pg_collection`` is
    ignored in favour of per-module PGs; the schedule-provided
    ``force_all_reduce`` flag is forwarded to MCore's standard finalizer for
    each active module.

    When a module's DP differs from the language DP (heterogeneous grids),
    its DDP mean normalizes over a different sample count than the language
    loss; gradients are rescaled after finalization — by the global token
    total on the per-token-loss path, by module_dp / llm_dp on the mean
    path — to stay commensurate.

    Args:
        model: Model list (passed by schedule, ignored - we use module_to_grid_tuple).
        num_tokens: Token count for gradient scaling.
        pg_collection: Schedule-provided PG (ignored - we use per-module PGs).
        force_all_reduce: Schedule-provided flag forwarded to each per-module finalizer.
        module_to_grid_map: Mapping of module names to their HyperCommGrids
            (keyword-only, bound via partial).
        pg_collections: Mapping of module names to ProcessGroupCollection or None
            (keyword-only, bound via partial).
        module_to_grid_tuple: List of (module, grid) tuples (keyword-only, bound via partial).
    """
    llm_grid = module_to_grid_map.get(MIMO_LANGUAGE_MODULE_KEY)
    llm_dp = _get_dp_size_from_grid(llm_grid) if llm_grid is not None else 1

    def _find_module(grid):
        for mn, mg in module_to_grid_map.items():
            if mg is grid:
                return mn, pg_collections.get(mn)
        return None, None

    if num_tokens is not None and llm_grid is not None:
        # calculate_per_token_loss=True path.
        #
        # Only LLM last-PP-stage ranks accumulated non-zero num_tokens;
        # _finalize_model_grads does PP broadcast + DP all-reduce on num_tokens
        # internally, which works for the LLM because each DP rank still holds
        # its own distinct count.  We must NOT broadcast num_tokens globally
        # before calling it — that would overwrite every rank with one DP
        # rank's value, and the subsequent DP all-reduce would sum dp_size
        # identical copies instead of distinct per-rank counts.
        #
        # Encoder ranks have num_tokens=0 (they don't compute loss): pass
        # num_tokens=None for them to skip the broken normalization, then
        # broadcast the correct total from LLM and apply it manually.  With
        # gradient_scaling_factor=1.0 (calculate_per_token_loss=True) DDP does
        # a plain SUM, so after dividing by the global token count the
        # gradient is correct — no DP compensation factor needed.

        # Phase 1: gradient all-reduce for each module.  Only the LLM gets
        # num_tokens so _finalize_model_grads can PP-broadcast + DP-all-reduce
        # the per-rank counts into the correct global total.
        for module, grid in module_to_grid_tuple:
            if module is not None and is_current_rank_in_grid(grid):
                module_name, module_pg = _find_module(grid)
                if module_pg is not None:
                    if module_name == MIMO_LANGUAGE_MODULE_KEY:
                        _finalize_model_grads(
                            [module],
                            num_tokens=num_tokens,
                            pg_collection=module_pg,
                            force_all_reduce=force_all_reduce,
                        )
                    else:
                        _finalize_model_grads(
                            [module],
                            num_tokens=None,
                            pg_collection=module_pg,
                            force_all_reduce=force_all_reduce,
                        )

        # Phase 2: broadcast the correct global total from LLM to encoder
        # ranks (_finalize_model_grads updated num_tokens in-place on LLM
        # ranks: PP broadcast + DP all-reduce → true global total).
        llm_last_rank = llm_grid.rank_offset + llm_grid.size - 1
        dist.broadcast(num_tokens, src=llm_last_rank)

        # Phase 3: scale encoder gradients by 1 / global_total.
        for module, grid in module_to_grid_tuple:
            if module is not None and is_current_rank_in_grid(grid):
                module_name, _ = _find_module(grid)
                if module_name != MIMO_LANGUAGE_MODULE_KEY and num_tokens > 0:
                    module.scale_gradients(1.0 / num_tokens.float().item())
    else:
        # calculate_per_token_loss=False path.
        #
        # Loss was already divided by num_tokens and num_microbatches in the
        # forward pass.  DDP pre-scales gradients by 1/dp_size, producing an
        # effective MEAN over each module's own DP group.  A module whose DP
        # differs from the language DP normalizes over a different sample
        # count; rescale by module_dp / llm_dp (either direction).
        for module, grid in module_to_grid_tuple:
            if module is not None and is_current_rank_in_grid(grid):
                _, module_pg = _find_module(grid)
                if module_pg is not None:
                    _finalize_model_grads(
                        [module],
                        num_tokens=None,
                        pg_collection=module_pg,
                        force_all_reduce=force_all_reduce,
                    )

                    module_dp = _get_dp_size_from_grid(grid)
                    if module_dp != llm_dp:
                        module.scale_gradients(float(module_dp) / float(llm_dp))


def validate_no_stub_ranks(module_to_grid_map: dict[str, HyperCommGrid], world_size: int):
    """Ensure every rank participates in at least one module.

    Stub ranks (ranks not participating in any module) are NOT supported;
    validation runs at setup time to fail fast with a clear error.

    Args:
        module_to_grid_map: Mapping of module names to their HyperCommGrids.
        world_size: Total number of ranks in the world.

    Raises:
        ValueError: If any rank doesn't participate in a module.
    """
    participating_ranks = set()
    for module_name, grid in module_to_grid_map.items():
        for rank in range(grid.rank_offset, grid.rank_offset + grid.size):
            participating_ranks.add(rank)

    all_ranks = set(range(world_size))
    stub_ranks = all_ranks - participating_ranks

    if stub_ranks:
        raise ValueError(
            f"Ranks {sorted(stub_ranks)} do not participate in any module. "
            f"Stub ranks are not supported. Adjust parallelism config to use all {world_size} GPUs, "
            f"or reduce world_size to {len(participating_ranks)}."
        )
