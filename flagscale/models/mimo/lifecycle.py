# Copyright (c) 2026, BAAI. All rights reserved.

"""Unified MIMO lifecycle facade.

Single boundary between the training code (training loop, training entry,
argument parsing) and the two MIMO layouts: every layout-aware decision
dispatches here on ``--mimo-layout`` — per-module DDP setup, optimizer
construction, post-load param-group sync, gradient-sync hooks, forward-backward
function, loss-reduction context, sample accounting, state destroy / release /
drop, force-all-reduce, parse/runtime contracts, batch preparation and the
dataloader shard policy.  Callers import only ``flagscale.models.mimo`` and
never touch ``mimo.colocated`` / ``mimo.grid`` internals; the grid imports
stay deferred so non-grid runs never pay for the MCore MIMO stack.
"""

import functools

import torch.distributed

from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.utils import get_model_config, unwrap_model

from .colocated.optimizer import (
    build_mimo_optimizer as build_colocated_optimizer,
    setup_mimo_ddp as setup_colocated_mimo_ddp,
)
from .colocated.utils import (
    drop_mimo_completed_macros as drop_colocated_completed_macros,
    release_mimo_training_state as release_colocated_training_state,
)
from .ddp_utils import get_mimo_ddp_wrappers
from .grid.training import (
    apply_grid_parse_time_contract,
    build_grid_optimizer,
    configure_grid_model_config_hooks,
    destroy_grid_training_states,
    get_logical_iteration_samples as get_grid_logical_iteration_samples,
    grid_training_state_from_model_chunk,
    prepare_grid_batch,
    reconfigure_grid_num_microbatches_calculator,
    setup_grid_mimo_ddp,
    sync_grid_optimizer_param_group_lr,
    validate_grid_runtime_contract,
)


def _use_mimo(args) -> bool:
    return bool(getattr(args, "use_mimo", False))


def _is_grid(args) -> bool:
    return _use_mimo(args) and getattr(args, "mimo_layout", "colocated") == "grid"


# ---------------------------------------------------------------------------
# Contracts and runtime setup.
# ---------------------------------------------------------------------------


def apply_parse_time_contract(args) -> None:
    """Grid parse-time contract (no-op for the other layouts).

    Must run before Megatron's argument validation, which silently rewrites
    conflicting values; see ``grid.training.apply_grid_parse_time_contract``.
    """
    if not _is_grid(args):
        return
    apply_grid_parse_time_contract(args)


def setup_mimo_runtime(args) -> None:
    """Runtime layout setup at the training entry (no-op for colocated).

    Grid: run the layout runtime contract, reject flags that would silently
    conflict with the grid path's own layout derivation, pin the global
    parallel state to TP=1/PP=1/DP=1 (module layouts come exclusively from
    ``--mimo-module-specs``), preserve the user's SP intent in
    ``args.mimo_sequence_parallel`` and rebase the num-microbatches calculator
    to the forced DP=1.
    """
    if not _use_mimo(args):
        return
    layout = getattr(args, "mimo_layout", "colocated")
    if layout == "colocated":
        return
    if layout != "grid":
        raise ValueError(f"Unsupported --mimo-layout {layout!r}: expected 'colocated' or 'grid'")
    validate_grid_runtime_contract(args)
    if args.mimo_module_specs is None:
        raise ValueError(
            "--mimo-layout=grid requires --mimo-module-specs, e.g. "
            "'images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2'."
        )
    if args.ckpt_format != "torch_dist":
        raise ValueError(
            "--mimo-layout=grid requires --ckpt-format=torch_dist (the "
            "grid path saves via MCore sharded state dicts)."
        )
    if args.rampup_batch_size is not None:
        raise ValueError("--mimo-layout=grid does not support rampup_batch_size (fail-fast).")
    # Pipeline-allocation overrides would silently conflict with the
    # grid path's own PP layout / layer split (computed from
    # --mimo-module-specs and --num-layers).
    if args.pipeline_model_parallel_layout is not None:
        raise ValueError(
            "--mimo-layout=grid does not support --pipeline-model-parallel-layout: "
            "the grid path allocates the language pipeline stages itself "
            "(even or uneven first/last split from --num-layers and the "
            "language PP in --mimo-module-specs) (fail-fast)."
        )
    if (
        getattr(args, "decoder_first_pipeline_num_layers", None) is not None
        or getattr(args, "decoder_last_pipeline_num_layers", None) is not None
    ):
        raise ValueError(
            "--mimo-layout=grid does not support "
            "--decoder-first-pipeline-num-layers / "
            "--decoder-last-pipeline-num-layers: the grid path computes "
            "the first/last stage layer counts itself (base+remainder / "
            "base when the layer count is not divisible by the language "
            "PP) (fail-fast)."
        )
    if getattr(args, "account_for_embedding_in_pipeline_split", False) or getattr(
        args, "account_for_loss_in_pipeline_split", False
    ):
        raise ValueError(
            "--mimo-layout=grid does not support "
            "--account-for-embedding-in-pipeline-split / "
            "--account-for-loss-in-pipeline-split: MCore's uneven "
            "pipeline allocation (used when the layer count is not "
            "divisible by the language PP) is incompatible with "
            "standalone embedding/loss stages (fail-fast)."
        )
    # Backstop pinning of the forced global state (the parse-time
    # contract already rejected non-default legacy parallel sizes).
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.data_parallel_size = 1
    args.context_parallel_size = 1
    args.expert_model_parallel_size = 1
    # SP cannot survive the forced global TP=1 (ModelParallelConfig
    # rejects SP without TP); the user's intent is preserved in
    # args.mimo_sequence_parallel and resolved per module at model
    # build time.  Fall back to the raw value if the parse-time
    # contract did not run.
    if not hasattr(args, "mimo_sequence_parallel"):
        args.mimo_sequence_parallel = args.sequence_parallel
    args.sequence_parallel = False
    reconfigure_grid_num_microbatches_calculator(args)
    from megatron.training.utils import print_rank_0

    print_rank_0(
        "> non-colocated grid MIMO: global parallel state forced to "
        "TP=1/PP=1/DP=1; module layouts from --mimo-module-specs"
    )


# ---------------------------------------------------------------------------
# Model / training lifecycle.
# ---------------------------------------------------------------------------


def setup_mimo_ddp(model, args, wrap_with_ddp: bool = True):
    """Layout-aware per-module DDP setup (no outer DDP).

    Returns ``(is_mimo, layout_state)``: whether ``model`` is a MIMO model of
    the selected layout whose DDP setup was performed, and the layout's
    training state (colocated: the model wrapper; grid: ``GridTrainingState``;
    ``None`` when not MIMO).
    """
    if not (wrap_with_ddp and _use_mimo(args)):
        return False, None
    layout = getattr(args, "mimo_layout", "colocated")
    if layout == "grid":
        return setup_grid_mimo_ddp(model, args, wrap_with_ddp)
    if layout == "colocated":
        return setup_colocated_mimo_ddp(model, args, wrap_with_ddp)
    raise ValueError(f"Unsupported --mimo-layout {layout!r}: expected 'colocated' or 'grid'")


def build_mimo_optimizer(config, config_overrides, model, args):
    """Build the layout-specific optimizer for a MIMO model.

    Colocated: one Megatron optimizer per module chained into
    ``ChainedOptimizer``; grid: MCore ``MimoOptimizer`` (module-namespaced
    shard keys) — see ``grid.training.build_grid_optimizer``.
    """
    unwrapped_model = unwrap_model(model)
    mimo_model = unwrapped_model[0] if isinstance(unwrapped_model, list) else unwrapped_model
    if _is_grid(args):
        return build_grid_optimizer(mimo_model, config)
    return build_colocated_optimizer(config, config_overrides, mimo_model, args)


def sync_optimizer_param_group_lr(optimizer, args) -> bool:
    """Re-sync param-group LRs after an optimizer checkpoint load (grid only).

    Grid-mode resume with ``--override-opt-param-scheduler``: see
    ``grid.training.sync_grid_optimizer_param_group_lr``.  No-op (returns
    False) for the other layouts.
    """
    if not _is_grid(args):
        return False
    return sync_grid_optimizer_param_group_lr(optimizer, args)


def configure_model_config_hooks(model, args) -> None:
    """Bind gradient synchronization hooks for the selected MIMO layout."""
    model_chunk = model[0]
    if _is_grid(args):
        grid_state = grid_training_state_from_model_chunk(model_chunk)
        assert grid_state is not None, "grid mode requires mimo_grid_state on the model"
        configure_grid_model_config_hooks(grid_state, model_chunk)
        return
    if not _use_mimo(args):
        return
    if not hasattr(model_chunk, "no_sync"):
        return
    config = get_model_config(model_chunk)

    # The standard training loop only installs these hooks for an outer DDP.
    # Colocated MIMO owns per-module DDP wrappers behind a Float16Module.
    if args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "colocated MIMO requires an unset no_sync_func when overlap_grad_reduce is enabled"
        )
        config.no_sync_func = model_chunk.no_sync
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = model_chunk.start_param_sync


def get_mimo_forward_backward_func(model, args):
    """Layout-specific forward-backward callable, or ``None`` for the standard schedule.

    Grid always drives the multi-module pipeline schedule (even when every
    module has PP == 1): it owns the cross-module activation/gradient
    transport via the ``MultiModulePipelineCommunicator`` and the
    dict-of-modules forward/backward contract.
    """
    if not _is_grid(args):
        return None
    grid_state = grid_training_state_from_model_chunk(model[0])
    assert grid_state is not None, "grid mode requires mimo_grid_state on the model"
    return functools.partial(
        forward_backward_pipelining_without_interleaving,
        p2p_communicator=grid_state.multimodule_communicator,
        pg_collection=grid_state.multimodule_pg_collection,
    )


def get_mimo_loss_reduction_context(model, args) -> tuple:
    """Loss-reduction override ``(is_last_stage, loss_dp_group)``.

    ``(None, None)`` means the standard contract: global-parallel last stage +
    global DP-CP reduction.  Grid returns the language module's last PP stage
    (the only ranks that produced losses) and its DP-CP group.
    """
    if not _is_grid(args):
        return None, None
    grid_state = grid_training_state_from_model_chunk(model[0])
    assert grid_state is not None, "grid mode requires mimo_grid_state on the model"
    return grid_state.is_language_last_stage, grid_state.local_pg_collection.dp_cp


def get_logical_iteration_samples(args, num_microbatches: int):
    """Sample count of one iteration, or ``None`` for standard sample accounting.

    Grid's data_parallel_size is forced to 1 (module-local DP slicing happens
    in the forward step), so one iteration consumes
    ``micro_batch_size * num_microbatches`` samples.
    """
    if not _is_grid(args):
        return None
    return get_grid_logical_iteration_samples(args, num_microbatches)


def destroy_mimo_training_states() -> None:
    """Release grid training state if the grid runtime was loaded."""
    destroy_grid_training_states()


def release_mimo_training_state(model) -> None:
    """Release scheduler-held training state ahead of an exit checkpoint save.

    Colocated scheduler state only (grid-owned state is released by
    :func:`destroy_mimo_training_states` at teardown); duck-typed, so non-MIMO
    and grid chunks are untouched.
    """
    release_colocated_training_state(model)


def drop_mimo_completed_macros(model) -> None:
    """Drop exhausted, gradient-free colocated macro batches before a periodic save.

    Safe while training continues; no-op for non-colocated chunks.
    """
    drop_colocated_completed_macros(model)


def set_mimo_force_all_reduce(model_chunk, value: bool) -> None:
    """Propagate ``force_all_reduce`` to the per-module DDP wrappers (both layouts)."""
    for ddp in get_mimo_ddp_wrappers(model_chunk):
        ddp.force_all_reduce = value


# ---------------------------------------------------------------------------
# Data path.
# ---------------------------------------------------------------------------


def prepare_mimo_batch(model, batch: dict):
    """Prepare the global micro-batch for the local module's forward.

    Grid: applies the module-local DP slice and assembles the exact forward
    kwargs of the model's registered batch preparer.  Other layouts: the batch
    is returned unchanged.  Returns ``(data_batch, grid_state)`` where
    ``grid_state`` is ``None`` outside grid mode.
    """
    grid_state = getattr(model, "mimo_grid_state", None)
    if grid_state is None:
        return batch, None
    return prepare_grid_batch(batch, grid_state), grid_state


def get_dataloader_shard_policy(args):
    """Sampler shard ``(rank, world_size, group)`` override, or ``None``.

    Grid: every data-loading rank samples the same global micro-batch
    (module-local DP slicing happens in the forward step), so the sampler must
    not shard (rank 0 over the world group).  Colocated/baseline: standard
    DP-sharded sampler.
    """
    if not _is_grid(args):
        return None
    return 0, 1, torch.distributed.group.WORLD
