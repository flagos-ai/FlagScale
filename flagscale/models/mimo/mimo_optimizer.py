# Copyright (c) 2025, BAAI. All rights reserved.

"""Per-module DDP and optimizer helpers for colocated MIMO deployment."""

import dataclasses
import inspect
import os
import types
from contextlib import ExitStack

import torch

import megatron.core.parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as DDP, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.training.utils import print_rank_0, unwrap_model

from .parallel_state_ctx import switch_parallel_state


def build_mimo_ddp_config(
    args, model, dp_world_size: int | None = None
) -> DistributedDataParallelConfig:
    """Build a ``DistributedDataParallelConfig`` matching Megatron's default path.

    The implementation is kept in sync with ``megatron.training.training.get_model``
    so that MIMO modules see the same DDP behavior as a non-MIMO model.

    ``dp_world_size`` overrides the default bucket-size heuristic; use it when the
    module's data-parallel size differs from the global default.
    """
    kwargs = {}
    num_parameters = sum(p.nelement() for p in model.parameters())
    for f in dataclasses.fields(DistributedDataParallelConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)

    kwargs["grad_reduce_in_fp32"] = args.accumulate_allreduce_grads_in_fp32
    kwargs["check_for_nan_in_grad"] = args.check_for_nan_in_loss_and_grad
    kwargs["check_for_large_grads"] = args.check_for_large_grads

    if args.ddp_num_buckets is not None:
        assert args.ddp_bucket_size is None, (
            "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
        )
        assert args.ddp_num_buckets > 0, "--ddp-num-buckets must be greater than 0"
        kwargs["bucket_size"] = num_parameters // args.ddp_num_buckets
    else:
        kwargs["bucket_size"] = args.ddp_bucket_size

    kwargs["pad_buckets_for_high_nccl_busbw"] = args.ddp_pad_buckets_for_high_nccl_busbw
    kwargs["reduce_scatter_with_fp32_accumulation"] = args.ddp_reduce_scatter_with_fp32_accumulation
    kwargs["param_name_patterns_for_fp32_local_accumulation"] = tuple(
        args.ddp_param_name_patterns_for_fp32_local_accumulation
    )
    kwargs["average_in_collective"] = args.ddp_average_in_collective
    kwargs["megatron_fsdp_main_params_dtype"] = args.megatron_fsdp_main_params_dtype
    kwargs["megatron_fsdp_main_grads_dtype"] = args.megatron_fsdp_main_grads_dtype
    kwargs["megatron_fsdp_grad_comm_dtype"] = args.megatron_fsdp_grad_comm_dtype

    ddp_config = DistributedDataParallelConfig(**kwargs)

    # Use a sane default bucket size when the user did not provide one.
    if ddp_config.bucket_size is None:
        effective_dp = dp_world_size
        if effective_dp is None:
            effective_dp = mpu.get_data_parallel_world_size(with_context_parallel=True)
        ddp_config.bucket_size = max(40000000, 1000000 * effective_dp)
    # Disable bucketing when gradient overlap is not requested.
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None

    return ddp_config


def wrap_mimo_ddp(mimo_model, args) -> None:
    """Wrap vision and language submodules with their own DDP groups.

    Preconditions:
        - ``mimo_model.vision_pg`` and ``mimo_model.language_pg`` are valid
          process group collections.
        - ``args.use_mimo`` is True (caller responsibility).

    The original ``vision_model`` / ``language_model`` attributes are kept
    unchanged so that the MIMO model's forward and helper methods continue to
    work.  The DDP wrappers are stored as ``vision_ddp`` / ``language_ddp`` and
    are used by the training loop for grad-buffer management and by the
    optimizer builders.
    """
    assert mimo_model.vision_pg is not None, "vision_pg must be set"
    assert mimo_model.language_pg is not None, "language_pg must be set"

    if mimo_model.vision_model is not None:
        with switch_parallel_state(mimo_model.vision_pg):
            vision_dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            vision_ddp_config = build_mimo_ddp_config(
                args, mimo_model.vision_model, dp_world_size=vision_dp_size
            )
            mimo_model.vision_ddp = DDP(
                config=mimo_model.vision_model.config,
                ddp_config=vision_ddp_config,
                module=mimo_model.vision_model,
            )

    with switch_parallel_state(mimo_model.language_pg):
        language_dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        language_ddp_config = build_mimo_ddp_config(
            args, mimo_model.language_model, dp_world_size=language_dp_size
        )
        mimo_model.language_ddp = DDP(
            config=mimo_model.language_model.config,
            ddp_config=language_ddp_config,
            module=mimo_model.language_model,
        )


def get_mimo_ddp_wrappers(model_chunk):
    """Return the per-module DDP wrappers inside a MIMO model chunk."""
    unwrapped = unwrap_model(model_chunk)
    ddps = []
    for attr in ("vision_ddp", "language_ddp"):
        ddp = getattr(unwrapped, attr, None)
        if ddp is not None:
            ddps.append(ddp)
    return ddps


def patch_mimo_model_chunk(model_chunk):
    """Bind DDP-like grad-sync and param-sync methods on the outer Float16Module wrapper.

    MIMO skips the outer DDP wrapper, so the training loop / Megatron
    helpers that call ``model_chunk.finish_grad_sync()`` etc. would otherwise
    fail.  The methods are delegated to the inner vision/language DDP modules.
    """
    ddp_wrappers = get_mimo_ddp_wrappers(model_chunk)
    if ddp_wrappers:
        # Use the language DDP config as the representative config; vision uses
        # the same settings.
        model_chunk.ddp_config = ddp_wrappers[-1].ddp_config
        # Megatron overlap code checks this attribute before registering hooks.
        model_chunk.remove_forward_pre_hook_handles = []

    for method_name in (
        "finish_grad_sync",
        "start_grad_sync",
        "zero_grad_buffer",
        "scale_gradients",
        "enable_forward_pre_hook",
        "disable_forward_pre_hook",
        "start_param_sync",
    ):

        def make_method(name):
            def method(self, *args, **kwargs):
                for ddp in get_mimo_ddp_wrappers(self):
                    fn = getattr(ddp, name, None)
                    if fn is not None:
                        fn(*args, **kwargs)

            return method

        setattr(
            model_chunk,
            method_name,
            types.MethodType(make_method(method_name), model_chunk),
        )

    # no_sync is a context manager on DDP; combine all inner DDP no_sync contexts.
    def _no_sync(self):
        stack = ExitStack()
        for ddp in get_mimo_ddp_wrappers(self):
            if hasattr(ddp, "no_sync"):
                stack.enter_context(ddp.no_sync())
        return stack

    model_chunk.no_sync = types.MethodType(_no_sync, model_chunk)

    # Expose the exit-time training-state cleanup on the outer wrapper so the
    # training loop's duck-typed pre-save cleanup (hasattr checks in
    # ``training.py``) can call it before checkpoint saves.  Bound by plain
    # attribute assignment (not ``types.MethodType``) so ``self`` stays the
    # unwrapped model, which owns ``self.scheduler``.
    unwrapped = unwrap_model(model_chunk)
    for name in ("release_training_state", "drop_completed_macros"):
        if hasattr(unwrapped, name):
            setattr(model_chunk, name, getattr(unwrapped, name))


def setup_mimo_ddp(model, args, wrap_with_ddp: bool = True):
    """Wrap MIMO submodules with per-module DDP and patch the outer wrapper.

    Returns ``(is_mimo, mimo_model)``: whether ``model`` is a colocated MIMO
    model whose DDP setup was performed, and the unwrapped MIMO model
    (``None`` when not MIMO).
    """
    unwrapped_model = unwrap_model(model)
    mimo_model = (
        unwrapped_model[0]
        if isinstance(unwrapped_model, list) and len(unwrapped_model) == 1
        else (unwrapped_model if not isinstance(unwrapped_model, list) else None)
    )
    is_mimo = (
        args.use_mimo
        and wrap_with_ddp
        and mimo_model is not None
        and hasattr(mimo_model, "vision_pg")
    )
    if not is_mimo:
        return False, None

    print_rank_0("Colocated MIMO: wrapping vision/language modules with per-module DDP.")
    wrap_mimo_ddp(mimo_model, args)
    for model_chunk in model:
        patch_mimo_model_chunk(model_chunk)
    return True, mimo_model


def set_mimo_force_all_reduce(model_chunk, value: bool):
    """Propagate ``force_all_reduce`` to inner MIMO DDP wrappers."""
    for ddp in get_mimo_ddp_wrappers(model_chunk):
        ddp.force_all_reduce = value


def _optimizer_state_dict(opt, is_loading: bool = False):
    """Call ``opt.state_dict`` forwarding ``is_loading`` only when supported."""
    sig = inspect.signature(opt.state_dict)
    if "is_loading" in sig.parameters:
        return opt.state_dict(is_loading=is_loading)
    return opt.state_dict()


class ChainedOptimizer:
    """Chain multiple Megatron optimizers so the training loop sees one object.

    If Megatron's training loop needs additional methods, add explicit
    forwarding here.
    """

    def __init__(self, optimizers: list):
        assert len(optimizers) > 0, "ChainedOptimizer requires at least one optimizer"
        self.optimizers = optimizers
        # Expose the same attribute Megatron uses for dist-optimizer chaining.
        self.chained_optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        """Step all wrapped optimizers and aggregate their return values.

        Each Megatron optimizer returns ``(update_successful, grad_norm,
        num_zeros_in_grad)``.  For the chained case we return the logical AND
        of successes, the combined global gradient norm, and the total zero
        count across all optimizers.
        """
        successes = []
        grad_norms = []
        num_zeros = []
        for opt in self.optimizers:
            success, grad_norm, zeros = opt.step()
            successes.append(success)
            grad_norms.append(grad_norm)
            num_zeros.append(zeros)

        update_successful = all(successes)

        # Combine per-optimizer grad norms into a single global norm.
        valid_norms = [gn for gn in grad_norms if gn is not None]
        if valid_norms:
            grad_norm = float(sum(gn * gn for gn in valid_norms) ** 0.5)
        else:
            grad_norm = None

        valid_zeros = [z for z in num_zeros if z is not None]
        num_zeros_in_grad = sum(valid_zeros) if valid_zeros else None

        return update_successful, grad_norm, num_zeros_in_grad

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss using the first optimizer's loss scale."""
        return self.optimizers[0].scale_loss(loss)

    def reload_model_params(self, state_dict=None):
        """Reload main params from model params on all wrapped optimizers."""
        for opt in self.optimizers:
            opt.reload_model_params(state_dict=state_dict)

    @property
    def is_stub_optimizer(self):
        """Return True if all wrapped optimizers are stubs."""
        return all(getattr(opt, "is_stub_optimizer", False) for opt in self.optimizers)

    def state_dict(self, is_loading: bool = False):
        # Stub optimizers (all their params frozen) have no inner optimizer
        # and Megatron's state_dict/load_state_dict carry no stub guard —
        # keep a None placeholder so positions stay aligned with load.
        return [
            None
            if getattr(opt, "is_stub_optimizer", False)
            else _optimizer_state_dict(opt, is_loading=is_loading)
            for opt in self.optimizers
        ]

    def sharded_state_dict(self, state_dict=None, **kwargs):
        """Return sharded state dict for distributed checkpoint formats."""
        return [opt.sharded_state_dict(state_dict, **kwargs) for opt in self.optimizers]

    def load_state_dict(self, state_dicts: list):
        assert len(state_dicts) == len(self.optimizers), (
            f"expected {len(self.optimizers)} optimizer state dicts, got {len(state_dicts)}"
        )
        for opt, sd in zip(self.optimizers, state_dicts):
            if getattr(opt, "is_stub_optimizer", False):
                # Stub: nothing was saved for it (None placeholder).
                continue
            opt.load_state_dict(sd)

    def load_state_dict_from_file(self, checkpoint_name: str):
        """Load each wrapped optimizer state from its own checkpoint file."""
        if len(self.optimizers) == 1:
            self.optimizers[0].load_state_dict_from_file(checkpoint_name)
            return
        for idx, opt in enumerate(self.optimizers):
            opt_filename = self._per_optimizer_filename(checkpoint_name, idx)
            opt.load_state_dict_from_file(opt_filename)

    @staticmethod
    def _per_optimizer_filename(filename: str, index: int) -> str:
        """Return a unique checkpoint filename for the ``index``-th optimizer."""
        base, ext = os.path.splitext(filename)
        return f"{base}_{index}{ext}"

    @staticmethod
    def _unwrap_distributed_optimizers(opt):
        """Return ALL underlying ``DistributedOptimizer``s of ``opt``.

        A MoE module optimizer is itself a chained optimizer (one
        ``DistributedOptimizer`` per parameter partition: dense, experts, ...).
        Each inner optimizer owns disjoint fp32 master-parameter state, so all
        of them must be saved/loaded.
        """
        if hasattr(opt, "chained_optimizers") and opt.chained_optimizers:
            return [
                inner
                for inner in opt.chained_optimizers
                if hasattr(inner, "get_parameter_state_dp_zero")
            ]
        if hasattr(opt, "get_parameter_state_dp_zero"):
            return [opt]
        return []

    def save_parameter_state(self, filename: str):
        """Save each wrapped optimizer's parameter state to a separate file.

        The per-module optimizers live in different data-parallel groups.  The
        state is gathered on gloo/CPU (Megatron's default) before writing on
        each group's DP rank 0, so the save does not allocate multi-GiB GPU
        buffers.

        A module optimizer with a single ``DistributedOptimizer`` keeps the
        historical single-state file format.  A MoE module optimizer (chained
        over dense/expert partitions) writes a list of states — one entry per
        inner ``DistributedOptimizer``, ``None`` where this rank holds nothing
        or the inner optimizer is a stub — mirroring Megatron's own
        ``ChainedOptimizer.save_parameter_state``.
        """
        if len(self.optimizers) == 1:
            self.optimizers[0].save_parameter_state(filename)
            return
        for idx, opt in enumerate(self.optimizers):
            opt_filename = self._per_optimizer_filename(filename, idx)
            inners = self._unwrap_distributed_optimizers(opt)
            if not inners:
                opt.save_parameter_state(opt_filename)
                continue
            if len(inners) == 1:
                inner = inners[0]
                if getattr(inner, "is_stub_optimizer", False):
                    # Stub DistributedOptimizer (all its params frozen): no
                    # parameter state exists and its DP group is uninitialized.
                    continue
                state = inner.get_parameter_state_dp_zero(use_gloo_comm=True)
                if state is not None:
                    torch.save(state, opt_filename)
                continue
            states = []
            save_states = False
            for inner in inners:
                if getattr(inner, "is_stub_optimizer", False):
                    # Stub: no parameter state; keep the placeholder so entry
                    # positions stay aligned with the load side.
                    states.append(None)
                    continue
                state = inner.get_parameter_state_dp_zero(use_gloo_comm=True)
                if inner.data_parallel_group.rank() == 0:
                    states.append(state)
                    save_states = True
                else:
                    assert state is None
                    states.append(None)
            if save_states:
                torch.save(states, opt_filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        """Load each wrapped optimizer's parameter state from its own file."""
        if len(self.optimizers) == 1:
            self.optimizers[0].load_parameter_state(
                filename, update_legacy_format=update_legacy_format
            )
            return
        for idx, opt in enumerate(self.optimizers):
            opt_filename = self._per_optimizer_filename(filename, idx)
            inners = self._unwrap_distributed_optimizers(opt)
            if not inners:
                opt.load_parameter_state(opt_filename, update_legacy_format=update_legacy_format)
                continue
            if len(inners) == 1:
                inner = inners[0]
                if getattr(inner, "is_stub_optimizer", False):
                    # Stub DistributedOptimizer (all its params frozen): nothing
                    # was saved for it and its DP group is uninitialized.
                    continue
                state = None
                if inner.data_parallel_group.rank() == 0:
                    state = torch.load(opt_filename)
                inner.load_parameter_state_from_dp_zero(
                    state, update_legacy_format=update_legacy_format
                )
                continue
            states = None
            for inner_idx, inner in enumerate(inners):
                if getattr(inner, "is_stub_optimizer", False):
                    # Stub: nothing was saved for it (None placeholder).
                    continue
                # Lazy loading: the state file is read only on DP rank 0 (each
                # inner optimizer has its own DP group).
                if inner.data_parallel_group.rank() == 0 and states is None:
                    states = torch.load(opt_filename)
                    assert isinstance(states, list), (
                        "checkpoint uses the legacy single-state format, which cannot contain MoE "
                        "expert state; this checkpoint predates the MoE resume fix and cannot be resumed"
                    )
                state = states[inner_idx] if states else None
                inner.load_parameter_state_from_dp_zero(
                    state, update_legacy_format=update_legacy_format
                )

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def get_loss_scale(self):
        """Return the loss scale of the first optimizer (used by Megatron logging)."""
        return self.optimizers[0].get_loss_scale()

    def get_config(self):
        """Return the optimizer config of the first optimizer."""
        return self.optimizers[0].get_config()


def _pad_param_group_collectives():
    """Pad the world collectives of one missing module-optimizer build.

    ``get_megatron_optimizer`` all-gathers param-group keys over the world
    group in ``_get_param_groups`` (three times per build: dense, MoE and
    engram filters), so every rank must issue the same NUMBER of
    ``get_megatron_optimizer`` calls.  Ranks without a vision module (language
    PP stages beyond the first) pad the missing vision-optimizer call here.
    If Megatron changes the number of world collectives per optimizer build,
    this padding must be updated to match.
    """
    world = torch.distributed.get_world_size()
    for _ in range(3):
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, [])


def build_mimo_optimizer(config, config_overrides, mimo_model, args):
    """Build separate optimizers for vision and language modules."""
    optimizers = []

    vision_ddp = getattr(mimo_model, "vision_ddp", None)
    if vision_ddp is not None:
        assert mimo_model.vision_pg is not None, "vision_pg must be set"
        with switch_parallel_state(mimo_model.vision_pg):
            vision_opt = get_megatron_optimizer(
                config,
                [vision_ddp],
                config_overrides=config_overrides,
                use_gloo_process_groups=args.use_gloo_process_groups,
                dump_param_to_param_group_map=args.dump_param_to_param_group_map,
            )
            optimizers.append(vision_opt)
        # Grad stats (norm / zero count) must not reduce over groups that
        # include ranks without a vision optimizer: the default
        # (intra_dist_opt = vision MP group) spans PP stages at language PP>1
        # and mismatches there.  Use the vision TP group instead — identical
        # membership to the vision MP group at vision PP=1 (so PP=1 numerics
        # are unchanged), and always intra-stage.
        for opt in getattr(vision_opt, "chained_optimizers", [vision_opt]):
            opt.grad_stats_parallel_group = mimo_model.vision_pg.tp
    else:
        _pad_param_group_collectives()

    assert mimo_model.language_pg is not None, "language_pg must be set"
    language_ddp = getattr(mimo_model, "language_ddp", mimo_model.language_model)
    with switch_parallel_state(mimo_model.language_pg):
        language_opt = get_megatron_optimizer(
            config,
            [language_ddp],
            config_overrides=config_overrides,
            use_gloo_process_groups=args.use_gloo_process_groups,
            dump_param_to_param_group_map=args.dump_param_to_param_group_map,
        )
        optimizers.append(language_opt)

    if len(optimizers) == 1:
        return optimizers[0]
    return ChainedOptimizer(optimizers)
