# Copyright (c) 2025, BAAI. All rights reserved.

"""Per-module DDP and optimizer helpers for colocated MIMO deployment."""

import inspect
import logging
import os
from contextlib import contextmanager

import torch

import megatron.core.parallel_state as mpu
from megatron.core.dist_checkpointing.utils import (
    add_prefix_for_sharding,
    replace_prefix_for_sharding,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.clip_grads import (
    clip_grad_by_total_norm_fp32,
    get_grad_norm_fp32,
)
from megatron.core.utils import log_single_rank, unwrap_model
from megatron.plugin.platform import get_platform

from ..ddp_utils import build_mimo_ddp_config, get_mimo_ddp_wrappers, patch_mimo_model_chunk
from .parallel_state_ctx import switch_parallel_state

logger = logging.getLogger(__name__)


def _optimizer_leaves(opt):
    """Flatten ``opt`` into its leaf optimizers, recursing through nested
    Megatron ``ChainedOptimizer`` wrappers."""
    nested = getattr(opt, "chained_optimizers", None)
    if not nested:
        return [opt]
    return [leaf for inner in nested for leaf in _optimizer_leaves(inner)]


@contextmanager
def _module_grad_stats_scope(opt):
    """Run one module optimizer's own collectives on its overflow-safe group.

    The joint step keeps ``grad_stats_parallel_group`` on WORLD for the single
    combined norm; the per-optimizer overflow and zero-count reduces must stay
    within the ranks that host this optimizer (``_mimo_local_stats_group``),
    because ranks without the module issue no per-module collectives.
    """
    group = getattr(opt, "_mimo_local_stats_group", None)
    if group is None:
        yield
        return
    leaves = _optimizer_leaves(opt)
    saved = [leaf.grad_stats_parallel_group for leaf in leaves]
    for leaf in leaves:
        leaf.grad_stats_parallel_group = group
    try:
        yield
    finally:
        for leaf, previous in zip(leaves, saved):
            leaf.grad_stats_parallel_group = previous


def wrap_mimo_ddp(mimo_model, args) -> None:
    """Wrap vision and language submodules with their own DDP groups.

    Caller must ensure ``args.use_mimo`` is True; the original ``vision_model``
    / ``language_model`` attributes stay untouched.
    """
    assert mimo_model.vision_pg is not None, "vision_pg must be set"
    assert mimo_model.language_pg is not None, "language_pg must be set"

    module_to_ddp = {}
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
            module_to_ddp["vision"] = mimo_model.vision_ddp

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
        module_to_ddp["language"] = mimo_model.language_ddp
    object.__setattr__(mimo_model, "module_to_ddp", module_to_ddp)


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

    # Keep package imports independent of the full megatron.training stack.
    from megatron.training.utils import print_rank_0

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
    """Chain multiple Megatron optimizers so the training loop sees one object."""

    def __init__(self, optimizers: list):
        assert len(optimizers) > 0, "ChainedOptimizer requires at least one optimizer"
        self.optimizers = optimizers
        # Expose the same attribute Megatron uses for dist-optimizer chaining.
        self.chained_optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        """Step every module with one global gradient norm and clip coefficient."""
        leaves = [leaf for opt in self.optimizers for leaf in _optimizer_leaves(opt)]
        found_inf = False
        for opt in self.optimizers:
            with _module_grad_stats_scope(opt):
                found_inf |= opt.prepare_grads()

        if any(getattr(leaf, "grad_scaler", None) is not None for leaf in leaves):
            found_inf_tensor = torch.tensor(
                [found_inf], dtype=torch.float32, device=get_platform().device_name()
            )
            torch.distributed.all_reduce(
                found_inf_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=torch.distributed.group.WORLD,
            )
            found_inf = found_inf_tensor.item() > 0
        if found_inf:
            return False, None, None

        active_leaves = [
            leaf
            for leaf in leaves
            if not getattr(leaf, "is_stub_optimizer", False) and leaf.get_parameters()
        ]
        grads_for_norm = [
            grad for leaf in active_leaves for grad in leaf.get_main_grads_for_grad_norm()
        ]
        if grads_for_norm:
            grad_norm = get_grad_norm_fp32(
                grads_for_norm, grad_stats_parallel_group=torch.distributed.group.WORLD
            )
        else:
            norm = torch.zeros(1, dtype=torch.float32, device=get_platform().device_name())
            torch.distributed.all_reduce(norm, group=torch.distributed.group.WORLD)
            grad_norm = 0.0

        should_skip = False
        for leaf in active_leaves:
            params = leaf.get_parameters()
            if leaf.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    params,
                    max_norm=leaf.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=(
                        leaf.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                        or (
                            leaf.config.use_precision_aware_optimizer
                            and getattr(params[0], "__fsdp_param__", False)
                        )
                    ),
                )
            if grad_norm > leaf.config.grad_norm_skip_threshold:
                log_single_rank(
                    logger,
                    logging.INFO,
                    "skipping update because grad norm is too large %s",
                    grad_norm,
                )
                should_skip = True

        num_zeros_in_grad = None
        if self.get_config().log_num_zeros_in_grad:
            num_zeros_in_grad = 0
            for opt in self.optimizers:
                with _module_grad_stats_scope(opt):
                    for leaf in _optimizer_leaves(opt):
                        if not getattr(leaf, "is_stub_optimizer", False):
                            num_zeros_in_grad += leaf.count_zeros()

        if should_skip:
            return False, grad_norm, num_zeros_in_grad

        update_successful = True
        for opt in self.optimizers:
            update_successful &= opt.step_with_ready_grads()

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
        if len(self.optimizers) == 1:
            return _optimizer_state_dict(self.optimizers[0], is_loading=is_loading)
        # Stub optimizers have no inner optimizer and Megatron carries no stub
        # guard — keep the None placeholder so positions align with load.
        return [
            None
            if getattr(opt, "is_stub_optimizer", False)
            else _optimizer_state_dict(opt, is_loading=is_loading)
            for opt in self.optimizers
        ]

    def sharded_state_dict(self, state_dict=None, **kwargs):
        """Return optimizer state with collision-free module prefixes."""
        if len(self.optimizers) == 1:
            return self.optimizers[0].sharded_state_dict(state_dict, **kwargs)
        sharded_state_dicts = [
            None
            if getattr(opt, "is_stub_optimizer", False)
            else opt.sharded_state_dict(state_dict, **kwargs)
            for opt in self.optimizers
        ]
        for idx, opt_sd in enumerate(sharded_state_dicts):
            if opt_sd is not None:
                add_prefix_for_sharding(opt_sd, f"chained_{idx}.")
        return sharded_state_dicts

    def load_state_dict(self, state_dicts):
        if len(self.optimizers) == 1:
            state = (
                state_dicts[0]
                if isinstance(state_dicts, list) and len(state_dicts) == 1
                else state_dicts
            )
            self.optimizers[0].load_state_dict(state)
            return
        assert len(state_dicts) == len(self.optimizers), (
            f"expected {len(self.optimizers)} optimizer state dicts, got {len(state_dicts)}"
        )
        for idx, (opt, sd) in enumerate(zip(self.optimizers, state_dicts)):
            if getattr(opt, "is_stub_optimizer", False):
                # Stub: nothing was saved for it (None placeholder).
                continue
            if sd is not None:
                replace_prefix_for_sharding(sd, f"chained_{idx}.", "")
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

        A MoE module optimizer is itself chained (one ``DistributedOptimizer``
        per parameter partition); each owns disjoint fp32 master-parameter state.
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

        State is gathered on gloo/CPU (Megatron default) before writing on each
        group's DP rank 0, so no multi-GiB GPU buffers are allocated.  A single
        ``DistributedOptimizer`` keeps the historical single-state file format; a
        MoE chained optimizer writes one list entry per inner optimizer (``None``
        where this rank holds nothing or it is a stub), mirroring Megatron's
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
                    # Stub: keep the None placeholder (positions align with load).
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
                    # Stub: nothing was saved and its DP group is uninitialized.
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
        """Return the first leaf optimizer's configuration."""
        return _optimizer_leaves(self.optimizers[0])[0].config


def _pad_param_group_collectives():
    """Pad the world collectives of one missing module-optimizer build.

    ``get_megatron_optimizer`` all-gathers param-group keys over the world group
    three times per build, so every rank must issue the same NUMBER of calls;
    ranks without a vision module pad the missing vision-optimizer call here.
    If Megatron changes the collectives per build, update this padding to match.
    """
    world = torch.distributed.get_world_size()
    for _ in range(3):
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, [])


def build_mimo_optimizer(config, config_overrides, mimo_model, args):
    """Build module optimizers with WORLD-wide joint gradient statistics."""
    assert args.use_distributed_optimizer, (
        "colocated MIMO joint gradient clipping requires the distributed optimizer"
    )
    assert not args.dump_param_to_param_group_map, (
        "colocated MIMO does not support dumping parameter-group maps"
    )
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
        vision_opt._mimo_local_stats_group = mimo_model.vision_pg.tp_dp_cp
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
    language_opt._mimo_local_stats_group = mimo_model.language_pg.tp_dp_cp

    world_group = torch.distributed.group.WORLD
    for opt in optimizers:
        for leaf in _optimizer_leaves(opt):
            leaf.grad_stats_parallel_group = world_group

    return ChainedOptimizer(optimizers)
