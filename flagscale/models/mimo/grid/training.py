# Copyright (c) 2026, BAAI. All rights reserved.

"""Non-colocated grid training lifecycle helpers (FlagScale-native).

Wires grid-mode MIMO into the FlagScale training lifecycle, mirroring the
Megatron-Bridge ``megatron_mimo`` setup against the FlagScale /
Megatron-LM-FL v0.18.2 APIs:

- :func:`setup_grid_mimo_ddp` — per-module DDP wrapping (no outer DDP) plus
  delegation of the DDP-ish methods on the outer wrapper.
- :func:`build_grid_multimodule_communicator` — the
  ``MultiModulePipelineCommunicator`` described by the registered model
  contract (see :mod:`.contracts`).
- :func:`configure_grid_model_config_hooks` — ``no_sync_func`` /
  ``finalize_model_grads_func`` bound to the per-module gradient helpers.
- :func:`prepare_grid_batch` — module-role batch preparation, delegated to
  the batch preparer registered by the model's grid provider.
- :func:`apply_grid_parse_time_contract` / :func:`validate_grid_runtime_contract`
  — layout-generic fail-fast contracts on the training args.
- :class:`GridTrainingState` — grid-mode training state, attached to the
  model chunk as ``mimo_grid_state``.

Invariants: process groups are created once by ``build_mimo_infra`` (never
here); only modules whose ``ProcessGroupCollection`` is non-None on this
rank get a DDP wrapper; gradient normalization uses per-module PGs
(language is authoritative for token counts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import ShardedBase
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.models.mimo import get_mimo_optimizer
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import MimoOptimizer
from megatron.core.num_microbatches_calculator import reconfigure_num_microbatches_calculator
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.utils import get_model_config, unwrap_model

from ..ddp_utils import build_mimo_ddp_config, patch_mimo_model_chunk
from .contracts import get_grid_communicator_contract
from .runtime import (
    build_pg_collection_for_schedule,
    finalize_model_grads_multimodule,
    get_active_module_pg,
    multimodule_no_sync,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from megatron.core.process_groups_config import (
        MultiModuleProcessGroupCollection,
        ProcessGroupCollection,
    )

    from .infra import MIMOInfra

logger = logging.getLogger(__name__)
_GRID_TRAINING_STATES: list[GridTrainingState] = []


#: Model-specific grid batch preparers, keyed like the communicator contracts
#: in :mod:`.contracts`.  The forward kwargs of a grid model are a property of
#: the model, so its provider module registers the preparer at import time;
#: adding a model to the grid path touches only its own providers module.
_GRID_BATCH_PREPARERS: dict[str, Callable[[dict, GridTrainingState], dict]] = {}


def register_grid_batch_preparer(key: str, preparer) -> None:
    """Register ``preparer`` as the grid batch preparer for ``key``.

    ``preparer(batch, grid_state)`` must return the forward kwargs for the
    local module role.  Raises ``ValueError`` on duplicate registration
    (almost always a copy-paste error).
    """
    if not callable(preparer):
        raise TypeError("grid batch preparer must be callable")
    if key in _GRID_BATCH_PREPARERS:
        raise ValueError(
            f"grid batch preparer '{key}' is already registered; refusing to overwrite"
        )
    _GRID_BATCH_PREPARERS[key] = preparer


def get_grid_batch_preparer(key: str | None = None):
    """Look up a registered batch preparer (single-registration fast path)."""
    if key is not None:
        try:
            return _GRID_BATCH_PREPARERS[key]
        except KeyError:
            registered = sorted(_GRID_BATCH_PREPARERS) or ["<none>"]
            raise KeyError(
                f"no grid batch preparer registered under '{key}' (registered: "
                f"{registered}); import the model's providers module to register it"
            ) from None
    if not _GRID_BATCH_PREPARERS:
        raise RuntimeError(
            "no grid batch preparer registered; import the model's providers "
            "module so it registers its batch preparer before preparing batches"
        )
    if len(_GRID_BATCH_PREPARERS) > 1:
        raise RuntimeError(
            f"multiple grid batch preparers registered ({sorted(_GRID_BATCH_PREPARERS)}); "
            "pass an explicit key"
        )
    return next(iter(_GRID_BATCH_PREPARERS.values()))


def prepare_grid_batch(batch: dict[str, Any], grid_state: GridTrainingState) -> dict[str, Any]:
    """Prepare the global micro-batch for this rank's grid module role.

    Every data-loading rank samples the *same* global micro-batch
    (``args.data_parallel_size == 1``, broadcast over the world TP group).
    Delegates to the batch preparer registered by the model's grid provider
    (see :func:`register_grid_batch_preparer`), which applies the module-local
    DP slice and assembles the exact kwargs the model forward accepts.
    """
    return get_grid_batch_preparer()(batch, grid_state)


def reconfigure_grid_num_microbatches_calculator(args) -> None:
    """Reconfigure the global num-microbatches calculator to grid-mode DP=1.

    Parse time initializes the calculator with the YAML's global data
    parallel size, but grid mode forces the global parallel state to DP=1
    (module-local DP slicing happens in the forward step), so the calculator
    must be rebased to DP=1 before the grid batch contract
    (``validate_grid_batch_divisibility``) and the schedule consume it.  Uses
    the public API with the same parameters as the parse-time init (only the
    DP differs); gbs/mbs and any step schedule are preserved.  The colocated
    path never calls this helper.
    """
    reconfigure_num_microbatches_calculator(
        rank=args.rank,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=args.data_parallel_size,
        decrease_batch_size_if_needed=getattr(args, "decrease_batch_size_if_needed", False),
        step_batch_size_schedule=getattr(args, "step_batch_size_schedule", None),
        seq_length=getattr(args, "seq_length", None),
    )


def get_logical_iteration_samples(args, num_microbatches: int) -> int | None:
    """Return grid's distinct sample count, or ``None`` for standard layouts."""
    if getattr(args, "use_mimo", False) and getattr(args, "mimo_layout", "colocated") == "grid":
        return args.micro_batch_size * num_microbatches
    return None


def _legacy_parallel_arg(args, name: str):
    """Read a legacy parallel arg from the top level or a nested namespace.

    The FlagScale runner flattens parallel sizes to top-level CLI flags; the
    megatron-native ``--yaml-cfg`` format nests them under
    ``args.model_parallel``.  Probe both shapes (top-level first) - correct
    for either input shape.
    """
    value = getattr(args, name, None)
    if value is None:
        model_parallel = getattr(args, "model_parallel", None)
        if model_parallel is not None:
            value = getattr(model_parallel, name, None)
    return value


def apply_grid_parse_time_contract(args) -> None:
    """Parse-time contract for the non-colocated grid layout (fail-fast).

    Must run before Megatron's argument validation, which silently rewrites
    conflicting values (``validate_yaml`` clamps PP via ``min``; both
    validators force ``sequence_parallel=False`` when TP == 1), so conflicts
    are reported against the user's actual input.  Covers both parse entries:
    ``FSTrainArguments.pre_validate_args`` calls this before the validation
    fork, whether validation goes through ``validate_args`` (runner/CLI
    flattened) or ``validate_yaml`` (``--yaml-cfg``).  Grid flags are read as
    top-level attributes everywhere downstream; ``_legacy_parallel_arg`` also
    probes the nested ``model_parallel`` namespace, and the training entry
    point pins the legacy parallel sizes to 1 as a backstop.

    In grid mode the module layouts come exclusively from
    ``--mimo-module-specs`` and the global parallel state runs
    TP=1/PP=1/DP=1; a non-default legacy parallel size only contradicts the
    specs.  ``data_parallel_size`` is exempt (validation derives it from the
    world size).  The user's sequence-parallel intent is preserved in
    ``args.mimo_sequence_parallel``: forced global TP=1 makes Megatron drop
    ``sequence_parallel``, while the grid path resolves SP per module at
    model build time.
    """
    offenders = {}
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    ):
        value = _legacy_parallel_arg(args, name)
        if value is not None and value != 1:
            offenders[name] = value
    for name in ("virtual_pipeline_model_parallel_size",):
        value = _legacy_parallel_arg(args, name)
        if value is not None:
            offenders[name] = value
    # ``dualpipev_pipeline_model_parallel_size`` is not an argparse attribute:
    # it is derived from the ``--use-dualpipev`` switch in
    # ``post_validate_args``, which runs after this parse-time contract, so
    # probe the switch itself.  Only truthy values are offenders: the
    # default (False / unset) must not trip the contract.
    use_dualpipev = getattr(args, "use_dualpipev", False)
    if not use_dualpipev:
        use_dualpipev = _legacy_parallel_arg(args, "use_dualpipev")
    if use_dualpipev:
        offenders["use_dualpipev"] = use_dualpipev
    if offenders:
        raise ValueError(
            "--mimo-layout=grid expresses module layouts exclusively via "
            "--mimo-module-specs; keep legacy parallel sizes and the "
            "dualpipev switch at their defaults "
            f"(1 / None / off). Offending values: {offenders}."
        )

    args.mimo_sequence_parallel = bool(_legacy_parallel_arg(args, "sequence_parallel") or False)


def validate_grid_runtime_contract(args) -> None:
    """Reject runtime features not supported by the non-colocated grid layout."""
    if getattr(args, "cuda_graph_impl", "none") == "full_iteration":
        raise ValueError(
            "--mimo-layout=grid does not support full-iteration CUDA graphs: "
            "the schedule uses dynamic P2P shapes, collectives, and host metadata."
        )
    if getattr(args, "eval_iters", 0) > 0:
        if args.eval_micro_batch_size != args.micro_batch_size:
            raise ValueError(
                "--mimo-layout=grid requires eval_micro_batch_size to equal "
                "micro_batch_size because validation uses the training batch contract."
            )
        if args.eval_global_batch_size % args.eval_micro_batch_size != 0:
            raise ValueError(
                "--mimo-layout=grid requires eval_global_batch_size to be divisible "
                "by eval_micro_batch_size."
            )
    for name in ("context_parallel_size", "expert_model_parallel_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"--mimo-layout=grid requires {name}=1 in this stage.")
    if getattr(args, "num_experts", None):
        raise ValueError("--mimo-layout=grid does not support MoE/num_experts in this stage.")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        raise ValueError("--mimo-layout=grid does not support virtual pipeline parallelism.")


@dataclass
class GridTrainingState:
    """Grid-mode training state attached to the model chunk (``mimo_grid_state``).

    Attributes:
        infra: the built ``MIMOInfra`` (grids + nullable PG collections).
        module_to_grid_tuple: ``(ddp_module, grid)`` pairs for the modules
            this rank participates in (gradient sync / zero-buffer helpers).
        multimodule_pg_collection / multimodule_communicator: schedule PG
            collection and P2P communicator.
        active_module_name / local_pg_collection: the single module this rank
            participates in and its ``ProcessGroupCollection``.
        world_size: distributed world size.
    """

    infra: MIMOInfra
    parallelism_config: object | None = None
    module_to_grid_tuple: list[tuple] = field(default_factory=list)
    multimodule_pg_collection: MultiModuleProcessGroupCollection | None = None
    multimodule_communicator: MultiModulePipelineCommunicator | None = None
    active_module_name: str | None = None
    local_pg_collection: ProcessGroupCollection | None = None
    world_size: int = 0
    _destroyed: bool = False

    def destroy(self) -> None:
        """Release grid-owned process groups once."""
        if self._destroyed:
            return
        self.infra.destroy()
        self.multimodule_communicator = None
        self.multimodule_pg_collection = None
        self._destroyed = True

    @property
    def is_language_rank(self) -> bool:
        return self.active_module_name == MIMO_LANGUAGE_MODULE_KEY

    @property
    def is_language_last_stage(self) -> bool:
        """True on ranks that produce the loss (language module's last PP stage)."""
        if not self.is_language_rank:
            return False
        pp_group = self.local_pg_collection.pp
        return dist.get_group_rank(pp_group, dist.get_rank()) == pp_group.size() - 1


def destroy_grid_training_states() -> None:
    """Destroy all registered grid states; safe to call during restart."""
    for state in list(_GRID_TRAINING_STATES):
        state.destroy()
    _GRID_TRAINING_STATES.clear()
    BridgeCommunicator.destroy_broadcast_pgs()


def _grid_module_from_model(mimo_model, module_name: str):
    """Return the raw (unwrapped) submodule for ``module_name``."""
    if module_name == MIMO_LANGUAGE_MODULE_KEY:
        return mimo_model.language_model
    submodules = mimo_model.modality_submodules
    # nn.ModuleDict has no ``get``; use membership + __getitem__.
    if submodules is not None and module_name in submodules:
        return submodules[module_name]
    return None


def setup_grid_mimo_ddp(model, args, wrap_with_ddp: bool = True):
    """Wrap each local grid submodule with per-module DDP (in place).

    Returns ``(is_grid, grid_state)`` (``grid_state`` is ``None`` when not
    grid).  Mirrors the Megatron-Bridge per-module DDP wiring: each submodule
    the rank participates in is **replaced in place** by its
    ``DistributedDataParallel`` wrapper, so the MIMO forward, the optimizer
    builder (``get_mimo_optimizer``) and ``MimoModel.sharded_state_dict`` all
    see the wrapped modules.  The wrappers are additionally aliased as
    ``language_ddp`` / ``vision_ddp`` via ``object.__setattr__`` (NOT
    registered as nn children - that would emit duplicate parameter keys into
    torch_dist checkpoints) so the colocated DDP helpers keep working.
    """
    # Keep package imports independent of the full megatron.training stack.
    from megatron.training.utils import print_rank_0

    unwrapped_model = unwrap_model(model)
    mimo_model = (
        unwrapped_model[0]
        if isinstance(unwrapped_model, list) and len(unwrapped_model) == 1
        else (unwrapped_model if not isinstance(unwrapped_model, list) else None)
    )
    grid_state = getattr(mimo_model, "mimo_grid_state", None) if mimo_model is not None else None
    is_grid = (
        getattr(args, "mimo_layout", "colocated") == "grid"
        and wrap_with_ddp
        and grid_state is not None
    )
    if not is_grid:
        return False, None

    print_rank_0("Non-colocated grid MIMO: wrapping local modules with per-module DDP.")

    module_to_grid_tuple: list[tuple] = []
    module_to_ddp: dict[str, DDP] = {}
    for module_name, pg_collection in grid_state.infra.module_to_pg_collection.items():
        if pg_collection is None:
            continue
        module = _grid_module_from_model(mimo_model, module_name)
        if module is None:
            raise RuntimeError(
                f"Rank {dist.get_rank()}: module '{module_name}' has a process "
                "group collection but the model has no such submodule."
            )
        dp_size = pg_collection.dp.size()
        ddp_config = build_mimo_ddp_config(args, module, dp_world_size=dp_size)
        # Per-module DDP needs a transformer config: language carries its
        # own, images gets the vision config threaded by its submodule spec;
        # fall back to the model config (``get_model_config`` rejects lists).
        module_config = getattr(module, "config", None)
        if module_config is None and mimo_model is not None:
            module_config = get_model_config(mimo_model)
        ddp = DDP(
            config=module_config,
            ddp_config=ddp_config,
            module=module,
            pg_collection=pg_collection,
        )
        # MCore DDP does not proxy module methods; proxy set_input_tensor,
        # which the MIMO forward calls on non-first PP stages (Bridge does
        # the same).
        if hasattr(module, "set_input_tensor"):
            ddp.set_input_tensor = module.set_input_tensor

        if module_name == MIMO_LANGUAGE_MODULE_KEY:
            mimo_model.language_model = ddp
        else:
            mimo_model.modality_submodules[module_name] = ddp

        # Colocated helpers (get_mimo_ddp_wrappers / set_mimo_force_all_reduce
        # / patch_mimo_model_chunk) key on these names; object.__setattr__
        # keeps them out of named_children() (the wrapper is already a child).
        if module_name == MIMO_LANGUAGE_MODULE_KEY:
            object.__setattr__(mimo_model, "language_ddp", ddp)
        else:
            object.__setattr__(mimo_model, "vision_ddp", ddp)
        module_to_grid_tuple.append((ddp, grid_state.infra.module_to_grid_map[module_name]))
        module_to_ddp[module_name] = ddp

    object.__setattr__(mimo_model, "module_to_ddp", module_to_ddp)
    grid_state.module_to_grid_tuple = module_to_grid_tuple
    for model_chunk in model:
        patch_mimo_model_chunk(model_chunk)
    return True, grid_state


def build_grid_multimodule_communicator(
    grid_state: GridTrainingState,
    model,
) -> MultiModulePipelineCommunicator:
    """Build the multi-module pipeline communicator for the grid path.

    Uses the language transformer config (``pipeline_dtype``, timers, ...) as
    the shared schedule config; vision PP is 1 by construction.
    """
    config = get_model_config(model)
    if config.pipeline_dtype is None:
        if getattr(config, "bf16", False):
            config.pipeline_dtype = torch.bfloat16
        elif getattr(config, "fp16", False):
            config.pipeline_dtype = torch.float16
        else:
            config.pipeline_dtype = torch.float32
    contract = get_grid_communicator_contract()
    communicator = MultiModulePipelineCommunicator(
        grid_state.infra.module_to_grid_map,
        contract.topology,
        config,
        dim_mapping=contract.dim_mapping,
        module_output_ndim=contract.module_output_ndim,
    )
    grid_state.multimodule_communicator = communicator
    return communicator


def finalize_grid_training_state(grid_state: GridTrainingState) -> None:
    """Fill the schedule-facing pieces of ``grid_state`` (idempotent)."""
    if grid_state not in _GRID_TRAINING_STATES:
        _GRID_TRAINING_STATES.append(grid_state)
    if grid_state.multimodule_pg_collection is None:
        grid_state.multimodule_pg_collection = build_pg_collection_for_schedule(
            grid_state.infra.module_to_pg_collection
        )
    active_module_name, local_pg_collection = get_active_module_pg(
        grid_state.infra.module_to_pg_collection
    )
    grid_state.active_module_name = active_module_name
    grid_state.local_pg_collection = local_pg_collection


def configure_grid_model_config_hooks(grid_state: GridTrainingState, model) -> None:
    """Bind per-module grad-sync hooks on the model config.

    Mirrors the standard path's ``no_sync_func`` / ``finalize_model_grads_func``
    wiring but with per-module process groups (``multimodule_no_sync`` /
    ``finalize_model_grads_multimodule``).  ``grad_scale_func`` is left to the
    training loop (it binds ``optimizer.scale_loss``).
    """
    config = get_model_config(model)
    module_to_grid_map = grid_state.infra.module_to_grid_map
    pg_collections = grid_state.infra.module_to_pg_collection
    module_to_grid_tuple = grid_state.module_to_grid_tuple

    config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)
    config.finalize_model_grads_func = partial(
        finalize_model_grads_multimodule,
        module_to_grid_map=module_to_grid_map,
        pg_collections=pg_collections,
        module_to_grid_tuple=module_to_grid_tuple,
    )
    if not config.variable_seq_lengths:
        raise ValueError(
            "Non-colocated grid MIMO requires variable_seq_lengths=True "
            "(enable_variable_seq_lengths in the config); the multi-module "
            "pipeline schedule exchanges tensor shapes dynamically."
        )


def build_grid_optimizer(mimo_model, optimizer_config) -> object:
    """Build the MCore ``MimoOptimizer`` for the grid path (module-namespaced).

    MCore builds one inner optimizer per module, each bound to its module's
    process groups.  The inner ``DistributedOptimizer`` shard keys themselves
    are *not* namespaced - both modules emit identical
    ``optimizer.distributed.dp_group_idx_<mp_rank>.*`` keys - so with
    heterogeneous module layouts (e.g. images TP2 on ranks [0, 2) and
    language TP1/DP6 on [2, 8)) the same key describes *different* global
    tensors on different ranks and the torch_dist save-time sharding
    validation fails with duplicate ShardedObject keys and ShardedTensor
    global-shape mismatches.  The MCore optimizer is therefore wrapped in
    :class:`GridMimoOptimizer`, which namespaces those shard keys by module
    on save and strips the namespace again on load.
    """
    mimo_optimizer = get_mimo_optimizer(mimo_model, optimizer_config)
    return GridMimoOptimizer(mimo_optimizer.module_infos, mimo_optimizer.config)


#: Shard-key prefix emitted by ``DistributedOptimizer``; carries no module
#: information and collides across a MIMO model's per-module optimizers.
_OPT_DIST_KEY_PREFIX = "optimizer.distributed."


def _namespace_module_opt_keys(module_sharded_sd, module_name: str) -> None:
    """Insert ``module_name`` into ``optimizer.distributed.*`` shard keys (in place).

    ``optimizer.distributed.dp_group_idx_0.optimizer`` becomes
    ``optimizer.distributed.<module_name>.dp_group_idx_0.optimizer``.  Keys
    that already carry module information (``optimizer.mimo.*`` extracted by
    MCore, ``optimizer.state.*`` model-space keys) are left untouched;
    idempotent for the same module name.
    """
    namespaced_prefix = f"{_OPT_DIST_KEY_PREFIX}{module_name}."

    def _rename(sh_base):
        if (
            isinstance(sh_base, ShardedBase)
            and sh_base.key.startswith(_OPT_DIST_KEY_PREFIX)
            and not sh_base.key.startswith(namespaced_prefix)
        ):
            sh_base.key = f"{namespaced_prefix}{sh_base.key[len(_OPT_DIST_KEY_PREFIX) :]}"
        return sh_base

    dict_list_map_inplace(_rename, module_sharded_sd)


def _unnamespace_module_opt_keys(module_sharded_sd, module_name: str) -> None:
    """Strip the module namespace inserted by :func:`_namespace_module_opt_keys`.

    Only ShardedBase objects are touched (after a distributed load the state
    dict contains plain tensors, so this is normally a no-op); plain-data
    entries are left as-is.
    """
    namespaced_prefix = f"{_OPT_DIST_KEY_PREFIX}{module_name}."

    def _rename(sh_base):
        if isinstance(sh_base, ShardedBase) and sh_base.key.startswith(namespaced_prefix):
            sh_base.key = f"{_OPT_DIST_KEY_PREFIX}{sh_base.key[len(namespaced_prefix) :]}"
        return sh_base

    dict_list_map_inplace(_rename, module_sharded_sd)


class GridMimoOptimizer(MimoOptimizer):
    """MCore ``MimoOptimizer`` with module-namespaced distributed-optimizer keys.

    MCore nests each module's optimizer state under ``{module_name: module_sd}``
    but leaves the inner ``optimizer.distributed.dp_group_idx_*`` shard keys
    identical across modules - and the shard key (not the dict nesting) is
    what the torch_dist checkpoint uses to identify global tensors, so the
    images and language optimizers collide at save-time validation.  This
    subclass namespaces those keys per module on save
    (``optimizer.distributed.<module>.dp_group_idx_*``) and restores the
    un-namespaced form before each inner optimizer's ``load_state_dict`` on
    load, keeping save and load symmetric without touching MCore.
    """

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        sharded_state = super().sharded_state_dict(model_sharded_state_dict, is_loading, **kwargs)
        for module_name, module_sd in sharded_state.items():
            _namespace_module_opt_keys(module_sd, module_name)
        return sharded_state

    def load_state_dict(self, state_dict: dict):
        if isinstance(state_dict, dict):
            for module_name, module_sd in state_dict.items():
                _unnamespace_module_opt_keys(module_sd, module_name)
        return super().load_state_dict(state_dict)


def sync_grid_optimizer_param_group_lr(optimizer, args) -> bool:
    """Re-sync per-param-group ``max_lr``/``min_lr`` after optimizer checkpoint load.

    Grid-mode resume with ``--override-opt-param-scheduler``: the checkpoint
    load restores every param group's ``max_lr``/``min_lr`` (and ``lr``), and
    ``OptimizerParamScheduler.get_lr`` prefers the param-group values over
    the scheduler's configured fields, so a configured ``lr``/``min_lr`` of 0
    (e.g. a zero-LR fine-tune) would be silently ignored after resume.  Walks
    the ``MimoOptimizer`` module nesting (skipping inactive modules) and any
    inner ``ChainedOptimizer``, resetting each active param group to
    ``args.lr`` / ``args.min_lr`` (the same values the scheduler keeps when
    overriding); no other hyperparameters are touched.  No-op (returns False)
    without the override flag or for a ``None`` optimizer (skip-train mode).
    """
    if optimizer is None or not getattr(args, "override_opt_param_scheduler", False):
        return False
    _reset_param_group_lr(optimizer, args.lr, args.min_lr)
    return True


def _reset_param_group_lr(optimizer, max_lr, min_lr) -> None:
    """Set ``max_lr``/``min_lr`` on every param group of ``optimizer``.

    Recurses through the public nesting surface used by the checkpointing
    code: ``MimoOptimizer.module_infos`` (``is_active`` gating) and Megatron's
    ``ChainedOptimizer.chained_optimizers``; anything else is treated as a
    plain Megatron optimizer exposing ``param_groups``.
    """
    if optimizer is None:
        return
    module_infos = getattr(optimizer, "module_infos", None)
    if module_infos is not None:
        for info in module_infos.values():
            if info.is_active and info.optimizer is not None:
                _reset_param_group_lr(info.optimizer, max_lr, min_lr)
        return
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained:
        for inner in chained:
            _reset_param_group_lr(inner, max_lr, min_lr)
        return
    for group in getattr(optimizer, "param_groups", None) or []:
        group["max_lr"] = max_lr
        group["min_lr"] = min_lr


def grid_training_state_from_model_chunk(model_chunk) -> GridTrainingState | None:
    """Read the grid training state from a (possibly wrapped) model chunk."""
    unwrapped = unwrap_model(model_chunk)
    if isinstance(unwrapped, list):
        unwrapped = unwrapped[0] if len(unwrapped) == 1 else None
    if unwrapped is None:
        return None
    return getattr(unwrapped, "mimo_grid_state", None)
