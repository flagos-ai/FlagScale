# Copyright (c) 2026, BAAI. All rights reserved.

"""Non-colocated grid training lifecycle helpers (FlagScale-native).

This module wires the grid prototype (``mimo_grid_utils`` /
``mimo_parallel_utils`` / ``mimo_grid_config``) into the FlagScale training
lifecycle, mirroring the Megatron-Bridge ``megatron_mimo`` setup but written
against the actual FlagScale / Megatron-LM-FL v0.18.2 APIs:

- :func:`setup_grid_mimo_ddp` — per-module DDP wrapping (one wrapper per
  module the rank participates in, no outer DDP) plus delegation of the
  DDP-ish methods on the outer Float16Module wrapper (reusing the colocated
  ``patch_mimo_model_chunk`` machinery so the training loop's
  duck-typed calls keep working).
- :func:`build_grid_multimodule_communicator` — the
  ``MultiModulePipelineCommunicator`` described by the registered model
  contract (module topology / tensor layout / output dims; see
  :mod:`.contracts`).
- :func:`configure_grid_model_config_hooks` — ``no_sync_func`` /
  ``finalize_model_grads_func`` bound to the per-module gradient helpers.
- :class:`GridTrainingState` — everything the training loop needs for grid
  mode, attached to the model chunk as ``mimo_grid_state``.

Design invariants (inherited from the prototype):

- Process groups are created once, by ``build_mimo_infra``, in deterministic
  global module order on every world rank; this module never creates PGs.
- Only modules whose ``ProcessGroupCollection`` is non-None on this rank get
  a DDP wrapper; stub ranks are rejected at config time.
- Gradient normalization uses per-module PGs (language is authoritative for
  token counts), see ``finalize_model_grads_multimodule``.
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
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.utils import get_model_config, unwrap_model

from ..ddp_utils import build_mimo_ddp_config, patch_mimo_model_chunk
from .contracts import get_grid_communicator_contract
from .data import (
    ModuleDataRole,
    drop_modality_inputs,
    slice_batch_for_module_dp,
)
from .runtime import (
    build_pg_collection_for_schedule,
    finalize_model_grads_multimodule,
    get_active_module_pg,
    multimodule_no_sync,
)

if TYPE_CHECKING:
    from megatron.core.process_groups_config import (
        MultiModuleProcessGroupCollection,
        ProcessGroupCollection,
    )

    from .infra import MIMOInfra

logger = logging.getLogger(__name__)
_GRID_TRAINING_STATES: list[GridTrainingState] = []

#: Exact keyword arguments accepted by ``Qwen35GridMIMOModel.forward``.  The
#: grid forward step calls ``model(**data_batch)``, so the batch dict must
#: contain exactly these keys - leftover batch keys (``imgs``, ``videos``,
#: ``image_thw_grids``, ``video_thw_grids``, ...) would raise TypeError.
GRID_LANGUAGE_FORWARD_KEYS: tuple[str, ...] = (
    "input_ids",
    "position_ids",
    "attention_mask",
    "loss_mask",
    "labels",
    "modality_inputs",
    "packing_kwargs",
    "image_input_mask",
    "video_input_mask",
    "video_start_index",
)


def build_language_forward_kwargs(
    batch: dict[str, Any],
    *,
    dp_rank: int,
    dp_size: int,
    pp_rank: int,
    pp_size: int,
) -> dict[str, Any]:
    """Assemble the exact ``Qwen35GridMIMOModel.forward`` kwargs for a language rank.

    Language-only ranks (non-colocated) consume encoder outputs from the MIMO
    bridge, so the raw modality inputs are dropped before the module-local DP
    slice (their leading dimension is not the language sample batch).  The
    returned dict contains exactly :data:`GRID_LANGUAGE_FORWARD_KEYS`:

    - ``input_ids`` is only set on the first PP stage (embedding lives there),
    - ``labels`` / ``loss_mask`` only on the last PP stage (loss lives there),
    - ``modality_inputs`` / ``packing_kwargs`` are always ``None`` in grid mode,
    - ``image_input_mask`` / ``video_input_mask`` / ``video_start_index`` are
      kept for the deepstack / video fail-fast handling in the model forward.

    The batch keys the model does NOT accept (``imgs``, ``videos``,
    ``image_thw_grids``, ``video_thw_grids``, ...) are never emitted, even as
    nulled entries - the grid forward step splats the dict into the model call.

    The raw modality tensors are dropped BEFORE the sample-DP slice: they are
    patch-packed (``imgs`` dim 0 is the total patch count across the batch's
    images, not the sample count), so the generic sample-DP slicer must never
    see them (e.g. 3080 patches with language DP 6 is not divisible by 6).
    Only the sample-aligned keys (``tokens``, ``labels``, ``loss_mask``,
    ``position_ids``, the input masks, ...) are sliced by the module-local DP.
    """
    data_batch = drop_modality_inputs(batch)
    data_batch = slice_batch_for_module_dp(data_batch, dp_rank, dp_size)
    role = ModuleDataRole(module_name=MIMO_LANGUAGE_MODULE_KEY, pp_rank=pp_rank, pp_size=pp_size)
    image_input_mask = data_batch.get("image_input_mask")
    video_start_index = (
        int(image_input_mask.sum().item()) if torch.is_tensor(image_input_mask) else 0
    )
    return {
        "input_ids": data_batch.get("tokens") if role.is_first_stage else None,
        "position_ids": data_batch.get("position_ids"),
        "attention_mask": data_batch.get("attention_mask"),
        "loss_mask": data_batch.get("loss_mask") if role.is_last_stage else None,
        "labels": data_batch.get("labels") if role.is_last_stage else None,
        "modality_inputs": None,
        "packing_kwargs": None,
        "image_input_mask": image_input_mask,
        "video_input_mask": data_batch.get("video_input_mask"),
        "video_start_index": video_start_index,
    }


def build_vision_forward_kwargs(
    batch: dict[str, Any],
    *,
    dp_rank: int,
    dp_size: int,
) -> dict[str, Any]:
    """Assemble the exact ``Qwen35GridMIMOModel.forward`` kwargs for a vision rank.

    Vision ranks slice the global micro-batch for the vision module's DP
    (e.g. vision DP 2 layouts).  The raw modality tensors are patch-packed -
    ``imgs`` / ``videos`` dim 0 is the TOTAL patch count across the batch's
    images and ``image_thw_grids`` / ``video_thw_grids`` rows are the images -
    so they cannot be sliced by the sample dimension.  They are packed into
    the patch-packed dict form (``{hidden_states, grid_thw}``) that
    ``slice_batch_for_module_dp`` routes to the joint per-image slicer
    (``_slice_patch_packed_visual_dict``); every other (sample-aligned) key is
    sliced by sample as usual.

    Videos are not supported by the grid path yet - fail fast instead of
    producing a silent embedding-count mismatch.

    Returns:
        Dict with exactly the keys ``Qwen35GridMIMOModel.forward`` accepts,
        ``modality_inputs`` carrying the DP-sliced ``vision_data`` / ``grid_thw``
        for the ``qwen3_vit`` encoder.
    """
    image_data = batch.get("imgs")
    video_data = batch.get("videos")
    image_grid = batch.get("image_thw_grids")
    video_grid = batch.get("video_thw_grids")
    data_tensors = [t for t in (image_data, video_data) if torch.is_tensor(t)]
    grid_tensors = [t for t in (image_grid, video_grid) if torch.is_tensor(t)]
    if not data_tensors or not grid_tensors:
        raise ValueError(
            "Qwen3.5 non-colocated grid requires tensor-valued visual data and "
            "grid metadata; use empty tensors for text-only batches."
        )
    vision_data = torch.cat(data_tensors, dim=0)
    vision_grid = torch.cat(grid_tensors, dim=0)
    video_mask = batch.get("video_input_mask")
    if video_mask is not None and bool(video_mask.any().item()):
        raise NotImplementedError(
            "Qwen3.5 non-colocated grid: video inputs are not supported in "
            "this stage; the images modality covers image data only "
            "(fail-fast)."
        )

    image_mask = batch.get("image_input_mask")
    if torch.is_tensor(image_mask):
        samples_with_images = int(image_mask.any(dim=-1).sum().item())
        num_images = int(image_grid.size(0)) if torch.is_tensor(image_grid) else 0
        if samples_with_images != num_images:
            raise ValueError(
                "Qwen3.5 non-colocated grid currently requires at most one image per "
                "sample; image grid rows must match the number of samples containing "
                f"image tokens (images={num_images}, samples={samples_with_images})."
            )

    # Remove the raw modality keys (they are patch-packed and must not be
    # touched by the sample-DP slicer) and add the packed dict form in their
    # place; the slicer routes it to the joint per-image slicing.
    sliceable = {
        key: value
        for key, value in batch.items()
        if key not in ("imgs", "videos", "image_thw_grids", "video_thw_grids")
    }
    sliceable["vision_packed"] = {"hidden_states": vision_data, "grid_thw": vision_grid}
    sliced = slice_batch_for_module_dp(sliceable, dp_rank, dp_size)
    packed = sliced.pop("vision_packed")

    modality_inputs = (
        {
            "images": {
                "qwen3_vit": {
                    "vision_data": packed["hidden_states"],
                    "grid_thw": packed["grid_thw"],
                }
            }
        }
        if packed["grid_thw"] is not None and packed["grid_thw"].numel() > 0
        else None
    )
    return {
        "input_ids": sliced.get("tokens"),
        "position_ids": None,
        "attention_mask": None,
        "loss_mask": None,
        "labels": None,
        "modality_inputs": modality_inputs,
    }


def reconfigure_grid_num_microbatches_calculator(args) -> None:
    """Reconfigure the global num-microbatches calculator to grid-mode DP=1.

    The calculator is initialized at *parse time* (FlagScale's
    ``parse_and_validate_args`` -> ``set_global_variables``) with the YAML's
    global data parallel size: e.g. gbs=48 / mbs=6 with DP=4 (YAML TP2 on 8
    ranks) yields 2 microbatches.  Grid mode then forces the global parallel
    state to DP=1 (module-local DP slicing happens in the forward step), so
    the calculator must be updated to DP=1 *before* the grid batch contract
    (``qwen35_grid_data_contract``) and the schedule consume it - 48/6 with
    DP=1 yields 8 microbatches (8 * 6 == 48).

    Uses the public ``reconfigure_num_microbatches_calculator`` API with the
    same parameters the parse-time init used (only the DP - already forced to
    1 by the caller - differs); the global batch size, micro batch size and
    any step-batch-size schedule are preserved.  The colocated (non-grid) path
    never calls this helper and keeps the parse-time calculator untouched.
    """
    from megatron.core.num_microbatches_calculator import (
        reconfigure_num_microbatches_calculator,
    )

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


def validate_qwen35_grid_runtime_contract(args) -> None:
    """Reject runtime features not supported by the Qwen3.5 grid adapter."""
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
        infra: The built ``MIMOInfra`` (grids + nullable PG collections).
        module_to_grid_tuple: ``(ddp_module, grid)`` pairs for the modules this
            rank participates in (gradient sync / zero-buffer helpers).
        multimodule_pg_collection: Schedule PG collection
            (``MultiModuleProcessGroupCollection``).
        multimodule_communicator: Schedule P2P communicator.
        active_module_name: The single module this rank participates in.
        local_pg_collection: That module's ``ProcessGroupCollection``.
        world_size: Distributed world size.
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

    Returns ``(is_grid, grid_state)``: whether ``model`` is a non-colocated
    grid MIMO model whose per-module DDP setup was performed, and the
    :class:`GridTrainingState` (``None`` when not grid).

    Mirrors the Megatron-Bridge per-module DDP wiring: each submodule the rank
    participates in is **replaced in place** by its ``DistributedDataParallel``
    wrapper (``mimo_model.language_model`` /
    ``mimo_model.modality_submodules[...]``), so the MIMO forward, the
    optimizer builder (``get_mimo_optimizer``) and ``MimoModel.sharded_state_dict``
    (which unwraps DDP children) all see the wrapped modules.  The wrappers are
    additionally aliased as ``language_ddp`` / ``vision_ddp`` via
    ``object.__setattr__`` (NOT registered as nn children - that would emit
    duplicate parameter keys into torch_dist checkpoints) so the colocated
    ``get_mimo_ddp_wrappers`` / ``set_mimo_force_all_reduce`` /
    ``patch_mimo_model_chunk`` helpers keep working.
    """
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
        # Per-module DDP needs a transformer config.  The language module
        # carries its own; the images submodule gets the vision transformer
        # config threaded by ``build_qwen35_images_submodule_spec``.  Fall
        # back to the model config (the language config) for robustness.
        # ``get_model_config`` rejects lists, so pass the unwrapped chunk.
        module_config = getattr(module, "config", None)
        if module_config is None and mimo_model is not None:
            module_config = get_model_config(mimo_model)
        ddp = DDP(
            config=module_config,
            ddp_config=ddp_config,
            module=module,
            pg_collection=pg_collection,
        )
        # MCore's DDP does not proxy arbitrary module methods; the MIMO
        # forward path calls language_model.set_input_tensor on non-first PP
        # stages, so proxy it on the wrapper (Bridge does the same).
        if hasattr(module, "set_input_tensor"):
            ddp.set_input_tensor = module.set_input_tensor

        # Replace the submodule in place with its DDP wrapper.
        if module_name == MIMO_LANGUAGE_MODULE_KEY:
            mimo_model.language_model = ddp
        else:
            mimo_model.modality_submodules[module_name] = ddp

        # Colocated-compatible attribute names: get_mimo_ddp_wrappers /
        # set_mimo_force_all_reduce / patch_mimo_model_chunk key on these.
        # object.__setattr__ keeps them out of named_children() (the DDP
        # wrapper is already a child under its real submodule name).
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

    The MCore ``MimoOptimizer`` builds one inner optimizer per module, each
    bound to its module's process groups.  Its ``sharded_state_dict`` nests
    the per-module optimizer state under ``{module_name: module_sd}``, but the
    inner ``DistributedOptimizer`` shard keys themselves are *not* namespaced:
    both modules emit identical keys of the form
    ``optimizer.distributed.dp_group_idx_<mp_rank>.{...}`` (``dp_group_idx``
    is the module-local model-parallel rank).  With heterogeneous module
    layouts (e.g. images TP2 on ranks [0, 2) and language TP1/DP6 on
    [2, 8)), the same key then describes *different* global tensors on
    different ranks, so the torch_dist save-time sharding validation fails
    with duplicate ShardedObject keys and ShardedTensor global-shape
    mismatches.  This function wraps the MCore optimizer with
    :class:`GridMimoOptimizer`, which namespaces the
    ``optimizer.distributed.*`` shard keys by module on save and strips the
    namespace again on load.
    """
    mimo_optimizer = get_mimo_optimizer(mimo_model, optimizer_config)
    return GridMimoOptimizer(mimo_optimizer.module_infos, mimo_optimizer.config)


#: Prefix of the shard keys emitted by ``DistributedOptimizer`` (see
#: ``megatron.core.optimizer.distrib_optimizer``).  These keys carry no module
#: information and collide across the per-module optimizers of a MIMO model.
_OPT_DIST_KEY_PREFIX = "optimizer.distributed."


def _namespace_module_opt_keys(module_sharded_sd, module_name: str) -> None:
    """Insert ``module_name`` into ``optimizer.distributed.*`` shard keys (in place).

    ``optimizer.distributed.dp_group_idx_0.optimizer`` becomes
    ``optimizer.distributed.<module_name>.dp_group_idx_0.optimizer``.
    Keys that already carry module information (``optimizer.mimo.*`` extracted
    by MCore, ``optimizer.state.*`` model-space keys that embed the unique
    model parameter path) are left untouched.  The renaming is idempotent for
    the same module name.
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

    MCore's ``MimoOptimizer.sharded_state_dict`` nests each module's optimizer
    state dict under ``{module_name: module_sd}`` but leaves the inner
    ``optimizer.distributed.dp_group_idx_*`` shard keys identical across
    modules.  Because the shard key (not the dict nesting) is what the
    torch_dist checkpoint uses to identify global tensors, the images and
    language optimizers collide: the save-time validation reports duplicate
    ShardedObject keys and ShardedTensor global-shape mismatches for the same
    key.  This subclass namespaces those keys per module on save
    (``optimizer.distributed.<module>.dp_group_idx_*``) and restores the
    un-namespaced form before delegating to each inner optimizer's
    ``load_state_dict`` on load, so save and load stay symmetric without
    touching MCore.
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

    Grid-mode resume with ``--override-opt-param-scheduler``: the optimizer
    checkpoint load restores every param group's ``max_lr``/``min_lr`` (and
    ``lr``) from the saved state, and ``OptimizerParamScheduler.get_lr``
    prefers the param-group values over the scheduler's own configured fields,
    so a configured ``lr``/``min_lr`` of 0 (e.g. a zero-LR fine-tune) would be
    silently ignored after resume.  This walks the ``GridMimoOptimizer`` /
    ``MimoOptimizer`` module nesting (skipping inactive modules) and any inner
    ``ChainedOptimizer`` (e.g. MoE module optimizers), resetting each active
    inner optimizer param group's ``max_lr``/``min_lr`` to the configured
    override values (``args.lr`` / ``args.min_lr``) - the same values the
    scheduler keeps when overriding.  No other hyperparameters (``lr``,
    ``weight_decay``, ``wd_mult``, ...) are touched.

    No-op (returns False) when ``args.override_opt_param_scheduler`` is not
    set, mirroring the scheduler's own override semantics; also a no-op for a
    ``None`` optimizer (e.g. skip-train mode).
    """
    if optimizer is None or not getattr(args, "override_opt_param_scheduler", False):
        return False
    _reset_param_group_lr(optimizer, args.lr, args.min_lr)
    return True


def _reset_param_group_lr(optimizer, max_lr, min_lr) -> None:
    """Set ``max_lr``/``min_lr`` on every param group of ``optimizer``.

    Recurses through the public nesting surface used by the surrounding
    checkpointing code: ``MimoOptimizer.module_infos`` (one inner optimizer
    per module, ``is_active`` gating) and Megatron's
    ``ChainedOptimizer.chained_optimizers`` (one inner optimizer per
    parameter partition, e.g. dense/expert splits).  Anything else is
    treated as a plain Megatron optimizer exposing ``param_groups``.
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
