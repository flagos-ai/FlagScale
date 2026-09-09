# Copyright (c) 2026, BAAI. All rights reserved.

"""Non-colocated grid layer for MCore-based MIMO training.

Adapts FlagScale's launcher/config/model/data/training lifecycle onto
Megatron-LM-FL's ``megatron.core`` MIMO infrastructure (``MimoModel`` /
``HyperCommGrid``) instead of re-implementing it.  Submodules: ``parallelism``
(layout config and generic layout algorithms, stdlib-only/torch-free),
``infra`` (grid and per-module PG construction), ``runtime``, ``data``,
``training`` (grid wiring).  Per-model builders live in ``providers/``; they
register their communicator contract and batch preparer here at import time
and are imported by the model entry, never by this package.
"""

from .data import (
    ModuleDataRole,
    SamplingInfo,
    drop_modality_inputs,
    get_sampling_info,
    is_patch_packed_visual_dict,
    needs_data_for_role,
    prepare_batch_for_module,
    should_drop_modality_inputs,
    slice_batch_for_module_dp,
)
from .infra import (
    MODULE_GRID_DIM_NAMES,
    MIMOInfra,
    ModuleGridConfig,
    build_mimo_infra,
    build_module_grids,
    build_module_pg_collections,
    create_module_pg_collection,
    grids_are_colocated,
    set_per_module_random_seed,
)
from .parallelism import (
    LANGUAGE_MODULE_NAME,
    MIMOLayout,
    MIMOParallelismConfig,
    classify_layout,
    compute_pipeline_layer_split,
    describe_grid_modules,
    parse_module_parallelism,
    parse_module_parallelisms,
    resolve_module_sequence_parallel,
    validate_grid_batch_divisibility,
)
from .runtime import (
    build_pg_collection_for_schedule,
    finalize_model_grads_multimodule,
    get_active_module_pg,
    get_module_to_grid_tuple,
    is_current_rank_in_grid,
    multimodule_no_sync,
    unwrap_mimo_model,
    validate_data_loader_contract,
    validate_no_stub_ranks,
    zero_grad_buffer_for_multimodule,
)
from .training import (
    GridTrainingState,
    apply_grid_parse_time_contract,
    build_grid_multimodule_communicator,
    build_grid_optimizer,
    configure_grid_model_config_hooks,
    destroy_grid_training_states,
    finalize_grid_training_state,
    get_grid_batch_preparer,
    get_logical_iteration_samples,
    grid_training_state_from_model_chunk,
    prepare_grid_batch,
    register_grid_batch_preparer,
    reconfigure_grid_num_microbatches_calculator,
    setup_grid_mimo_ddp,
    sync_grid_optimizer_param_group_lr,
    validate_grid_runtime_contract,
)

__all__ = [
    # parallelism
    "LANGUAGE_MODULE_NAME",
    "MIMOLayout",
    "MIMOParallelismConfig",
    "classify_layout",
    "compute_pipeline_layer_split",
    "describe_grid_modules",
    "parse_module_parallelism",
    "parse_module_parallelisms",
    "resolve_module_sequence_parallel",
    "validate_grid_batch_divisibility",
    # infra
    "MODULE_GRID_DIM_NAMES",
    "ModuleGridConfig",
    "MIMOInfra",
    "build_module_grids",
    "create_module_pg_collection",
    "build_module_pg_collections",
    "build_mimo_infra",
    "grids_are_colocated",
    "set_per_module_random_seed",
    # runtime
    "unwrap_mimo_model",
    "is_current_rank_in_grid",
    "get_active_module_pg",
    "get_module_to_grid_tuple",
    "build_pg_collection_for_schedule",
    "multimodule_no_sync",
    "finalize_model_grads_multimodule",
    "zero_grad_buffer_for_multimodule",
    "validate_no_stub_ranks",
    "validate_data_loader_contract",
    # data
    "ModuleDataRole",
    "SamplingInfo",
    "needs_data_for_role",
    "get_sampling_info",
    "slice_batch_for_module_dp",
    "is_patch_packed_visual_dict",
    "should_drop_modality_inputs",
    "drop_modality_inputs",
    "prepare_batch_for_module",
    # training
    "GridTrainingState",
    "setup_grid_mimo_ddp",
    "build_grid_multimodule_communicator",
    "build_grid_optimizer",
    "configure_grid_model_config_hooks",
    "destroy_grid_training_states",
    "get_logical_iteration_samples",
    "prepare_grid_batch",
    "register_grid_batch_preparer",
    "get_grid_batch_preparer",
    "apply_grid_parse_time_contract",
    "validate_grid_runtime_contract",
    "reconfigure_grid_num_microbatches_calculator",
    "finalize_grid_training_state",
    "grid_training_state_from_model_chunk",
    "sync_grid_optimizer_param_group_lr",
]
