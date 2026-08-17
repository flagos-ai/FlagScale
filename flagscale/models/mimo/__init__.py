# Copyright (c) 2025, BAAI. All rights reserved.

"""MIMO building blocks: colocated in-house path + MCore bridge adapter.

- ``colocated/``: the production colocated MIMO implementation (macro/micro
  batch scheduling, delayed ViT backward), behind ``--mimo-layout=colocated``.
- ``bridge/``: the Megatron-Bridge-style adapter layer over Megatron-LM-FL's
  ``megatron.core`` MIMO infrastructure (MimoModel / HyperCommGrid /
  MultiModulePipelineCommunicator / MimoOptimizer), behind
  ``--mimo-layout=grid``; supports non-colocated component layouts.
- ``ddp_utils``: per-module DDP helpers shared by both paths.

The package-level exports intentionally remain the colocated public
API for backward compatibility; bridge consumers import from
``flagscale.models.mimo.bridge`` explicitly.
"""

from .colocated import (
    ChainedOptimizer,
    ColocatedMIMOModel,
    ModuleParallelismConfig,
    build_colocated_pg_collections,
    build_mimo_optimizer,
    compute_microbatch_token_counts,
    drop_mimo_completed_macros,
    release_mimo_training_state,
    set_mimo_force_all_reduce,
    setup_mimo_ddp,
    switch_parallel_state,
    validate_mimo_config,
)

__all__ = [
    "ModuleParallelismConfig",
    "validate_mimo_config",
    "build_colocated_pg_collections",
    "switch_parallel_state",
    "ColocatedMIMOModel",
    "ChainedOptimizer",
    "setup_mimo_ddp",
    "build_mimo_optimizer",
    "set_mimo_force_all_reduce",
    "compute_microbatch_token_counts",
    "drop_mimo_completed_macros",
    "release_mimo_training_state",
]
