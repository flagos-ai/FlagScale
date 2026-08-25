# Copyright (c) 2025, BAAI. All rights reserved.

"""Generic MIMO building blocks for colocated deployment."""

from .hetero_pg_utils import build_colocated_pg_collections
from .mimo_config import ModuleParallelismConfig, validate_mimo_config
from .mimo_model import ColocatedMIMOModel
from .mimo_optimizer import (
    ChainedOptimizer,
    build_mimo_optimizer,
    set_mimo_force_all_reduce,
    setup_mimo_ddp,
)
from .mimo_utils import (
    compute_microbatch_token_counts,
    drop_mimo_completed_macros,
    release_mimo_training_state,
)
from .parallel_state_ctx import switch_parallel_state

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
