# Copyright (c) 2025, BAAI. All rights reserved.

"""MIMO building blocks: colocated in-house path + non-colocated grid path.

- ``colocated/``: production colocated MIMO implementation (macro/micro batch
  scheduling, delayed ViT backward), behind ``--mimo-layout=colocated``.
- ``grid/``: non-colocated adapter layer over Megatron-LM-FL's
  ``megatron.core`` MIMO infrastructure (``MimoModel`` / ``HyperCommGrid``),
  behind ``--mimo-layout=grid``; supports per-module (disjoint) layouts.
- ``ddp_utils``: per-module DDP helpers shared by both paths.
- ``lifecycle``: the unified facade every consumer outside this package uses
  (training loop, training entry, argument parsing) — layout dispatch lives
  there, never in the callers.
"""

from .lifecycle import (
    apply_parse_time_contract,
    build_mimo_optimizer,
    configure_model_config_hooks,
    destroy_mimo_training_states,
    drop_mimo_completed_macros,
    get_dataloader_shard_policy,
    get_logical_iteration_samples,
    get_mimo_forward_backward_func,
    get_mimo_loss_reduction_context,
    prepare_mimo_batch,
    release_mimo_training_state,
    set_mimo_force_all_reduce,
    setup_mimo_ddp,
    setup_mimo_runtime,
    sync_optimizer_param_group_lr,
)

__all__ = [
    # lifecycle facade (the public API of this package)
    "apply_parse_time_contract",
    "setup_mimo_runtime",
    "setup_mimo_ddp",
    "build_mimo_optimizer",
    "sync_optimizer_param_group_lr",
    "configure_model_config_hooks",
    "get_mimo_forward_backward_func",
    "get_mimo_loss_reduction_context",
    "get_logical_iteration_samples",
    "destroy_mimo_training_states",
    "release_mimo_training_state",
    "drop_mimo_completed_macros",
    "set_mimo_force_all_reduce",
    "prepare_mimo_batch",
    "get_dataloader_shard_policy",
]
