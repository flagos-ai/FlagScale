# Copyright (c) 2025, BAAI. All rights reserved.

"""Per-module DDP helpers shared by the colocated and grid MIMO paths.

Both execution paths wrap each local submodule with its own
``DistributedDataParallel`` instance and skip the outer DDP wrapper, so the
DDP-config construction and the outer-chunk method patching live here once.
"""

import dataclasses
import types
from contextlib import ExitStack

import megatron.core.parallel_state as mpu
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.utils import unwrap_model


def build_mimo_ddp_config(
    args, model, dp_world_size: int | None = None
) -> DistributedDataParallelConfig:
    """Build a ``DistributedDataParallelConfig`` matching Megatron's default path.

    Kept in sync with ``megatron.training.training.get_model`` so MIMO modules
    see the same DDP behavior as a non-MIMO model.  ``dp_world_size``
    overrides the default bucket-size heuristic; use it when the module's
    data-parallel size differs from the global default.
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


def get_mimo_ddp_wrappers(model_chunk):
    """Return per-module DDP wrappers in deterministic module order."""
    unwrapped = unwrap_model(model_chunk)
    module_to_ddp = getattr(unwrapped, "module_to_ddp", None)
    if module_to_ddp is not None:
        return [ddp for ddp in module_to_ddp.values() if ddp is not None]

    ddps = []
    for attr in ("vision_ddp", "language_ddp"):
        ddp = getattr(unwrapped, attr, None)
        if ddp is not None:
            ddps.append(ddp)
    return ddps


def patch_mimo_model_chunk(model_chunk):
    """Bind DDP-like grad-sync and param-sync methods on the outer Float16Module wrapper.

    MIMO skips the outer DDP wrapper, so the training loop / Megatron helpers
    that call ``model_chunk.finish_grad_sync()`` etc. would otherwise fail;
    the methods delegate to the inner vision/language DDP modules.
    """
    ddp_wrappers = get_mimo_ddp_wrappers(model_chunk)
    if ddp_wrappers:
        # The language DDP config (last wrapper) is representative; vision
        # uses the same settings.
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

    # Expose the exit-time training-state cleanup on the outer wrapper for the
    # training loop's duck-typed pre-save cleanup (hasattr checks in
    # ``training.py``).  Bound by plain attribute assignment (not
    # ``types.MethodType``) so ``self`` stays the unwrapped model, which owns
    # ``self.scheduler``.
    unwrapped = unwrap_model(model_chunk)
    for name in ("release_training_state", "drop_completed_macros"):
        if hasattr(unwrapped, name):
            setattr(model_chunk, name, getattr(unwrapped, name))
