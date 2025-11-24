# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the model.

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh


class ParallelDims:
    {}


class JobConfig:
    {}


class CompileConfig:
    {}


def parallelize(model: nn.Module, parallel_dims: ParallelDims, job_config: JobConfig):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.
    """
    pass


def apply_tp(
    model: nn.Module, tp_mesh: DeviceMesh, loss_parallel: bool, enable_float8_tensorwise_tp: bool
):
    """Apply tensor parallelism."""
    pass


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    pass


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    pass


def apply_ddp(
    model: nn.Module, dp_mesh: DeviceMesh, enable_compile: bool, enable_compiled_autograd: bool
):
    pass
