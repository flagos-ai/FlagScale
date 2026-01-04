# This file applies the PT-D pipeline parallelism to the model.

import torch
import torch.nn as nn

from torch.distributed.pipelining.schedules import _PipelineSchedule


class ParallelDims:
    {}


class JobConfig:
    {}


class BaseModelArgs:
    {}


class ParallelizeFunction:
    {}


class LossFunction:
    {}


def pipeline(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: torch.device,
    model_args: BaseModelArgs,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    pass
