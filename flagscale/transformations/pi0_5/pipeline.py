# This file applies the PT-D pipeline parallelism to the Pi0.5 model.

import torch
import torch.nn as nn

from torch.distributed.pipelining.schedules import _PipelineSchedule


class ParallelDims:
    {}


class JobConfig:
    {}


class BaseModelArgs:
    {}
