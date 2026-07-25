# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
import torch

from megatron.plugin.platform import get_platform
from functools import wraps

cur_platform = get_platform()

logger = logging.getLogger(__name__)


def get_device(local_rank=None):
    logger.info("FlagScale Plugin Decorator: Unsing overrided dist_singal_handler.get_device")
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if local_rank is None:
            device = torch.device(cur_platform.device_name())
        else:
            device = torch.device(f"{cur_platform.device_name()}:{local_rank}")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device
