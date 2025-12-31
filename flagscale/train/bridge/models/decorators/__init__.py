# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

from flagscale.train.bridge.models.decorators.dispatch import dispatch
from flagscale.train.bridge.models.decorators.torchrun import torchrun_main

__all__ = ["dispatch", "torchrun_main"]
