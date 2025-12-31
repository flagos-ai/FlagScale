# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

from flagscale.train.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from flagscale.train.bridge.models.hf_pretrained.vlm import PreTrainedVLM

__all__ = ["PreTrainedCausalLM", "PreTrainedVLM"]
