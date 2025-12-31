# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

import torch

from megatron.core.models.gpt.gpt_model import GPTModel

from flagscale.train.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from flagscale.train.bridge.models.conversion.model_bridge import MegatronModelBridge
from flagscale.train.bridge.models.deepseek.common import (
    get_common_configs,
    get_common_mapping_list,
)
from flagscale.train.bridge.models.deepseek.deepseek_provider import DeepSeekV2ModelProvider
from flagscale.train.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(source="DeepseekV2ForCausalLM", target=GPTModel)
class DeepSeekV2Bridge(MegatronModelBridge):
    """
    Megatron Bridge for DeepSeek-V2.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from flagscale.train.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV2ModelProvider:
        hf_config = hf_pretrained.config
        configs = get_common_configs(hf_pretrained)

        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        configs["make_vocab_size_divisible_by"] = 3200
        configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        provider = DeepSeekV2ModelProvider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()
        return MegatronMappingRegistry(*mapping_list)
