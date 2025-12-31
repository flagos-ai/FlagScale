# Copyright (c) 2025, BAAI. All rights reserved.
#
# Copied from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

import torch

from megatron.core.models.gpt.gpt_model import GPTModel

from flagscale.train.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from flagscale.train.bridge.models.conversion.model_bridge import MegatronModelBridge
from flagscale.train.bridge.models.conversion.param_mapping import AutoMapping
from flagscale.train.bridge.models.deepseek.common import (
    get_common_configs,
    get_common_mapping_list,
)
from flagscale.train.bridge.models.deepseek.deepseek_provider import DeepSeekV3ModelProvider
from flagscale.train.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(source="DeepseekV3ForCausalLM", target=GPTModel)
class DeepSeekV3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for DeepSeek-V3.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from flagscale.train.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3ModelProvider:
        hf_config = hf_pretrained.config
        configs = get_common_configs(hf_pretrained)

        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        # aux_loss_alpha is not set in all DSv3 HF configs
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        # TODO: mtp

        provider = DeepSeekV3ModelProvider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()

        param_mappings = {
            # expert bias
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias"
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        return MegatronMappingRegistry(*mapping_list)
