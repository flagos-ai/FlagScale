# Copyright (c) 2025, BAAI. All rights reserved.
#
# Mainly adapted from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

import torch

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.nemo_bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.nemo_bridge.models.conversion.param_mapping import AutoMapping
from megatron.nemo_bridge.models.deepseek.common import (
    get_common_configs,
    get_common_mapping_list,
)
from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekV3ModelProvider
from megatron.nemo_bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(source="DeepseekV3ForCausalLM", target=GPTModel)
class DeepSeekV3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for DeepSeek-V3.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.nemo_bridge import AutoBridge
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

    def save_args_mg2hf(self, args, save_path):
        from transformers import DeepseekV3Config
        first_k_dense_replace = args.moe_layer_freq.index(1)
        seq_aux = True if args.moe_router_load_balancing_type == "seq_aux_loss" else False
        config = DeepseekV3Config(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.ffn_hidden_size,
            moe_intermediate_size=args.moe_ffn_hidden_size,
            num_hidden_layers=args.num_layers,
            num_nextn_predict_layers= args.mtp_num_layers if args.mtp_num_layers else 0 ,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_query_groups,
            n_shared_experts=args.moe_shared_expert_intermediate_size // args.moe_ffn_hidden_size,
            n_routed_experts=args.num_experts,
            routed_scaling_factor=args.moe_router_topk_scaling_factor,
            kv_lora_rank=args.kv_lora_rank,
            q_lora_rank=args.q_lora_rank if args.q_lora_rank else 0,
            qk_rope_head_dim=args.qk_pos_emb_head_dim,
            v_head_dim=args.v_head_dim,
            qk_nope_head_dim=args.qk_head_dim,
            n_group=args.moe_router_num_groups,
            topk_group=args.moe_router_group_topk,
            num_experts_per_tok=args.moe_router_topk,
            moe_layer_freq = 1,
            topk_method = "noaux_tc",
            first_k_dense_replace=first_k_dense_replace,
            scoring_func=args.moe_router_score_function,
            seq_aux=seq_aux,
            max_position_embeddings=args.max_position_embeddings,
            initializer_range=args.init_method_std,
            rms_norm_eps=args.norm_epsilon,
            tie_word_embeddings=not args.untie_embeddings_and_output_weights,
            rope_theta=args.rotary_base,
            attention_dropout=args.attention_dropout,
            torch_dtype=args.params_dtype,
        )

        auto_map = dict()
        auto_map["AutoConfig"] = "configuration_deepseek.DeepseekV3Config"
        auto_map["AutoModel"] = "modeling_deepseek.DeepseekV3Model"
        auto_map["AutoModelForCausalLM"] = "modeling_deepseek.DeepseekV3ForCausalLM"
        config.auto_map = auto_map
        config.architectures = ["DeepseekV3ForCausalLM"]
        config.save_pretrained(save_path)
        return config
