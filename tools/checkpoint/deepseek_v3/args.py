import json
import os
import sys

sys.path.append("..")
from moonlight_deepseek.configuration_deepseek import DeepseekV3Config


def load_args_hf2mg(args):
    # Read deepseek_v3 args.
    deepseek_v3_args_path = os.path.join(args.load, "config.json")
    with open(deepseek_v3_args_path) as f:
        deepseek_v3_args = json.load(f)

    # Update Megatron args.
    args.vocab_size = deepseek_v3_args["vocab_size"]
    args.hidden_size = deepseek_v3_args["hidden_size"]
    args.ffn_hidden_size = deepseek_v3_args["intermediate_size"]
    args.num_layers = deepseek_v3_args["num_hidden_layers"]
    args.num_attention_heads = deepseek_v3_args["num_attention_heads"]
    args.num_query_groups = deepseek_v3_args["num_key_value_heads"]
    hidden_act = deepseek_v3_args["hidden_act"]
    args.swiglu = True if hidden_act == "silu" else False
    args.max_position_embeddings = deepseek_v3_args["max_position_embeddings"]
    args.init_method_std = deepseek_v3_args["initializer_range"]
    args.norm_epsilon = deepseek_v3_args["rms_norm_eps"]
    args.untie_embeddings_and_output_weights = not deepseek_v3_args["tie_word_embeddings"]
    args.rotary_base = deepseek_v3_args["rope_theta"]
    args.disable_bias_linear = not deepseek_v3_args["attention_bias"]
    args.attention_dropout = deepseek_v3_args["attention_dropout"]
    args.qk_layernorm = True
    args.bf16 = deepseek_v3_args["torch_dtype"] == "bfloat16"
    args.fp16 = deepseek_v3_args["torch_dtype"] == "float16"
    args.apply_rope_fusion = False
    args.seq_length = 4096
    args.global_batch_size = 800
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.group_query_attention = True
    args.normalization = "RMSNorm"
    args.use_rotary_position_embeddings = True
    args.moe_router_load_balancing_type = "aux_loss"
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.make_vocab_size_divisible_by = 64
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.norm_has_bias = False
    args.tokenizer_type = "QwenTokenizerFS"

    # MoE Related
    args.moe_ffn_hidden_size = deepseek_v3_args["moe_intermediate_size"]
    n_shared_experts = deepseek_v3_args["n_shared_experts"]
    if n_shared_experts > 0:
        args.moe_shared_expert_intermediate_size = n_shared_experts * args.moe_ffn_hidden_size
    args.moe_grouped_gemm = True
    args.num_experts = deepseek_v3_args["n_routed_experts"]
    args.moe_router_topk_scaling_factor = deepseek_v3_args["routed_scaling_factor"]
    args.moe_router_num_groups = deepseek_v3_args["n_group"]
    args.moe_router_group_topk = deepseek_v3_args["topk_group"]
    args.moe_router_topk = deepseek_v3_args["num_experts_per_tok"]
    args.moe_layer_freq = deepseek_v3_args["moe_layer_freq"]
    # if set first k dense replace, then updating moe_layer_freq
    first_k_dense_replace = deepseek_v3_args["first_k_dense_replace"]
    args.moe_layer_freq = eval(
        "[0]*" + str(first_k_dense_replace) + "+[1]*" + str(args.num_layers - first_k_dense_replace)
    )
    args.moe_router_score_function = deepseek_v3_args["scoring_func"]
    if args.moe_router_score_function == "sigmoid":
        args.moe_router_enable_expert_bias = True
        args.moe_router_bias_update_rate = 0.001
    seq_aux = deepseek_v3_args["seq_aux"]
    if seq_aux:
        args.moe_router_load_balancing_type = "seq_aux_loss"

    # MLA Related
    if deepseek_v3_args["q_lora_rank"] != "null":
        args.q_lora_rank = deepseek_v3_args["q_lora_rank"]
    args.kv_lora_rank = deepseek_v3_args["kv_lora_rank"]
    args.qk_head_dim = deepseek_v3_args["qk_nope_head_dim"]
    args.qk_pos_emb_head_dim = deepseek_v3_args["qk_rope_head_dim"]
    args.v_head_dim = deepseek_v3_args["v_head_dim"]
    args.multi_latent_attention = True

    # MTP Related
    # MTP is used in DeepSeek V3
    if "num_nextn_predict_layers" in deepseek_v3_args:
        args.mtp_num_layers = deepseek_v3_args["num_nextn_predict_layers"]

    return args, args


def save_args_mg2hf(args):
    first_k_dense_replace = args.moe_layer_freq.index(1)
    seq_aux = True if args.moe_router_load_balancing_type == "seq_aux_loss" else False
    config = DeepseekV3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        moe_intermediate_size=args.moe_ffn_hidden_size,
        num_hidden_layers=args.num_layers,
        num_nextn_predict_layers=args.mtp_num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups,
        n_shared_experts=args.moe_shared_expert_intermediate_size // args.moe_ffn_hidden_size,
        n_routed_experts=args.num_experts,
        routed_scaling_factor=args.moe_router_topk_scaling_factor,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        qk_rope_head_dim=args.qk_pos_emb_head_dim,
        v_head_dim=args.v_head_dim,
        qk_nope_head_dim=args.qk_head_dim,
        n_group=args.moe_router_num_groups,
        topk_group=args.moe_router_group_topk,
        num_experts_per_tok=args.moe_router_topk,
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
    config.save_pretrained(args.save)

    return config
