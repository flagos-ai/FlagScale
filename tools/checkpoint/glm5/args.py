import json
import os
import sys

sys.path.append("..")

try:
    import transformers

    major, minor, _ = map(int, transformers.__version__.split("."))
    assert major >= 5 and minor >= 12, "transformers version >= 5.12.1 is required"
except:
    raise ImportError(
        "transformers version >= 5.12.1 is required, please install it via `pip install 'transformers>=5.12.1'"
    )


def load_args_hf2mg(args):
    """Load GLM5 HuggingFace config and map to Megatron args."""
    glm5_args_path = os.path.join(args.load, "config.json")
    with open(glm5_args_path) as f:
        glm5_args = json.load(f)

    # Basic model args
    args.vocab_size = glm5_args["vocab_size"]
    args.hidden_size = glm5_args["hidden_size"]
    args.ffn_hidden_size = glm5_args["intermediate_size"]
    args.num_layers = glm5_args["num_hidden_layers"]
    args.num_attention_heads = glm5_args["num_attention_heads"]
    args.num_query_groups = glm5_args.get("num_key_value_heads", args.num_attention_heads)
    hidden_act = glm5_args.get("hidden_act", "silu")
    args.swiglu = True if hidden_act == "silu" else False
    args.max_position_embeddings = glm5_args["max_position_embeddings"]
    args.init_method_std = glm5_args.get("initializer_range", 0.02)
    args.layernorm_epsilon = glm5_args.get("rms_norm_eps", 1e-5)
    args.untie_embeddings_and_output_weights = not glm5_args.get("tie_word_embeddings", False)
    args.disable_bias_linear = not glm5_args.get("attention_bias", False)
    args.attention_dropout = glm5_args.get("attention_dropout", 0.0)

    # RoPE
    rope_params = glm5_args.get("rope_parameters", {})
    args.rotary_base = rope_params.get("rope_theta", 1000000)
    args.apply_rope_fusion = False
    args.use_rotary_position_embeddings = True
    args.add_position_embedding = False
    args.position_embedding_type = "rope"

    # Precision & misc
    dtype = glm5_args.get("dtype", "bfloat16")
    args.bf16 = dtype == "bfloat16"
    args.fp16 = dtype == "float16"
    args.normalization = "RMSNorm"
    args.qk_layernorm = True
    args.group_query_attention = False
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.make_vocab_size_divisible_by = 64
    args.seq_length = 4096
    args.global_batch_size = 128
    args.iteration = 1
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.norm_has_bias = False
    args.tokenizer_type = "HuggingFaceTokenizer"

    # MLA Related
    args.multi_latent_attention = True
    q_lora_rank = glm5_args.get("q_lora_rank")
    if q_lora_rank and q_lora_rank != "null":
        args.q_lora_rank = q_lora_rank
    else:
        args.q_lora_rank = None
    args.kv_lora_rank = glm5_args["kv_lora_rank"]
    args.qk_head_dim = glm5_args["qk_nope_head_dim"]
    args.qk_pos_emb_head_dim = glm5_args["qk_rope_head_dim"]
    args.v_head_dim = glm5_args["v_head_dim"]

    # MoE Related
    args.moe_ffn_hidden_size = glm5_args["moe_intermediate_size"]
    n_shared_experts = glm5_args.get("n_shared_experts", 1)
    if n_shared_experts > 0:
        args.moe_shared_expert_intermediate_size = n_shared_experts * args.moe_ffn_hidden_size
    args.moe_grouped_gemm = True
    args.num_experts = glm5_args["n_routed_experts"]
    args.moe_router_topk_scaling_factor = glm5_args.get("routed_scaling_factor", 2.5)
    args.moe_router_num_groups = glm5_args.get("n_group", 1)
    args.moe_router_group_topk = glm5_args.get("topk_group", 1)
    args.moe_router_topk = glm5_args["num_experts_per_tok"]
    args.moe_router_load_balancing_type = "none"
    args.moe_router_score_function = glm5_args.get("scoring_func", "sigmoid")
    if args.moe_router_score_function == "sigmoid":
        args.moe_router_enable_expert_bias = True
        args.moe_router_bias_update_rate = 0.001

    # moe_layer_freq: first_k_dense_replace layers are dense, rest are MoE
    first_k_dense_replace = glm5_args.get("first_k_dense_replace", 3)
    args.moe_layer_freq = [0] * first_k_dense_replace + [1] * (
        args.num_layers - first_k_dense_replace
    )

    # DSA (Dynamic Sparse Attention) Indexer Related
    args.experimental_attention_variant = "dsa"
    args.dsa_indexer_n_heads = glm5_args.get("index_n_heads", 32)
    args.dsa_indexer_head_dim = glm5_args.get("index_head_dim", 128)
    args.dsa_indexer_topk = glm5_args.get("index_topk", 2048)

    # indexer_types: per-layer indexer type ("full" or "shared")
    args.indexer_types = glm5_args.get("indexer_types", None)

    # MTP (Multi-Token Prediction) Related
    mtp_num_layers = glm5_args.get("num_nextn_predict_layers", 0)
    args.mtp_num_layers = mtp_num_layers if mtp_num_layers else 0

    return args, args


def save_args_mg2hf(args):
    """Construct HF config from Megatron args and save."""
    first_k_dense_replace = (
        args.moe_layer_freq.index(1) if 1 in args.moe_layer_freq else args.num_layers
    )
    mtp_num_layers = getattr(args, "mtp_num_layers", 0) or 0

    config_dict = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.ffn_hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": getattr(args, "num_query_groups", args.num_attention_heads),
        "hidden_act": "silu" if args.swiglu else "gelu",
        "max_position_embeddings": args.max_position_embeddings,
        "initializer_range": args.init_method_std,
        "rms_norm_eps": args.layernorm_epsilon,
        "tie_word_embeddings": not args.untie_embeddings_and_output_weights,
        "attention_bias": not getattr(args, "disable_bias_linear", True),
        "attention_dropout": getattr(args, "attention_dropout", 0.0),
        "dtype": "bfloat16" if args.bf16 else ("float16" if args.fp16 else "float32"),
        "rope_parameters": {
            "rope_theta": args.rotary_base,
            "rope_type": "default",
        },
        "rope_interleave": True,
        # MLA
        "q_lora_rank": args.q_lora_rank,
        "kv_lora_rank": args.kv_lora_rank,
        "qk_nope_head_dim": args.qk_head_dim,
        "qk_rope_head_dim": args.qk_pos_emb_head_dim,
        "qk_head_dim": args.qk_head_dim + args.qk_pos_emb_head_dim,
        "v_head_dim": args.v_head_dim,
        "head_dim": args.qk_pos_emb_head_dim,
        # MoE
        "moe_intermediate_size": args.moe_ffn_hidden_size,
        "n_routed_experts": args.num_experts,
        "n_shared_experts": args.moe_shared_expert_intermediate_size // args.moe_ffn_hidden_size,
        "num_experts_per_tok": args.moe_router_topk,
        "routed_scaling_factor": args.moe_router_topk_scaling_factor,
        "n_group": args.moe_router_num_groups,
        "topk_group": args.moe_router_group_topk,
        "first_k_dense_replace": first_k_dense_replace,
        "moe_layer_freq": 1,
        "scoring_func": args.moe_router_score_function,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        # DSA
        "index_n_heads": getattr(args, "dsa_indexer_n_heads", 32),
        "index_head_dim": getattr(args, "dsa_indexer_head_dim", 128),
        "index_topk": getattr(args, "dsa_indexer_topk", 2048),
        "indexer_rope_interleave": True,
        "indexer_types": getattr(args, "indexer_types", None),
        # MTP
        "num_nextn_predict_layers": mtp_num_layers,
        # Misc
        "ep_size": 1,
        "pretraining_tp": 1,
        "use_cache": True,
        "transformers_version": "5.0.2.dev0",
    }

    os.makedirs(args.save, exist_ok=True)
    config_path = os.path.join(args.save, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved HF config to {config_path}")

    return config_dict
