import json
import os


def load_args_hf2mg(args):
    # Read transformers args.
    hf_args_path = os.path.join(args.load, "config.json")
    with open(hf_args_path) as f:
        hf_args = json.load(f)

    # Update Megatron args.
    args.attention_dropout = hf_args["attention_dropout"]
    args.hidden_dropout = hf_args["attention_dropout"]
    args.hidden_size = hf_args["hidden_size"]
    args.init_method_std = hf_args["initializer_range"]
    args.ffn_hidden_size = hf_args["intermediate_size"]
    args.max_position_embeddings = hf_args["max_position_embeddings"]
    args.model_type = hf_args["model_type"]
    args.kv_channels = hf_args["head_dim"]
    args.num_attention_heads = hf_args["num_attention_heads"]
    args.num_layers = hf_args["num_hidden_layers"]
    args.num_query_groups = hf_args["num_key_value_heads"]
    args.norm_epsilon = hf_args["rms_norm_eps"]
    args.rotary_base = hf_args["rope_theta"]
    args.untie_embeddings_and_output_weights = not hf_args["tie_word_embeddings"]
    args.bf16 = hf_args["torch_dtype"] == "bfloat16"
    args.fp16 = hf_args["torch_dtype"] == "float16"
    args.vocab_size = hf_args["vocab_size"]
    args.padded_vocab_size = hf_args["vocab_size"]

    args.seq_length = 2048
    args.global_batch_size = 1024
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.swiglu = True  # hf_args["hidden_act"] -> "silu"
    args.group_query_attention = True
    args.qk_layernorm = True
    args.normalization = "RMSNorm"
    args.tokenizer_type = "Qwen2TokenizerFS"
    args.use_rotary_position_embeddings = True
    args.add_bias_linear = False
    args.add_qkv_bias = False  # hf_args["attention_bias"] -> false
    args.make_vocab_size_divisible_by = 64
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.norm_has_bias = False

    return args


def save_args_mg2hf(args):
    from .modeling_hf.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.encoder_num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups,
        head_dim=args.kv_channels,
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=args.init_method_std,
        rms_norm_eps=args.norm_epsilon,
        use_cache=True,
        tie_word_embeddings=(not args.untie_embeddings_and_output_weights),
        rope_theta=args.rotary_base,
        attention_dropout=args.attention_dropout,
        torch_dtype=args.params_dtype,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    auto_map = dict()
    auto_map["AutoConfig"] = "configuration_qwen3.Qwen3Config"
    auto_map["AutoModelForCausalLM"] = "modeling_qwen3.Qwen3ForCausalLM"
    config.auto_map = auto_map
    config.save_pretrained(args.save)

    return config
