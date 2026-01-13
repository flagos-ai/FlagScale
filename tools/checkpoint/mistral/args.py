import json
import os


def load_args_hf2mg(args):
    # Read mistral args.
    mistral_args_path = os.path.join(args.load, "config.json")
    with open(mistral_args_path) as f:
        mistral_args = json.load(f)

    # Update Megatron args.
    args.attention_dropout = mistral_args["attention_dropout"]
    args.hidden_dropout = mistral_args["attention_dropout"]
    args.hidden_size = mistral_args["hidden_size"]
    args.swiglu = mistral_args["hidden_act"] == "silu"
    args.init_method_std = mistral_args["initializer_range"]
    args.ffn_hidden_size = mistral_args["intermediate_size"]
    args.max_position_embeddings = mistral_args["max_position_embeddings"]
    args.model_type = mistral_args["model_type"]
    args.num_attention_heads = mistral_args["num_attention_heads"]
    args.num_layers = mistral_args["num_hidden_layers"]
    args.num_query_groups = mistral_args["num_key_value_heads"]
    args.norm_epsilon = mistral_args["rms_norm_eps"]
    args.rotary_seq_len_interpolation_factor = (
        None if mistral_args["rope_scaling"] == "null" else mistral_args["rope_scaling"]
    )
    args.rotary_base = mistral_args["rope_theta"]
    args.untie_embeddings_and_output_weights = not mistral_args["tie_word_embeddings"]
    args.bf16 = mistral_args["torch_dtype"] == "bfloat16"
    args.fp16 = mistral_args["torch_dtype"] == "float16"
    args.vocab_size = mistral_args["vocab_size"]
    args.padded_vocab_size = mistral_args["vocab_size"]

    args.seq_length = 2048
    args.global_batch_size = 1024
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.group_query_attention = True
    args.normalization = "RMSNorm"
    args.use_rotary_position_embeddings = True
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.make_vocab_size_divisible_by = 64
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.norm_has_bias = False


def save_args_mg2hf(args):
    from transformers import MistralConfig

    config = MistralConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.encoder_num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.norm_epsilon,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        rope_theta=args.rotary_base,
        attention_dropout=args.attention_dropout,
        torch_dtype=args.params_dtype,
        attention_bias=args.add_qkv_bias,
    )
    config.save_pretrained(args.save)

    return config
