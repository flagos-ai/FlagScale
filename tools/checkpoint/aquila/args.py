import json
import os


def load_args_hf2mg(args):
    # Read llama args.
    llama_args_path = os.path.join(args.load, "config.json")
    with open(llama_args_path) as f:
        llama_args = json.load(f)

    # Update Megatron args.
    args.attention_dropout = llama_args["attention_dropout"]
    args.hidden_dropout = llama_args["attention_dropout"]
    args.hidden_size = llama_args["hidden_size"]
    args.swiglu = llama_args["hidden_act"] == "silu"
    args.init_method_std = llama_args["initializer_range"]
    args.ffn_hidden_size = llama_args["intermediate_size"]
    args.max_position_embeddings = llama_args["max_position_embeddings"]
    args.model_type = llama_args["model_type"]
    args.num_attention_heads = llama_args["num_attention_heads"]
    args.num_layers = llama_args["num_hidden_layers"]
    args.num_query_groups = llama_args["num_key_value_heads"]
    args.norm_epsilon = llama_args["rms_norm_eps"]
    args.rotary_seq_len_interpolation_factor = (
        None if llama_args["rope_scaling"] == "null" else llama_args["rope_scaling"]
    )
    args.rotary_base = llama_args["rope_theta"]
    args.untie_embeddings_and_output_weights = not llama_args["tie_word_embeddings"]
    args.bf16 = llama_args["torch_dtype"] == "bfloat16"
    args.fp16 = llama_args["torch_dtype"] == "float16"
    args.vocab_size = llama_args["vocab_size"]
    args.padded_vocab_size = llama_args["vocab_size"]

    args.seq_length = 4096
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
    from .llama_model.configuration_llama import LlamaConfig

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.encoder_num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups,
        hidden_act="silu" if args.swiglu else False,
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=args.init_method_std,
        rms_norm_eps=args.norm_epsilon,
        use_cache=True,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        rope_theta=args.rotary_base,
        rope_scaling=args.rotary_seq_len_interpolation_factor,
        attention_bias=args.add_qkv_bias,
        attention_dropout=args.attention_dropout,
        torch_dtype=args.params_dtype,
        bias_dropout_fusion=args.bias_dropout_fusion,
        end_weight_decay=args.end_weight_decay,
        global_batch_size=args.global_batch_size,
        hidden_dropout=args.hidden_dropout,
        lr=args.lr,
        lr_decay_style=args.lr_decay_style,
        make_vocab_size_divisible_by=args.make_vocab_size_divisible_by,
        masked_softmax_fusion=args.masked_softmax_fusion,
        min_lr=args.min_lr,
        norm_init_weight=args.norm_init_weight,
        perform_initialization=args.perform_initialization,
        reset_attention_mask=args.reset_attention_mask,
        reset_position_ids=args.reset_position_ids,
        rotary_base=args.rotary_base,
        seed=args.seed,
        split=args.split,
        start_weight_decay=args.start_weight_decay,
        use_flash_attn=args.use_flash_attn,
        weight_decay_incr_style=args.weight_decay_incr_style,
    )
    config.save_pretrained(args.save)

    return config
