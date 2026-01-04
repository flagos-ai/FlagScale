import os
import sys
import types
import importlib

from utils import print_memory_usage


def add_arguments(parser):
    group = parser.add_argument_group(title='Transformers loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')


def _load_checkpoint(queue, args):

    """
    prepare import module
    """

    try:
        import transformers
        major, minor, _ = map(int, transformers.__version__.split('.'))
        assert major >= 4 and minor >= 36
    except:
        raise ImportError("transformers version >= 4.36.0 ")

    # Search in directory above this.
    root_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir))
    sys.path.insert(0, root_path)
    sys.path.insert(0, os.path.join(root_path, "flagscale/train"))

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    try:
        args_plugin = importlib.import_module(args.model_type + ".args")
        ckpt_plugin = importlib.import_module(args.model_type + ".ckpt")
        model_plugin = importlib.import_module(args.model_type + ".model")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please check model_type, args.py, ckpt.py or model.py")

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    """
    prepare megatron arguments (margs)
    """

    # We want all arguments to come from us.
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--mock-data', # To pass the "blend data checks" in arguments.py
        '--use-mcore-models',
        '--transformer-impl', 'transformer_engine',
        '--load', args.load_dir,
        '--exit-on-missing-checkpoint',
        '--use-mp-args-from-checkpoint-args',
        '--no-one-logger',
    ]

    margs = parse_args()
    args_plugin.load_args_hf2mg(margs)

    print("*"*20 + "validate loader arguments" + "*"*20)
    margs = validate_args(margs)
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size * margs.expert_model_parallel_size

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)
    check_for_arg('disable_bias_linear', not getattr(margs, "add_bias_linear", False))
    check_for_arg('add_qkv_bias', getattr(margs, "add_bias_linear_qkv", False))
    check_for_arg('qk_layernorm', getattr(margs, "qk_layernorm", False))

    """
    use megatron args build object and init env
    """

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.add_bias_linear = margs.add_bias_linear
    md.add_qkv_bias = margs.add_qkv_bias
    md.qk_layernorm = margs.qk_layernorm
    md.norm_has_bias = margs.norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_num_experts = margs.num_experts
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_decoder_first_pipeline_num_layers = margs.decoder_first_pipeline_num_layers
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.previous_virtual_pipeline_parallel_size = margs.virtual_pipeline_model_parallel_size
    md.true_vocab_size = args.true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = margs.consumed_train_samples
    md.consumed_valid_samples = margs.consumed_valid_samples
    queue.put(md)

    # get transformers model
    hf_model = model_plugin.get_hf_model(margs.params_dtype, margs.load)
    print_memory_usage("loader", 0, 0)

    """
    start sending ckpt
    """

    # Send embeddings.
    message = dict()
    message["word embeddings"]= hf_model.model.embed_tokens.weight
    queue_put("embeddings", message)

    # Send transformer layers
    for layer_id in range(margs.num_layers):
        message = dict()

        ckpt_plugin.get_hf_attn_ckpt(message, hf_model, layer_id, margs)
        ckpt_plugin.get_hf_mlp_ckpt(message, hf_model, layer_id, margs)

        queue_put(f"transformer layer {layer_id}", message)

    # Send final norm from tp_rank 0.
    message = {"weight": hf_model.model.norm.weight.data}
    if md.norm_has_bias:
        message["bias"] = hf_model.model.norm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        message = {"weight": hf_model.lm_head.weight.data}
        queue_put("output layer", message)

    message = dict()
    if margs.mtp_num_layers:
        for mtp_layer_id in range(margs.mtp_num_layers):
            message = dict()
            ckpt_plugin.get_hf_mtp_ckpt(message, hf_model, mtp_layer_id, margs)
            queue_put(f"mtp module {mtp_layer_id}", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
