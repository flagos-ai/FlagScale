import os
import sys
import json
import types
import importlib

import torch

from utils import print_memory_usage


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

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

    # Search in directory above this
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
        from megatron.training.global_vars import set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.legacy import fused_kernels
        from megatron.core.tensor_parallel.random import (
                get_cuda_rng_tracker, _DATA_PARALLEL_RNG_TRACKER_NAME,
                _EXPERT_PARALLEL_RNG_TRACKER_NAME, _MODEL_PARALLEL_RNG_TRACKER_NAME
            )
        from tools.checkpoint.utils import _ConverterFakeProcessGroup
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    try:
        ckpt_plugin = importlib.import_module(args.model_type + ".ckpt")
        model_plugin = importlib.import_module(args.model_type + ".model")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please check model_type or model.py")

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
        '--transformer-impl', 'transformer_engine',
        '--load', args.load_dir,
        '--exit-on-missing-checkpoint',
        '--use-mp-args-from-checkpoint-args',
        '--no-one-logger',
    ]

    margs = parse_args()
    margs, checkpoint_args = load_args_from_checkpoint(margs)

    def _set_arg(arg_name):
        ckpt_value = getattr(checkpoint_args, arg_name, None)
        setattr(margs, arg_name, ckpt_value)

    _set_arg("decoder_first_pipeline_num_layers")

    # for mla
    _set_arg("q_lora_rank")
    _set_arg("kv_lora_rank")
    _set_arg("q_head_dim")
    _set_arg("qk_pos_emb_head_dim")
    _set_arg("v_head_dim")
    _set_arg("multi_latent_attention")
    _set_arg("apply_rope_fusion")
    _set_arg("qk_layernorm")
    # for moe
    _set_arg("moe_grouped_gemm")
    _set_arg("moe_router_enable_expert_bias")
    _set_arg("moe_router_score_function")
    # for mtp
    _set_arg("mtp_num_layers")

    # for hetero
    _set_arg("enable_hetero")
    _set_arg("hetero_process_meshes")
    _set_arg("hetero_pipeline_layer_split")

    # for hetero
    if margs.hetero_process_meshes is not None:
        margs.pipeline_model_parallel_size = sum(row[-1] for row in margs.hetero_process_meshes)
    margs.data_parallel_size = 1
    margs.micro_batch_size = 1

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size * margs.expert_model_parallel_size

    # Explicitly copy data types from checkpoint.
    margs.fp16 = checkpoint_args.fp16
    margs.bf16 = checkpoint_args.bf16

    # Expert parallelism requires sequence parallelism
    if margs.expert_model_parallel_size > 1:
        margs.sequence_parallel = True

    # set env for moe
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        margs.norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        margs.norm_has_bias = True

    print("*"*20 + "validate loader arguments" + "*"*20)
    margs = validate_args(margs)

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

    # Determine how to make our models.
    margs.model_type = model_plugin.model_type

    """
    use megatron args build object and init env
    """

    # build global variable (eg: tokenizer)
    set_global_variables(margs, build_tokenizer=False)

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    # fake init torch distributed
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
    )

    # fake initializing mpu process groups
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size or 1
    mpu.set_tensor_model_parallel_world_size(tp_size)
    mpu.set_pipeline_model_parallel_world_size(pp_size)
    mpu.set_expert_model_parallel_world_size(ep_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    # For backward compatibility during local parallel states refactoring
    fake_tp_group = _ConverterFakeProcessGroup(size=tp_size)
    fake_ep_group = _ConverterFakeProcessGroup(size=ep_size)
    mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
    mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group
    # For backward compatibility during local parallel states refactoring
    fake_pp_group = _ConverterFakeProcessGroup(size=margs.pipeline_model_parallel_size)
    fake_cp_group = _ConverterFakeProcessGroup(size=margs.context_parallel_size)
    fake_dp_group = _ConverterFakeProcessGroup(size=margs.data_parallel_size)
    fake_etp_group = _ConverterFakeProcessGroup(size=margs.expert_tensor_parallel_size)
    edp_parallel_size = margs.tensor_model_parallel_size * margs.context_parallel_size // (margs.expert_tensor_parallel_size * margs.expert_model_parallel_size)
    fake_edp_group = _ConverterFakeProcessGroup(size=edp_parallel_size)
    fake_etp_ep_group = _ConverterFakeProcessGroup(size=margs.expert_tensor_parallel_size*margs.expert_model_parallel_size)
    fake_tcp_group = _ConverterFakeProcessGroup(size=margs.tensor_model_parallel_size*margs.context_parallel_size)
    mpu._PIPELINE_MODEL_PARALLEL_GROUP = fake_pp_group
    mpu._CONTEXT_PARALLEL_GROUP = fake_cp_group
    mpu._DATA_PARALLEL_GROUP = fake_dp_group
    mpu._EXPERT_TENSOR_PARALLEL_GROUP = fake_etp_group
    mpu._EXPERT_DATA_PARALLEL_GROUP = fake_edp_group
    mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = fake_etp_ep_group
    mpu._TENSOR_AND_CONTEXT_PARALLEL_GROUP = fake_tcp_group
    mpu._EXPERT_TENSOR_PARALLEL_GROUP = fake_tp_group
    mpu._DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
    mpu._LAST_RANK_WHEN_USING_PIPELINE = pp_size - 1

    # fused kernel
    fused_kernels.load(margs)

    # random
    CUDA_RNG_STATE_TRACKER = get_cuda_rng_tracker()
    torch.cuda.manual_seed(42)
    CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, 43)
    CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 44)
    CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, 45)

    # metadata
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
    md.norm_has_bias = margs.norm_has_bias
    md.qk_layernorm = margs.qk_layernorm
    md.swiglu = margs.swiglu
    md.previous_num_experts = margs.num_experts
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.previous_decoder_first_pipeline_num_layers = margs.decoder_first_pipeline_num_layers
    md.true_vocab_size = args.true_vocab_size # true (non-padded) vocab size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args

    consumed_train_samples = None
    consumed_valid_samples = None
    def get_models(count, dtype):
        # for one pp stage
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        tp_size = margs.tensor_model_parallel_size
        pp_size = margs.pipeline_model_parallel_size
        vp_size = margs.virtual_pipeline_model_parallel_size or 1

        models = [[] for _ in range(vp_size)]
        for rank_id in range(count):
            tp_rank = rank_id % tp_size
            ep_rank = rank_id // tp_size
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_expert_model_parallel_rank(ep_rank)
            if pp_size > 1 and vp_size > 1:
                model_ = []
                for vp_rank in range(vp_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
                    # Set pre_process and post_process only after virtual rank is set.
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_plugin.get_mg_model(dtype, pre_process, post_process)
                    model_.append(this_model)
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                model_ = [model_plugin.get_mg_model(dtype, pre_process, post_process)]

            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            margs.exit_on_missing_checkpoint = True
            load_checkpoint(model_, None, None)

            if consumed_train_samples is not None:
                assert(margs.consumed_train_samples == consumed_train_samples)
            else:
                consumed_train_samples = margs.consumed_train_samples

            if consumed_valid_samples is not None:
                assert(margs.consumed_valid_samples == consumed_valid_samples)
            else:
                consumed_valid_samples = margs.consumed_valid_samples

            for vp_rank in range(vp_size):
                models[vp_rank].append(model_[vp_rank])

            # Print memory usage.
            print_memory_usage("loader", rank_id, count)

        return models

    # Get first pipe stage and load ckpt
    mpu.set_pipeline_model_parallel_rank(0)
    fake_pp_group = _ConverterFakeProcessGroup(rank=0, size=pp_size)
    mpu._PIPELINE_MODEL_PARALLEL_GROUP = fake_pp_group
    all_models = [get_models(tp_size * ep_size, margs.params_dtype)]
    models = all_models[0][0] # pp0vpp0

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    """
    start sending ckpt
    """

    # Send embeddings
    message = dict()
    ckpt_plugin.get_embedding_ckpt(message, models, margs)
    queue_put("embeddings", message)

    # Send transformer layer
    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            fake_pp_group = _ConverterFakeProcessGroup(rank=pp_rank, size=pp_size)
            mpu._PIPELINE_MODEL_PARALLEL_GROUP = fake_pp_group

            if pp_rank > 0 and vp_rank == 0:
                all_models.append(get_models(tp_size * ep_size, margs.params_dtype))

            models = all_models[pp_rank][vp_rank]
            for layer_id in range(len(models[0].decoder.layers)):
                message = dict()
                margs.total_layer_num = total_layer_num
                ckpt_plugin.get_attn_ckpt(message, models, layer_id, margs)
                ckpt_plugin.get_mlp_ckpt(message, models, layer_id, margs)

                queue_put(f"transformer layer {total_layer_num}", message)
                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    message = dict()
    ckpt_plugin.get_final_norm_ckpt(message, models, margs)
    queue_put("final norm", message)

    if md.output_layer:
        message = dict()
        ckpt_plugin.get_output_layer_ckpt(message, models, margs)
        queue_put("output layer", message)

    message = dict()
    if margs.mtp_num_layers:
        for mtp_layer_id in range(margs.mtp_num_layers):
            message = dict()
            ckpt_plugin.get_mtp_ckpt(message, models, mtp_layer_id, margs)
            queue_put(f"mtp module {mtp_layer_id}", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
