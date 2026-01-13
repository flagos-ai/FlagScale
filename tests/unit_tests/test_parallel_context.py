import torch

import megatron.training.global_vars as mcore_global_vars
from megatron.plugin.hetero.parallel_context import ParallelContext
from megatron.training.arguments import parse_args
from megatron.training.arguments_fs import FSTrainArguments
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from tests.unit_tests.test_utilities import Utils as MegatronUtils


def init_parallel_context() -> ParallelContext:
    args = parse_args(ignore_unknown_args=True)
    args.tensor_model_parallel_size = 2
    args.pipeline_model_parallel_size = 3
    args.virtual_pipeline_model_parallel_size = None
    args.disable_bias_linear = True
    args.use_flash_attn = True
    args.sequence_parallel = True
    args.use_distributed_optimizer = True
    args.use_mcore_models = True
    args.transformer_impl = "transformer_engine"
    args.enable_hetero = True
    args.hetero_pipeline_layer_split = [6, 2, 4]
    args.hetero_process_meshes = [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    args.hetero_device_types = ["A800", "A800", "A800"]
    args.hetero_current_device_type = "A800"
    args.micro_batch_size = 1
    args.global_batch_size = 32

    #
    args.recompute_granularity = "full"
    args.recompute_method = "uniform"
    args.recompute_num_layers = 1

    # recompute per stage micro batch
    args.recompute_granularity_per_stage_micro_batch = [
        "1, 30, 0, 2, 0",
        "1, 30, 1, 2, 1",
        "1, 30, 1, 2, 1",
    ]
    args.recompute_method_per_stage_micro_batch = [
        "1, 30, 0, 2, 0",
        "1, 30, 0, 2, 0",
        "1, 30, 1, 2, 1",
    ]
    args.recompute_num_layers_per_stage_micro_batch = [
        "1, 30, 2, 2, 2",
        "1, 30, 1, 2, 1",
        "1, 30, 2, 2, 2",
    ]

    # extra transformer config
    args.params_dtype = torch.bfloat16
    args.num_attention_heads = 32
    args.hidden_size = 1024
    args.num_layers = 12

    # extra optimizer config
    args.main_grads_dtype = torch.float32
    args.main_params_dtype = torch.float32
    args.exp_avg_dtype = torch.float32
    args.exp_avg_sq_dtype = torch.float32

    train_args = FSTrainArguments(args)
    train_args.pre_validate_args()
    train_args.post_validate_args()

    # for building datasets
    mcore_global_vars._GLOBAL_TOKENIZER = _NullTokenizer(vocab_size=64)
    para_ctx = ParallelContext(args)
    return para_ctx


def test_parallel_config():
    MegatronUtils.initialize_distributed()

    para_ctx = init_parallel_context()

    assert para_ctx is not None
    assert para_ctx.get_ddp_config() is not None
    assert para_ctx.get_transformer_config() is not None
    assert para_ctx.get_dataset_config() is not None
