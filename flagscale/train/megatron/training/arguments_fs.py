import ast
import itertools
import types
import warnings

from datetime import timedelta

import torch

try:
    import flagcx
except:
    warnings.warn(
        "flagcx is not installed, you can't use flagcx backend for communication.", ImportWarning
    )
from megatron.plugin.hetero.parallel_context import RankMapper

class FSTrainArguments:
    """Extend the Megatron arguments with FlagScale specific arguments."""

    def __init__(self, args, rank_mapper=None):
        self.args = args
        self._rank_mapper = rank_mapper

    def __getattr__(self, name):
        if name == "rank_mapper":
            return self._rank_mapper
        return getattr(self.args, name)

    def _initialize_distributed(self):
        """Initialize torch.distributed and core model parallel."""
        args = self.args

        device_count = torch.cuda.device_count()
        if torch.distributed.is_initialized():

            if args.rank == 0:
                print(
                    "torch distributed is already initialized, " "skipping initialization ...",
                    flush=True,
                )
            args.rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()

        else:

            if args.rank == 0:
                print("> initializing torch distributed ...", flush=True)
            # Manually set the device ids.
            if device_count > 0:
                torch.cuda.set_device(args.local_rank)
                device_id = torch.device(f"cuda:{args.local_rank}")
            else:
                device_id = None

            # Call the init process
            init_process_group_kwargs = {
                "backend": args.distributed_backend,
                "world_size": args.world_size,
                "rank": args.rank,
                "timeout": timedelta(minutes=args.distributed_timeout_minutes),
            }
            if args.distributed_backend == "flagcx":
                init_process_group_kwargs["backend"] = "cpu:gloo,cuda:flagcx"
            # for communication based cpu
            if args.enable_hetero and args.hetero_use_cpu_communication:
                # if not all(device_type == args.hetero_device_types[0] for device_type in args.hetero_device_types):
                #     init_process_group_kwargs['backend'] = 'cpu:gloo'
                # Force the group of backend gloo only support cpu
                init_process_group_kwargs["backend"] = "cpu:gloo"
            torch.distributed.init_process_group(**init_process_group_kwargs)

    def _build_rank_mapper(self):
        self._initialize_distributed()
        self._rank_mapper = RankMapper(self.args)
        return self._rank_mapper

    def pre_validate_args(self):
        """Pre-validate the arguments before Megatron function `validate_args`."""
        if self._rank_mapper is None:
            self._build_rank_mapper()

        if self.args.hetero_process_meshes is not None:
            assert (
                len(self.args.hetero_process_meshes) % 5 == 0
            ), f"length of hetero_process_meshes {self.args.hetero_process_meshes} should be divisible by 5, the format should be tp0, cp0, dp0, pp0, tp1, cp1, dp1, pp1, ..."
            hetero_process_meshes_tp = self.args.hetero_process_meshes[0::5]
            hetero_process_meshes_cp = self.args.hetero_process_meshes[1::5]
            hetero_process_meshes_ep = self.args.hetero_process_meshes[2::5]
            hetero_process_meshes_dp = self.args.hetero_process_meshes[3::5]
            hetero_process_meshes_pp = self.args.hetero_process_meshes[4::5]

            # Check if tensor parallel sizes are inconsistent across meshes
            # NOTE: If TP degrees differ, sequence parallelism must be enabled
            if len(set(hetero_process_meshes_tp)) > 1:
                assert (
                    self.args.sequence_parallel
                ), f"Sequence parallelism must be enabled (`sequence_parallel=True`) when tensor parallelism degrees differ across heterogeneous meshes. Found TP degrees: {hetero_process_meshes_tp}"
            # Expert tensor parallel size
            if self.expert_tensor_parallel_size_per_process_mesh is not None:
                assert len(self.expert_tensor_parallel_size_per_process_mesh) == len(
                    hetero_process_meshes_tp
                ), f"length of expert_tensor_parallel_size_per_process_mesh {len(self.expert_tensor_parallel_size_per_process_mesh)} should be equal to length of hetero_process_meshes_tp {len(hetero_process_meshes_tp)}"
            # Data parallel size
            # NOTE: Use the first data parallel size as the global data parallel size to loader data
            self.args.data_parallel_size = hetero_process_meshes_dp[0]
            assert all(
                self.args.data_parallel_size * self.args.micro_batch_size % hetero_dp == 0
                for hetero_dp in hetero_process_meshes_dp
            ), f"data_parallel_size * micro_batch_size {self.args.data_parallel_size * self.args.micro_batch_size} should be divisible by all hetero_process_meshes_dp {hetero_process_meshes_dp}!"

            # NOTE: Only support cp and ep size to be the same
            assert all(
                hetero_cp == hetero_process_meshes_cp[0] for hetero_cp in hetero_process_meshes_cp
            ), f"all hetero_process_meshes_cp {hetero_process_meshes_cp} should be the same!"

            # Note: Ep size should all be 1 or all be not 1
            assert all(1 == hetero_ep for hetero_ep in hetero_process_meshes_ep) or any(
                1 != hetero_ep for hetero_ep in hetero_process_meshes_ep
            ), f"all hetero_process_meshes_ep {hetero_process_meshes_ep} should be the 1 or none of hetero_process_meshes_ep is not 1!"

            # Pipeline model parallel size
            assert self.args.pipeline_model_parallel_size == sum(
                hetero_process_meshes_pp
            ), f"origin pipeline_model_parallel_size {self.args.pipeline_model_parallel_size} should match sum of hetero_process_meshes_pp {hetero_process_meshes_pp}!"
            assert (
                self.args.standalone_embedding_stage == False
            ), "standalone not supported with process_meshes set!"
            self.args.transformer_pipeline_model_parallel_size = self.args.pipeline_model_parallel_size

            # if untie_embeddings_and_output_weights is False, the first and last stage should have the same tp degree
            if self.args.untie_embeddings_and_output_weights == False or self.args.mtp_num_layers:
                assert (
                    hetero_process_meshes_tp[0] == hetero_process_meshes_tp[-1]
                ), f"if untie_embeddings_and_output_weights is False or mtp_num_layers is not 0, the first and last stage should have the same tp degree!"

                if (
                    hetero_process_meshes_dp[0] != hetero_process_meshes_dp[-1]
                    and self.args.use_distributed_optimizer
                ):
                    assert (
                        hetero_process_meshes_dp[0] % hetero_process_meshes_dp[-1] == 0
                        or hetero_process_meshes_dp[-1] % hetero_process_meshes_dp[0] == 0
                    ), (
                        f"if untie_embeddings_and_output_weights is False and  hetero_process_meshes_dp[0] and hetero_process_meshes_dp[-1] are different, "
                        "the hetero_process_meshes_dp[0] should be divisible by hetero_process_meshes_dp[-1] or hetero_process_meshes_dp[-1] should be divisible by hetero_process_meshes_dp[0] currently!"
                    )
                    assert self.args.use_partial_reduce_for_shared_embedding == True, (
                        f"if untie_embeddings_and_output_weights is False and  hetero_process_meshes_dp[0] and hetero_process_meshes_dp[-1] are different, "
                        "the use_partial_reduce_for_shared_embedding should be True currently!"
                    )

            # Virtual parallel size.
            if self.args.enable_hetero:
                assert (
                    self.args.num_layers_per_virtual_pipeline_stage == None
                ), "virtual pipeline not support now!"

            # Model layer splits
            if self.args.hetero_pipeline_layer_split is None:
                num_layers_per_pipeline_stage = (
                    self.args.num_layers // self.args.transformer_pipeline_model_parallel_size
                )
                self.args.hetero_pipeline_layer_split = [
                    num_layers_per_pipeline_stage
                ] * self.args.pipeline_model_parallel_size
            else:
                assert (
                    sum(self.args.hetero_pipeline_layer_split) == self.args.num_layers
                ), f"sum of hetero_pipeline_layer_split {self.args.hetero_pipeline_layer_split} should be equal to num_layers {self.args.num_layers}"
                assert self.args.pipeline_model_parallel_size == len(
                    self.args.hetero_pipeline_layer_split
                ), f"pipeline_model_parallel_size {self.args.pipeline_model_parallel_size} should be equal to the length of hetero_pipeline_layer_split {self.args.hetero_pipeline_layer_split}"
            setattr(
                self.args, "all_pipeline_model_parallel_size", self.args.pipeline_model_parallel_size
            )

            hetero_process_meshes = []
            for i in range(0, len(self.args.hetero_process_meshes), 5):
                hetero_process_meshes.append(self.args.hetero_process_meshes[i : i + 5])
            self.args.hetero_process_meshes = hetero_process_meshes

            # Device types
            assert len(hetero_process_meshes) == len(
                self.args.hetero_device_types
            ), f"length of hetero_process_meshes {len(hetero_process_meshes)} should match length of hetero_device_types {len(self.args.hetero_device_types)}"
            assert (
                self.args.hetero_current_device_type in self.args.hetero_device_types
            ), f"hetero_current_device_type {self.args.hetero_current_device_type} should be in hetero_device_types {self.args.hetero_device_types}"

            current_process_mesh_idx = 0
            accumulated_world_size = 0
            rank = torch.distributed.get_rank()
            logical_rank = self.rank_mapper.to_logical_ranks([rank])[0]
            for tp, cp, ep, dp, pp in self.args.hetero_process_meshes:
                temp_world_size = tp * cp * dp * pp
                if (
                    logical_rank >= accumulated_world_size
                    and logical_rank < accumulated_world_size + temp_world_size
                ):
                    # update some associated args
                    self.args.micro_batch_size = (
                        self.args.data_parallel_size * self.args.micro_batch_size // dp
                    )

                    # update parallel sizes
                    self.args.tensor_model_parallel_size = tp
                    self.args.context_parallel_size = cp
                    self.args.expert_model_parallel_size = ep
                    self.args.data_parallel_size = dp
                    self.args.pipeline_model_parallel_size = pp
                    if self.args.expert_tensor_parallel_size_per_process_mesh is not None:
                        self.args.expert_tensor_parallel_size = (
                            self.args.expert_tensor_parallel_size_per_process_mesh[
                                current_process_mesh_idx
                            ]
                        )

                    # Sequence parallel
                    if self.args.tensor_model_parallel_size == 1:
                        self.args.sequence_parallel = False

                    # TODO: update other args if need

                accumulated_world_size += temp_world_size
                current_process_mesh_idx += 1

            

    def post_validate_args(self):
        """Post-validate the arguments after Megatron function `validate_args`."""
        args = self.args

        if args.hetero_process_meshes is not None:
            # Validate the refined-recompute configuration
            def _parse_recompute_refined_config(recom_config, recom_config_name):
                """Parse refined recompute configuration."""
                if recom_config is None:
                    return None
                assert isinstance(
                    recom_config, list
                ), f"[{recom_config_name}] recompute configuration, is not list."
                recom_config = [ast.literal_eval(item) for item in recom_config]
                parsed_pp_size = 0
                parsed_pp_chunk_config = []
                for pp_chunk_id in range(len(recom_config)):
                    cur_pp_chunk_config = recom_config[pp_chunk_id]
                    for _ in range(cur_pp_chunk_config[0]):
                        parsed_pp_size = parsed_pp_size + 1
                        mc_chunks = len(cur_pp_chunk_config) // 2
                        cur_pp_stage_per_mc = []
                        for mc_chunk in range(mc_chunks):
                            cur_pp_stage_per_mc += itertools.repeat(
                                cur_pp_chunk_config[2 + mc_chunk * 2],
                                cur_pp_chunk_config[1 + mc_chunk * 2],
                            )
                        assert len(cur_pp_stage_per_mc) == args.global_batch_size // (
                            args.micro_batch_size * args.data_parallel_size
                        ), (
                            f"for [{recom_config_name}] refined recompute "
                            f"configuration, the sum [{len(cur_pp_stage_per_mc)}] of n0, n1, ... of sub-list should be equal to nums_micro_batch [{args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)}]."
                        )
                        if "method" in recom_config_name or "granularity" in recom_config_name:
                            assert all(
                                val == 0 or val == 1 for val in cur_pp_stage_per_mc
                            ), f"the config-flag of {recom_config_name} must be 0 or 1"
                        parsed_pp_chunk_config.append(cur_pp_stage_per_mc)
                if args.virtual_pipeline_model_parallel_size != None:
                    assert (
                        parsed_pp_size
                        == args.all_pipeline_model_parallel_size
                        * args.virtual_pipeline_model_parallel_size
                    ), "for refined recompute configuration, the sum of axis 0 should be equal to pipeline-model-parallel-size * args.virtual_pipeline_model_parallel_size."
                else:
                    assert (
                        parsed_pp_size == args.all_pipeline_model_parallel_size
                    ), "for refined recompute configuration, the sum of axis 0 should be equal to pipeline-model-parallel-size."
                return parsed_pp_chunk_config

            if args.recompute_granularity_per_stage_micro_batch != None:
                assert args.recompute_granularity == "full", (
                    "recompute-granularity-per-stage is only"
                    "application to full recompute granularity mode"
                )
                assert args.recompute_method is not None, (
                    "for distributed recompute activations to work you "
                    "need to use a recompute method "
                )

            args.recompute_granularity_per_stage_micro_batch = _parse_recompute_refined_config(
                args.recompute_granularity_per_stage_micro_batch,
                "recompute_granularity_per_stage_micro_batch",
            )
            args.recompute_method_per_stage_micro_batch = _parse_recompute_refined_config(
                args.recompute_method_per_stage_micro_batch, "recompute_method_per_stage_micro_batch"
            )
            args.recompute_num_layers_per_stage_micro_batch = _parse_recompute_refined_config(
                args.recompute_num_layers_per_stage_micro_batch,
                "recompute_num_layers_per_stage_micro_batch",
            )

        # DualPipeV related
        if args.use_dualpipev:
            assert args.pipeline_model_parallel_size > 1, (
                "DualPipeV can only be used for pipeline scheduling in MoE models, "
            "thus requiring both pipeline parallelism and expert parallelism."
            )
            assert args.expert_model_parallel_size > 1, (
                "DualPipeV can only be used for pipeline scheduling in MoE models, "
            "thus requiring both pipeline parallelism and expert parallelism."
            )

            middle_stage_layers = args.num_layers
            num_middle_stages = args.pipeline_model_parallel_size
            if args.decoder_first_pipeline_num_layers is not None:
                middle_stage_layers = middle_stage_layers - args.decoder_first_pipeline_num_layers
                num_middle_stages = num_middle_stages - 1
                assert args.decoder_first_pipeline_num_layers % 2 == 0, (
                    "The first pipeline stage must contain an even number of Transformer layers, "
                    "so that DualPipeV can split it into two model chunks."
                )
            if args.decoder_last_pipeline_num_layers is not None:
                middle_stage_layers = middle_stage_layers - args.decoder_last_pipeline_num_layers
                num_middle_stages = num_middle_stages - 1
                assert args.decoder_last_pipeline_num_layers % 2 == 0, (
                    "The last pipeline stage must contain an even number of Transformer layers, "
                    "so that DualPipeV can split it into two model chunks."
                )
            if num_middle_stages > 0:
                assert middle_stage_layers > 0, "Layers can not be empty"
                assert middle_stage_layers % num_middle_stages == 0, "Layers must be even split"
                num_layers_in_middle_stages = middle_stage_layers // num_middle_stages
                assert num_layers_in_middle_stages % 2 == 0, (
                    "The middle pipeline stage must contain an even number of Transformer layers, "
                    "so that DualPipeV can split it into two model chunks."
                )

            assert args.moe_shared_expert_overlap is False, (
                    " DualPipeV does not support simultaneous use with moe_shared_expert_overlap currently."
            )

            if args.moe_fb_overlap:
                assert args.overlap_grad_reduce is False and args.overlap_param_gather is False, (
                    " DualPipeV configured with moe_fb_overlap is incompatible with either overlap_grad_reduce or overlap_param_gather. "
                    " When moe_fb_overlap is enabled, DualPipeV activates the DW-split mechanism provided by Transformer Engine, "
                    " which causes all param.grad attributes to be None during the backward-for-inputs phase. "
                    " This absence of gradient tensors violates the assumptions of both overlap_grad_reduce and overlap_param_gather, precipitating an assertion failure within DDP."
                )
                assert not args.moe_use_legacy_grouped_gemm, \
                    'delay_wgrad_compute is not supported with legacy groupedgemm implementation'
                assert args.transformer_impl == 'transformer_engine', \
                    'delay_wgrad_compute is only supported with transformer_engine implementation'

            assert args.untie_embeddings_and_output_weights is True, (
                " DualPipeV is not supported with shared embedding and lm head"
            )
            assert args.mtp_num_layers is None, (
                "DualPipeV is not supported with multi-token-predictor currently"
            )

        if args.peft_type is not None:
            assert args.transformer_impl == 'transformer_engine', \
                'PEFT is only supported with transformer_engine implementation'
            if args.num_experts is not None and args.moe_shared_expert_intermediate_size is not None:
                assert not args.moe_shared_expert_overlap, \
                    'PEFT is incompatible with moe_shared_expert_overlap'
            assert args.num_experts is None, "PEFT is not tested with MoE currently"
            assert args.recompute_method is None and args.recompute_granularity is None and args.recompute_num_layers is None, "PEFT will raise comfilcts with recompute currently"
            assert args.ckpt_format == 'torch', "PEFT is only tested with torch format checkpoint"

        # DualPipeV related
        if args.use_dualpipev:
            assert args.pipeline_model_parallel_size > 1, (
                "DualPipeV can only be used for pipeline scheduling in MoE models, "
            "thus requiring both pipeline parallelism and expert parallelism."
            )
            assert args.expert_model_parallel_size > 1, (
                "DualPipeV can only be used for pipeline scheduling in MoE models, "
            "thus requiring both pipeline parallelism and expert parallelism."
            )

            middle_stage_layers = args.num_layers
            num_middle_stages = args.pipeline_model_parallel_size
            if args.decoder_first_pipeline_num_layers is not None:
                middle_stage_layers = middle_stage_layers - args.decoder_first_pipeline_num_layers
                num_middle_stages = num_middle_stages - 1
                assert args.decoder_first_pipeline_num_layers % 2 == 0, (
                    "The first pipeline stage must contain an even number of Transformer layers, "
                    "so that DualPipeV can split it into two model chunks."
                )
            if args.decoder_last_pipeline_num_layers is not None:
                middle_stage_layers = middle_stage_layers - args.decoder_last_pipeline_num_layers
                num_middle_stages = num_middle_stages - 1
                assert args.decoder_last_pipeline_num_layers % 2 == 0, (
                    "The last pipeline stage must contain an even number of Transformer layers, "
                    "so that DualPipeV can split it into two model chunks."
                )
            if num_middle_stages > 0:
                assert middle_stage_layers > 0, "Layers can not be empty"
                assert middle_stage_layers % num_middle_stages == 0, "Layers must be even split"
                num_layers_in_middle_stages = middle_stage_layers // num_middle_stages
                assert num_layers_in_middle_stages % 2 == 0, (
                    "The middle pipeline stage must contain an even number of Transformer layers, "
                    "so that DualPipeV can split it into two model chunks."
                )

            assert args.moe_shared_expert_overlap is False, (
                    " DualPipeV does not support simultaneous use with moe_shared_expert_overlap currently."
            )

            if args.moe_fb_overlap:
                assert args.overlap_grad_reduce is False and args.overlap_param_gather is False, (
                    " DualPipeV configured with moe_fb_overlap is incompatible with either overlap_grad_reduce or overlap_param_gather. "
                    " When moe_fb_overlap is enabled, DualPipeV activates the DW-split mechanism provided by Transformer Engine, "
                    " which causes all param.grad attributes to be None during the backward-for-inputs phase. "
                    " This absence of gradient tensors violates the assumptions of both overlap_grad_reduce and overlap_param_gather, precipitating an assertion failure within DDP."
                )
                assert not args.moe_use_legacy_grouped_gemm, \
                    'delay_wgrad_compute is not supported with legacy groupedgemm implementation'
                assert args.transformer_impl == 'transformer_engine', \
                    'delay_wgrad_compute is only supported with transformer_engine implementation'

            assert args.untie_embeddings_and_output_weights is True, (
                " DualPipeV is not supported with shared embedding and lm head"
            )
            assert args.mtp_num_layers is None, (
                "DualPipeV is not supported with multi-token-predictor currently"
            )

        if args.peft_type is not None:
                assert args.transformer_impl == 'transformer_engine', \
                    'PEFT is only supported with transformer_engine implementation'
                assert args.num_experts is None, "PEFT is not tested with MoE currently"
                assert args.recompute_method is None and args.recompute_granularity is None and args.recompute_num_layers is None, "PEFT will raise comfilcts with recompute currently"
                assert args.ckpt_format == 'torch', "PEFT is only tested with torch format checkpoint"


def _add_hetero_args(parser):
    """Add heterogeneous training related arguments (FlagScale specific)."""
    group = parser.add_argument_group(title="flagscale heterogeneous training")

    group.add_argument(
        "--enable-hetero",
        action="store_true",
        help="the mode of heterogeneous training",
    )
    group.add_argument(
        "--hetero-device-types",
        nargs="*",
        type=str,
        default=None,
        help="the list of device types: device_type_0 device_type_1 ...",
    )
    group.add_argument(
        "--hetero-current-device-type",
        type=str,
        default=None,
        help="the current device type",
    )
    group.add_argument(
        "--hetero-pipeline-layer-split",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Incompatible with --num-layers-per-virtual-pipeline-stage for now. "
            "hetero-pipeline-layer-split must be in the form: layers_0 layers_1 ... layers_n. "
            "The number of the list should be equal to pipeline-model-parallel-size."
        ),
    )
    group.add_argument(
        "--hetero-process-meshes",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Use this arg to set TP-CP-DP-PP of each process mesh. "
            "This argument must be in the form: TP0, CP0, DP0, PP0, TP1, CP0, DP1, PP1...TPN, CPN, DPN, PPN. "
            "CP and TP size can be different, sum of PP should match pipeline-model-parallel-size, DP size should be the same."
        ),
    )
    group.add_argument(
        "--expert-tensor-parallel-size-per-process-mesh",
        nargs="*",
        type=int,
        default=None,
        help=(
            "The number of tensor parallel experts for each process-mesh. "
            "The number of the list should be equal to the number of process-meshes."
        ),
    )
    group.add_argument(
        "--hetero-use-cpu-communication",
        action="store_true",
        help="Use CPU for communication for heterogeneous communication.",
    )

    return parser


def _add_auto_tuner_args(parser):
    """Add auto tuner arguments (FlagScale specific)."""
    group = parser.add_argument_group(title="flagscale auto tuner")
    group.add_argument(
        "--auto-tune",
        action="store_true",
        help="use auto tuner",
    )
    return parser


def _add_auto_skip_spiky_loss(parser):
    """Add auto skip spiky loss arguments (FlagScale specific)."""
    group = parser.add_argument_group(title="flagscale auto skip spiky loss")
    group.add_argument(
        "--auto-skip-spiky-loss",
        action="store_true",
        help="Automatically skip spiky loss iterations.",
    )
    group.add_argument(
        "--spiky-loss-threshold",
        type=float,
        default=0.2,
        help="Threshold for skipping spiky loss iterations.",
    )
    return parser


def _add_peft_args(parser):
    """Add PEFT / LoRA arguments (FlagScale specific)."""
    group = parser.add_argument_group(title="flagscale peft")

    group.add_argument(
        "--peft-type",
        type=str,
        default=None,
        help="PEFT type",
    )
    group.add_argument(
        "--lora-target-modules",
        nargs="*",
        choices=[
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "linear_q_proj",
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
        ],
        default=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        help=(
            "LoRA target modules list. Valid choices: linear_qkv, linear_proj, "
            "linear_fc1, linear_fc2. Default selects all."
        ),
    )
    group.add_argument(
        "--lora-dim",
        type=int,
        default=8,
    )
    group.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )
    group.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="Dropout prob of lora linear",
    )
    group.add_argument(
        "--lora-dropout-position",
        type=str,
        default="pre",
        choices=["pre", "post"],
        help="Dropout position of lora linear",
    )
    group.add_argument(
        "--lora-in-init-method",
        type=str,
        default="xavier",
        choices=["normal", "kaiming", "xavier", "zero"],
        help="Init method of lora a",
    )
    group.add_argument(
        "--lora-out-init-method",
        type=str,
        default="zero",
        choices=["normal", "kaiming", "xavier", "zero"],
        help="Init method of lora b",
    )
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='flagscale network size')

    group.add_argument('--norm-init-weight', type=float, default=None,
                       help="Norm weight initialization.")
    group.add_argument('--multiple-of', type=int, default=None,
                       help='Multiplier for setting Feed-Forward Network hidden size when swiglu.')
    group.add_argument('--hidden-dim-multiplier', type=float, default=None,
                       help='Custom Multiplier for setting Feed-Forward Network hidden dim when swiglu.')
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='flagscale logging')

    group.add_argument('--wandb-mode', type=str, choices=['online', 'offline', 'disabled'], default='offline',
                       help='Can be "online", "offline" or "disabled". Defaults to "offline".')
    group.add_argument('--wandb-api-key', type=str, default='',
                       help='The wandb API keys and must be provided if using online mode.')
    group.add_argument('--wandb-log-model', action='store_true',
                       help='If set, write model to wandb.')
    group.add_argument('--wandb-log-model-interval', type=int, default=1000,
                       help='The interval to save the model to wandb.')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='flagscale training')

    group.add_argument('--recompute-granularity-per-stage-micro-batch', nargs='*', type=str, default=None,
                       help='used with recompute-granularity=full, setting recompute granularity'
                       'of each stage and each micro-batch. This argument must be a two-dimension list, '
                       'the sum of the first item of all the sub-lists should be equal to pipeline-model-parallel-size.'
                       'Every sub-list is in the form: n0, flag0, n1, flag1,... except the first item, which is the stage number.'
                       'The sum of n0, n1, ... should be equal to nums-micro-batch.'
                       'granularity flag: 0 means turning off full recompute, 1 means turning on')
    group.add_argument('--recompute-method-per-stage-micro-batch', nargs='*', type=str, default=None,
                       help='used with recompute-granularity=full, setting recompute method '
                       'of each stage and each micro-batch. This argument must be a two-dimension list, '
                       'the sum of the first item of all the sub-lists should be equal to pipeline-model-parallel-size.'
                       'Every sub-list is in the form: n0, flag0, n1, flag1,... except the first item, which is the stage number.'
                       'The sum of n0, n1, ... should be equal to nums-micro-batch.'
                       'method: 0 means uniform, 1 means block')
    group.add_argument('--recompute-num-layers-per-stage-micro-batch', nargs='*', type=str, default=None,
                       help='used with recompute-granularity=full, setting recompute num layers '
                       'of each stage and each micro-batch. This argument must be a two-dimension list, '
                       'Every sub-list is in the form: n0, num_laryers0, n1, num_laryers1,... except the first item, which is the stage number.'
                       'The sum of n0, n1, ... should be equal to nums-micro-batch. ')
    group.add_argument('--skip-samples-range', nargs='+', type=int, default=None,
                       help='Range of samples to skip during training.')
    group.add_argument('--skip-iters-range', nargs='+', type=int, default=None,
                       help='Range of iterations to skip during training.')
    group.add_argument('--use-dualpipev', action='store_true',
                       help='Use DualPipeV pipeline schedule method')
    group.add_argument('--moe-fb-overlap', action='store_true',
                       help='DualPipeV overlapping of moe a2a communication and forward/backward computation')
    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='flagscale learning rate')

    ## stablelm2-scheduler consists of multiple stages
    group.add_argument('--lr-decay-stablelm2-cosine-samples', type=int, default=0,
                       help='Samples number of cosine scheduler including warmup samples, used in stablelm2 scheduler.')
    group.add_argument('--lr-decay-stablelm2-cosine-max-lr', type=float, default=None,
                       help='Maximum lr of cosine scheduler, used in stablelm2 scheduler.')
    group.add_argument('--lr-decay-stablelm2-cosine-period-samples', type=int, default=0,
                       help='Period of cosine scheduler, used in stablelm2 scheduler.')
    group.add_argument('--lr-decay-stablelm2-rsqrt-samples', type=int, default=0,
                       help='Samples number of rsqrt scheduler used in stablelm2 scheduler.')
    group.add_argument('--lr-decay-stablelm2-decay-samples', type=int, default=0,
                       help='Samples number of decay scheduler used in stablelm2 scheduler.')
    group.add_argument('--lr-decay-stablelm2-alpha', type=float, default=1.0,
                       help='Numerator used in stablelm2 scheduler.')
    group.add_argument('--lr-decay-stablelm2-beta', type=float, default=0.0,
                       help='Denominator used in stablelm2 scheduler.')
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='flagscale checkpointing')
    
    group.add_argument('--rampup-save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.in the ramup phase.')
    group.add_argument('--save-when-num-microbatches-change', action='store_true',
                       help='Save param name to index maps only')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='flagscale distributed')
    
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-partial-reduce-for-shared-embedding', action='store_true',
                       help='Use partial reduce for shared word embedding.')
    group.add_argument('--no-shared-fs', action='store_true', 
                       help='Indicate whether not running on a shared file system.')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='flagscale validation')

    group.add_argument('--extra-eval-interval', type=int, default=None,
                       help='Interval between running evaluation on '
                       'extra validation sets.')
    return parser


def _add_tokenizer_args(parser):
    group = parser.add_argument_group(title='flagscale tokenizer')
    
    group.add_argument('--special-tokens-file', type=str, default=None,
                       help='Path to the BPE special tokens file.')
    group.add_argument('--tokenizer-path', type=str, default=None,
                       help='Path to the huggingface tokenizer.')
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='flagscale data')
    
    group.add_argument('--extra-valid-data-path', nargs='*', default=None,
                       help='The weight, prefix list for an independent extra validation dataset. '
                       'The accepted format is a list of weight, prefix and tag, '
                       'e.g. weight1 prefix1 tag1 weight2 prefix2 tag2. '
                       'The weight1 means the number of tokens in the prefix1 dataset. ')
    group.add_argument('--finetune-dataset-type', type=str, default=None,
                       choices=['CPT', None],
                       help='datasets type during finetunning.')
    group.add_argument('--apply-sft-dataset-separated-loss-mask-if-existed', action='store_true',
                       help='If set, use sft dataset with separated loss mask files, '
                       'if _loss_mask_document.bin and _loss_mask_document.idx existed.')
    return parser


def _add_vision_args(parser):
    group = parser.add_argument_group(title='flagscale vision')
    
    group.add_argument('--qk-layernorm-hidden-dim', action='store_true',
                       help='Whether to layer normalize the q and k attention embeddings on hidden dimension rather than head dimension')
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='flagscale regularization')

    group.add_argument('--muon-matched-adamw-rms', type=float, default=0.2,
                       help="The RMS of the matched AdamW's, typically 0.2 ~ 0.4")
    group.add_argument('--muon-momentum', type=float, default=0.95,
                       help='Momentum beta for muon')
    group.add_argument('--muon-ns-steps', type=int, default=5,
                       help='Number of Newton-Schultz iteartion steps for muon')
    group.add_argument('--no-muon-nesterov', action='store_false',
                       dest='muon_nesterov', default=True,
                       help='If set, disable Nesterov momentum for muon')
    return parser


def _add_flagos_args(parser):
    group = parser.add_argument_group(title="flagscale transformer engine fl")
    group.add_argument('--te-fl-prefer', type=str, choices=['flagos', 'vendor', 'reference'], default='vendor',
                       help='Backend selection for transformer engine fl.')
    group.add_argument('--te-fl-per-op', type=str, default=None,
                       help='Backend selection for custom ops.')
    group.add_argument('--te-fl-allow-vendors', type=str, default=None,
                       help='Allow vendors for transformer engine fl.')
    group.add_argument('--te-fl-deny-vendors', type=str, default=None,
                       help='Deny vendors for transformer engine fl.')
    group.add_argument('--enable-flag-gems', action='store_true',
                       help='Enable flag gems to replace torch ops for distributed training.')
    group.add_argument('--flag-gems-log-path', type=str, default=None,
                        help='Path of flag gems logging')
    group.add_argument(
        '--flag-gems-unused',
        nargs='*',
        default=None,
        help='Flag Gems unused ops list'
    )
    return parser


def _add_engram_args(parser):
    group = parser.add_argument_group(title="flagscale engram")
    group.add_argument('--use-engram', action='store_true',
                       help='Use Engram module.')
    group.add_argument(
        '--engram-tokenizer-name-or-path',
        type=str,
        default=None,
        help='Tokenizer name or path used by Engram',
    )
    group.add_argument(
        '--engram-vocab-size',
        nargs='*',
        type=int,
        default=None,
        help='Engram vocab size per layer (list of ints)',
    )
    group.add_argument(
        '--max-ngram-size',
        type=int,
        default=1,
        help='Maximum n-gram size for Engram',
    )
    group.add_argument(
        '--n-embed-per-ngram',
        type=int,
        default=None,
        help='Embedding dimension per n-gram',
    )
    group.add_argument(
        '--n-head-per-ngram',
        type=int,
        default=1,
        help='Number of heads per n-gram',
    )
    group.add_argument(
        '--engram-layer-ids',
        nargs='*',
        type=int,
        default=None,
        help='Layer ids where Engram is applied',
    )
    group.add_argument(
        '--engram-pad-id',
        type=int,
        default=0,
        help='Pad token id for Engram hashing',
    )
    group.add_argument(
        '--engram-seed',
        type=int,
        default=0,
        help='Random seed for Engram hashing',
    )
    group.add_argument(
        '--engram-kernel-size',
        type=int,
        default=1,
        help='Kernel size for Engram short convolution',
    )
    group.add_argument(
        '--engram-hc-mult',
        type=int,
        default=1,
        help='Hyper-connection multiplicity for Engram',
    )
    return parser


def add_flagscale_arguments(parser):
    """
    Add all FlagScale-specific arguments to a Megatron parser.

    This is intended to be passed as part of Megatron's extra_args_provider
    so that Megatron-LM-FL itself stays free of FlagScale-specific options.
    """
    parser = _add_network_size_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_training_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_tokenizer_args(parser)
    parser = _add_data_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_hetero_args(parser)
    parser = _add_auto_tuner_args(parser)
    parser = _add_auto_skip_spiky_loss(parser)
    parser = _add_peft_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_flagos_args(parser)
    parser = _add_engram_args(parser)
    return parser
