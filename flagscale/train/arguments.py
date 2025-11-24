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

from flagscale.train.hetero.parallel_context import RankMapper


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

        assert (
            self.args.hetero_process_meshes is not None
        ), "hetero_process_meshes should be specified when enable_hetero is True"
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

        # TODO: update other args if need
