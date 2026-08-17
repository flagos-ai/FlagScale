# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from flagscale.train.megatron.train_qwen3_vl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
from functools import partial
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch._dynamo
import torch.distributed as dist

from argparse import Namespace

from megatron.core import parallel_state
from megatron.training.checkpointing import get_checkpoint_name
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from megatron.training.utils import unwrap_model
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from megatron.training.training import pretrain
stimer = StragglerDetector()

# Qwen2.5-VL data handling
from megatron.core.num_microbatches_calculator import get_num_microbatches
torch._dynamo.config.suppress_errors = True
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
)
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)

from megatron.training.tokenizer.tokenizer import build_tokenizer
from megatron.training.global_vars import get_tokenizer

from flagscale.models.megatron.qwen2_5_vl.tensor_parallel import broadcast_data

from flagscale.models.megatron.qwen35.qwen35_model import Qwen35Model
from flagscale.models.megatron.qwen35.qwen35_mimo_model import Qwen35MIMOModel
from flagscale.models.mimo.bridge.providers.qwen35 import (
    LANGUAGE_MODULE_NAME,
    VISION_MODALITY_NAME,
    qwen35_grid_mimo_model_provider,
)
from flagscale.models.megatron.qwen35.transformer_config import (
    Qwen35TransformerConfig,
    get_vision_model_config,
    get_vision_projection_config,
)
from flagscale.models.megatron.qwen35.layer_specs import (
    get_qwen35_language_model_spec,
    get_qwen35_mtp_block_spec,
    get_mlp_module_spec
)
from flagscale.models.megatron.qwen3_vl.layer_specs import get_qwen3vl_vision_model_spec

from flagscale.models.mimo import (
    ModuleParallelismConfig,
    build_colocated_pg_collections,
    validate_mimo_config,
)
from flagscale.models.mimo.bridge.recipe.qwen35 import (
    build_qwen35_grid_config_from_args,
    compute_qwen35_grid_sequence_parallel,
    compute_qwen35_pipeline_layer_split,
    describe_qwen35_grid_modules,
    qwen35_grid_data_contract,
)
from flagscale.models.mimo.bridge.training import (
    GridTrainingState,
    build_grid_multimodule_communicator,
    build_language_forward_kwargs,
    build_vision_forward_kwargs,
    finalize_grid_training_state,
    reconfigure_grid_num_microbatches_calculator,
    validate_qwen35_grid_runtime_contract,
)
from flagscale.models.mimo.bridge.infra import build_mimo_infra
from flagscale.models.mimo.bridge.runtime import validate_no_stub_ranks

from megatron.plugin.platform import get_platform
cur_platform = get_platform()

from tools.datasets.qwenvl.data.dataset_helpers import TaskEncoder, print_error_handler

IGNORE_IDX = -100


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True,
    vp_stage=None, config=None, pg_collection=None, **kwargs
) -> Union[Qwen35Model]:
    """Provide a Qwen3.5 model instance."""
    args = get_args()
    print_rank_0("start building qwen3.5 model ...")

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')

            filename = f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
            torch.cuda.memory._dump_snapshot(filename)

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    # Build transformer config with Qwen35 config class
    if config is None:
        config = core_transformer_config_from_args(args, Qwen35TransformerConfig)
    # Qwen3.5 uses zero-centered gamma for RMSNorm; override if needed
    # (core_transformer_config_from_args may be affected by apply_layernorm_1p)
    config.layernorm_zero_centered_gamma = getattr(args, 'layernorm_zero_centered_gamma', True)
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("Qwen3.5 model is only implemented with TransformerEngine!")

    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently does not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    # Vision configs (identical encoder to Qwen3-VL)
    enable_vision = getattr(args, 'enable_vision', True)
    if enable_vision:
        vision_config = get_vision_model_config(args, deepcopy(config))
        vision_config.pipeline_model_parallel_size = 1
        vision_config.first_pipeline_num_layers = None
        vision_projector_config = get_vision_projection_config(
            deepcopy(config), vision_config.hidden_size, args.spatial_merge_size
        )
    else:
        print_rank_0("Vision is disabled, building text-only model...")
        vision_config = None
        vision_projector_config = None

    print_rank_0("building Qwen3.5 model in TE...")

    # Language model spec: hybrid GDN + Attention
    language_layer_spec = get_qwen35_language_model_spec(config, vp_stage=vp_stage)

    # Vision model spec (identical to Qwen3-VL)
    if enable_vision:
        vision_model_spec = get_qwen3vl_vision_model_spec()
        vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules
    else:
        vision_model_spec = None
        vision_projector_spec = None

    if args.enable_variable_seq_lengths:
        config.variable_seq_lengths = True

    # MTP (Multi-Token Prediction) spec
    mtp_block_spec = get_qwen35_mtp_block_spec(args, config)

    # args.padded_vocab_size = args.vocab_size
    # print(f"set args.padded_vocab_size to before init model: {args.padded_vocab_size=}")

    if args.use_mimo:
        assert enable_vision, (
            "--use-mimo requires the vision module; it is incompatible with "
            "--no-enable-vision"
        )
        # ``--mimo-layout`` is a subordinate selector of ``--use-mimo``: both
        # layouts are first-class, explicitly named branches; anything else is
        # a bug (argparse ``choices`` normally guarantees the value set).
        mimo_layout = getattr(args, "mimo_layout", "colocated")
        if mimo_layout == "grid":
            # Non-colocated grid MIMO (MCore MimoModel / HyperCommGrid path).
            # Every rank participates in exactly one module; the language module
            # config mirrors the global args (TP/PP below are the language's).
            if getattr(args, "mtp_num_layers", None):
                raise ValueError(
                    "Qwen3.5 non-colocated grid: MTP (mtp_num_layers > 0) is not "
                    "supported in this stage - the MCore MimoModel grid path has "
                    "no MTP wiring. Set mtp_num_layers=0 (fail-fast)."
                )
            world_size = torch.distributed.get_world_size()
            mimo_config = build_qwen35_grid_config_from_args(
                args.mimo_module_specs,
                world_size,
                num_layers=config.num_layers,
                num_mtp_layers=getattr(args, "mtp_num_layers", None),
            )

            # Batch contract: the sampler is unsharded (data_parallel_size == 1)
            # and module-local DP slicing happens in the forward step.
            num_microbatches = get_num_microbatches()
            per_module_dp = qwen35_grid_data_contract(
                mimo_config,
                micro_batch_size=args.micro_batch_size,
                global_batch_size=args.global_batch_size,
                num_microbatches=num_microbatches,
            )
            print_rank_0(
                f"Non-colocated grid MIMO: {describe_qwen35_grid_modules(mimo_config)}; "
                f"module DP: {per_module_dp}"
            )

            # Grid + nullable process groups.  Collective on every world rank.
            infra = build_mimo_infra(mimo_config.module_parallelisms)
            validate_no_stub_ranks(infra.module_to_grid_map, world_size)

            # Grid mode runs the num-microbatches calculator with DP=1; the
            # module-level contract was validated above by qwen35_grid_data_contract.
            language_parallelism = mimo_config.get_parallelism(LANGUAGE_MODULE_NAME)
            vision_parallelism = mimo_config.get_parallelism(VISION_MODALITY_NAME)

            # The language transformer config must describe the *language* module
            # (per-stage layer slicing via pipeline_model_parallel_size), while the
            # global parallel state is initialized with TP=1/PP=1 in grid mode.
            config.tensor_model_parallel_size = language_parallelism.tensor_model_parallel_size
            config.pipeline_model_parallel_size = language_parallelism.pipeline_model_parallel_size
            config.data_parallel_size = language_parallelism.data_parallel_size
            config.context_parallel_size = 1
            config.expert_model_parallel_size = 1

            # Uneven pipeline layer allocation: MCore supports explicit first/last
            # stage layer counts (TransformerConfig.num_layers_in_first/last_pipeline_stage),
            # so non-divisible layer counts (e.g. 32 layers with PP3/PP6) are valid
            # - the first stage gets base+remainder layers, the last stage base
            # (32 layers -> PP3 12/10/10, PP6 7/5/5/5/5/5; the middle stages split
            # evenly by construction).  Even splits leave the fields None (the
            # default MCore even split).  The user-facing pipeline-allocation
            # overrides that would conflict with this are rejected at startup
            # (see __main__).
            split = compute_qwen35_pipeline_layer_split(
                config.num_layers, config.pipeline_model_parallel_size
            )
            if len(set(split)) > 1:
                config.num_layers_in_first_pipeline_stage = split[0]
                config.num_layers_in_last_pipeline_stage = split[-1]

            # The vision encoder TP-shards by its own module TP (config field +
            # per-module pg_collection), not by the global parallel state.
            vision_config.tensor_model_parallel_size = vision_parallelism.tensor_model_parallel_size
            vision_config.context_parallel_size = 1
            vision_config.expert_model_parallel_size = 1
            vision_projector_config.tensor_model_parallel_size = (
                vision_parallelism.tensor_model_parallel_size
            )
            vision_projector_config.context_parallel_size = 1
            vision_projector_config.expert_model_parallel_size = 1
            # Vision is never pipelined in MIMO; the ViT asserts post_process.
            vision_config.pipeline_model_parallel_size = 1

            # Sequence parallelism is a per-module property in grid mode: the
            # global parallel state is TP=1 and the global args.sequence_parallel
            # was forced False at startup, so the user's requested value
            # (preserved in args.mimo_sequence_parallel) applies only where a
            # module has TP > 1 AND its implementation supports SP.
            # ``compute_qwen35_grid_sequence_parallel`` conservatively marks BOTH
            # modules SP-incapable in this grid path (its default is the empty
            # capable set): the language module cannot use SP because the grid
            # forward does not shard the embeddings (QwenVLLanguageModelEmbedding
            # asserts no scatter-to-SP) and the mRoPE freqs stay full-length - an
            # SP-enabled TP2 qkv would all-gather dim 0 to 2x the sequence (4096
            # vs freqs 2048).  The Qwen3-VL vision encoder has the same packed-seq
            # limitation (6720-vs-3360).  Requested global SP therefore resolves
            # to per-module False for every accepted layout; TP itself is
            # unaffected (language-TP2 layouts remain valid).
            requested_sp = bool(getattr(args, "mimo_sequence_parallel", False))
            per_module_sp = compute_qwen35_grid_sequence_parallel(mimo_config, requested_sp)
            config.sequence_parallel = per_module_sp[LANGUAGE_MODULE_NAME]
            vision_config.sequence_parallel = per_module_sp[VISION_MODALITY_NAME]
            if requested_sp and not any(per_module_sp.values()):
                print_rank_0(
                    "Non-colocated grid MIMO: requested sequence parallelism is "
                    "disabled for BOTH modules. The language module cannot use SP "
                    "in this grid path: the grid forward does not shard the "
                    "embeddings and the mRoPE freqs remain full-length, so the "
                    "TP2 column-parallel qkv all-gather would double dim 0 "
                    "(4096 tokens vs freqs 2048). The Qwen3-VL vision encoder "
                    "likewise cannot use SP (packed-seq attention/rotary operate "
                    "on the full token dimension). Both modules keep tensor "
                    "parallelism with SP disabled (fail-safe)."
                )

            # Per-module RNG: the standard path seeds by the *global* TP/PP ranks
            # (all zero in grid mode), which would initialize TP-sharded module
            # weights differently across a module's TP group.  Re-seed by each
            # module's own TP/PP/EP ranks (Bridge-style) before building.
            _set_per_module_random_seed(args, infra)

            # Per-rank language PP stage flags: the language module is built per
            # stage (embedding on the first stage, output layer on the last).  The
            # layer count per stage comes from the language config's
            # pipeline_model_parallel_size (== language_parallelism.PP).
            language_grid = infra.module_to_grid_map.get(LANGUAGE_MODULE_NAME)
            language_pg = infra.module_to_pg_collection.get(LANGUAGE_MODULE_NAME)
            # Language PP rank of this rank (0 for encoder-only ranks, where the
            # language module is never built).  Mirrors the safe pattern in
            # ``_set_per_module_random_seed``: never call ``get_group_rank`` on a
            # nullable module collection - ``None`` means the rank is outside the
            # language grid, not that the language module has PP=1.
            language_pp_rank = 0
            if language_pg is not None:
                pre_process = is_pp_first_stage(language_pg.pp)
                post_process = is_pp_last_stage(language_pg.pp)
                if language_grid is not None and language_grid.is_current_rank_in_grid():
                    language_pp_rank = torch.distributed.get_group_rank(
                        language_pg.pp, torch.distributed.get_rank()
                    )
            else:
                # Encoder-only rank: no language module is built (role-driven).
                pre_process = post_process = True

            # The language layer spec must describe *this* rank's PP stage.  It
            # was built above from the pre-mutation config, when the global
            # parallel state (TP=1/PP=1 in grid mode) made every stage slice all
            # ``config.num_layers`` layer specs: both stages would build the full
            # stack and save overlapping checkpoint keys (e.g. layers 12..23 for
            # a 24-layer model with PP2).  Rebuild it now with the language
            # module's explicit PP rank so each stage slices exactly its own
            # layers (24 layers -> PP2: 12 + 12 at global offsets 0 and 12;
            # 32 layers -> PP3: 12/10/10 at offsets 0/12/22).  The offset/count
            # come from the language config's pipeline fields set above, and the
            # built layers still get the same global numbering at construction
            # time (``TransformerLayer.layer_number``), so checkpoint keys stay
            # non-overlapping across stages.
            language_layer_spec = get_qwen35_language_model_spec(
                config, pp_rank=language_pp_rank
            )

            model = qwen35_grid_mimo_model_provider(
                language_transformer_config=config,
                language_transformer_layer_spec=language_layer_spec,
                language_vocab_size=args.padded_vocab_size,
                language_max_sequence_length=args.max_position_embeddings,
                vision_transformer_config=vision_config,
                vision_transformer_layer_spec=vision_model_spec,
                vision_projection_config=vision_projector_config,
                vision_projection_layer_spec=vision_projector_spec,
                vision_projection_type='mlp',
                language_position_embedding_type=args.position_embedding_type,
                language_rotary_percent=args.rotary_percent,
                language_rotary_base=args.rotary_base,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                mtp_block_spec=mtp_block_spec,
                mimo_infra=infra,
            )

            # Training-lifecycle state: schedule PGs / communicator + the local
            # module's collection for logging/checkpoint reductions.
            grid_state = GridTrainingState(
                infra=infra,
                parallelism_config=mimo_config,
                world_size=world_size,
            )
            finalize_grid_training_state(grid_state)
            model.mimo_grid_state = grid_state
            model.pg_collection = grid_state.local_pg_collection
            build_grid_multimodule_communicator(grid_state, model)
            print_rank_0(
                f"Rank {torch.distributed.get_rank()}: grid module "
                f"'{grid_state.active_module_name}' ("
                f"tp={torch.distributed.get_world_size(grid_state.local_pg_collection.tp)}, "
                f"dp={torch.distributed.get_world_size(grid_state.local_pg_collection.dp)}, "
                f"pp={torch.distributed.get_world_size(grid_state.local_pg_collection.pp)})"
            )
        elif mimo_layout == "colocated":
            # Colocated MIMO: vision and language modules run under different
            # parallel configurations on the same ranks.
            world_size = torch.distributed.get_world_size()
            vision_tp = getattr(args, "vision_tensor_model_parallel_size", None) or args.tensor_model_parallel_size
            vision_pp = getattr(args, "vision_pipeline_model_parallel_size", None) or args.pipeline_model_parallel_size
            # Vision DP is always derived, never set manually (same as Megatron DP).
            vision_dp = world_size // vision_tp // vision_pp
            vision_parallelism = ModuleParallelismConfig(
                tensor_model_parallel_size=vision_tp,
                pipeline_model_parallel_size=vision_pp,
                data_parallel_size=vision_dp,
            )
            language_parallelism = ModuleParallelismConfig(
                tensor_model_parallel_size=args.tensor_model_parallel_size,
                pipeline_model_parallel_size=args.pipeline_model_parallel_size,
                data_parallel_size=world_size // args.tensor_model_parallel_size // args.pipeline_model_parallel_size,
                expert_model_parallel_size=getattr(args, "expert_model_parallel_size", 1),
            )
            pg_collections = build_colocated_pg_collections(
                vision_parallelism, language_parallelism, world_size
            )
            pg_summary = ", ".join(
                f"{name}(tp={dist.get_world_size(pgs.tp)}, "
                f"dp={dist.get_world_size(pgs.dp)}, "
                f"pp={dist.get_world_size(pgs.pp)})"
                for name, pgs in pg_collections.items()
            )
            print_rank_0(f"MIMO process group collections: {pg_summary}")

            # Single-point validation of model-agnostic MIMO config constraints.
            vit_batch_factor = validate_mimo_config(
                args, vision_parallelism, language_parallelism, get_num_microbatches()
            )
            print_rank_0(
                f"MIMO vit_batch_factor={vit_batch_factor} "
                f"(vision_dp={vision_parallelism.data_parallel_size}, "
                f"language_dp={language_parallelism.data_parallel_size}, "
                f"num_microbatches={get_num_microbatches()})"
            )

            model = Qwen35MIMOModel(
                language_transformer_config=config,
                language_transformer_layer_spec=language_layer_spec,
                language_vocab_size=args.padded_vocab_size,
                language_max_sequence_length=args.max_position_embeddings,

                vision_transformer_config=vision_config,
                vision_transformer_layer_spec=vision_model_spec,
                vision_projection_config=vision_projector_config,
                vision_projection_layer_spec=vision_projector_spec,
                pg_collections=pg_collections,
                vision_parallelism=vision_parallelism,
                language_parallelism=language_parallelism,

                vision_projection_type='mlp',
                language_position_embedding_type=args.position_embedding_type,
                language_rotary_percent=args.rotary_percent,
                language_rotary_base=args.rotary_base,

                pre_process=pre_process,
                post_process=post_process,
                add_decoder=add_decoder,
                add_encoder=add_encoder,

                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                mtp_block_spec=mtp_block_spec,
                vit_batch_factor=vit_batch_factor,
                use_fp32_grad_cache=getattr(args, "mimo_fp32_grad_cache", False),
            )

            # Attach the language pg_collection to the wrapper for compatibility with
            # code that expects a top-level pg_collection attribute.
            model.pg_collection = pg_collections["language"]
        else:
            assert False, (
                f"Unsupported --mimo-layout {mimo_layout!r}: "
                "expected 'colocated' or 'grid'"
            )
    else:
        model = Qwen35Model(
            language_transformer_config=config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=args.padded_vocab_size,
            language_max_sequence_length=args.max_position_embeddings,

            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_model_spec,
            vision_projection_config=vision_projector_config,
            vision_projection_layer_spec=vision_projector_spec,
            vision_projection_type='mlp',

            language_position_embedding_type=args.position_embedding_type,
            language_rotary_percent=args.rotary_percent,
            language_rotary_base=args.rotary_base,

            pre_process=pre_process,
            post_process=post_process,
            add_decoder=add_decoder,
            add_encoder=add_encoder,
            enable_vision=enable_vision,

            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT if enable_vision else False,
        freeze_vision_projection=False,
    )

    return model


def _set_per_module_random_seed(args, infra) -> None:
    """Re-seed Python/NumPy/torch/MCore RNG by the rank's module TP/PP ranks.

    In grid mode the global parallel state is initialized with TP=1/PP=1, so
    the standard seed path gives every rank the same seed; TP-sharded module
    weights would then be initialized differently across a module's TP group.
    Mirror the Megatron-Bridge ``_set_per_module_random_seeds``: seed by the
    module's own PP rank (different stages get different seeds) and fork the
    MCore CUDA RNG tracker with the module's TP/EP/ETP ranks.
    """
    import random

    import numpy as np
    from megatron.core import tensor_parallel

    seed = args.seed
    tp_rank = ep_rank = etp_rank = 0
    pp_rank = 0
    for module_name, grid in infra.module_to_grid_map.items():
        if not grid.is_current_rank_in_grid():
            continue
        pg_collection = infra.module_to_pg_collection.get(module_name)
        if pg_collection is None:
            continue
        current_rank = torch.distributed.get_rank()
        tp_rank = torch.distributed.get_group_rank(pg_collection.tp, current_rank)
        pp_rank = torch.distributed.get_group_rank(pg_collection.pp, current_rank)
        if getattr(pg_collection, "ep", None) is not None:
            ep_rank = torch.distributed.get_group_rank(pg_collection.ep, current_rank)
        if getattr(pg_collection, "expt_tp", None) is not None:
            etp_rank = torch.distributed.get_group_rank(pg_collection.expt_tp, current_rank)
        break

    pp_seed = seed + (100 * pp_rank)
    random.seed(pp_seed)
    np.random.seed(pp_seed)
    torch.manual_seed(pp_seed)
    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(
            pp_seed, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank
        )


def get_ltor_masks_and_position_ids(
    input_ids,
    image_thw_grids,
    video_thw_grids,
    target,
    pad_token,
    second_per_grid_ts,
    ignore_index=None,
    model: Qwen35Model = None,
):
    """Build masks and position ids for left-to-right model."""
    args = get_args()

    if not getattr(args, 'enable_vision', True):
        # Text-only: position_ids is [3, B, S] with all three dimensions identical
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_ids = pos.unsqueeze(0).expand(3, -1, -1)
    else:
        # Multimodal: compute mRoPE position indices from vision grids
        position_ids, _ = model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_thw_grids,
            video_grid_thw=video_thw_grids,
            attention_mask=input_ids != pad_token,
        )

    # Loss mask
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0

    # Attention mask
    attention_mask = None

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, model: Qwen35Model = None) -> Dict:
    """Generate a batch."""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    args = get_args()
    enable_vision = getattr(args, 'enable_vision', True)

    cur_platform.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        pad_token_id = IGNORE_IDX
        while (data["target"] == pad_token_id).all():
            logging.getLogger(__name__).warning(
                "The current data is invalid because the target is all pad_token_id! "
                "Get next data to avoid fail, but it's better to check the data!"
            )
            data = next(data_iterator)
    else:
        data = None

    data_text = broadcast_data(["text"], data, torch.int64)["text"]
    target = broadcast_data(["target"], data, torch.int64)["target"]

    if enable_vision:
        imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]
        videos = broadcast_data(["videos"], data, torch.float32)["videos"]
        image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]
    else:
        imgs = None
        videos = None
        image_thw_grids = None

    if enable_vision:
        video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
        second_per_grid_ts = broadcast_data(['second_per_grid_ts'], data, torch.float32)['second_per_grid_ts']
        image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
        video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
    else:
        video_thw_grids = None
        second_per_grid_ts = None
        image_input_mask = None
        video_input_mask = None
    cur_platform.range_pop()

    cur_platform.range_push("index tokens")
    tokenizer = get_tokenizer()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    cur_platform.range_pop()

    cur_platform.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        image_thw_grids,
        video_thw_grids,
        labels,
        pad_token=tokenizer.pad_token_id,
        second_per_grid_ts=second_per_grid_ts,
        ignore_index=IGNORE_IDX,
        model=model,
    )
    cur_platform.range_pop()

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "imgs": imgs,
        "videos": videos,
        "image_thw_grids": image_thw_grids,
        "video_thw_grids": video_thw_grids,
        "image_input_mask": image_input_mask,
        "video_input_mask": video_input_mask,
    }


SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[Qwen35Model] = None,
):
    """Loss function."""
    args = get_args()

    if has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )

    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def _grid_prepare_batch(batch, model, grid_state) -> Dict:
    """Prepare the global micro-batch for this rank's grid module role.

    Every data-loading rank samples the *same* global micro-batch
    (``args.data_parallel_size == 1`` in grid mode, broadcast over the world
    TP group).  This function then:

    1. drops the raw modality inputs on language-only ranks (they consume
       encoder outputs from the MIMO bridge) and assembles the exact kwargs
       accepted by ``Qwen35GridMIMOModel.forward`` (no leftover batch keys),
    2. contiguously slices the batch for the module-local DP shard,
    3. nulls out fields the module does not consume (input_ids on non-first
       language PP stages; labels/loss_mask on non-last stages),
    4. assembles ``modality_inputs`` for vision ranks.
    """
    from flagscale.models.mimo.bridge import (
        ModuleDataRole,
    )

    module_name = grid_state.active_module_name
    grid = grid_state.infra.module_to_grid_map[module_name]
    dp_size = grid.shape[grid.dim_names.index("dp")]
    pg_collection = grid_state.local_pg_collection
    dp_rank = torch.distributed.get_group_rank(pg_collection.dp, torch.distributed.get_rank())
    pp_size = pg_collection.pp.size()
    pp_rank = torch.distributed.get_group_rank(pg_collection.pp, torch.distributed.get_rank())
    role = ModuleDataRole(module_name=module_name, pp_rank=pp_rank, pp_size=pp_size)

    if role.is_language:
        # Language-only ranks (non-colocated) get encoder outputs from the
        # bridge.  build_language_forward_kwargs emits exactly the keyword
        # arguments the model forward accepts - nulled leftovers such as
        # ``imgs`` / ``videos`` / ``image_thw_grids`` / ``video_thw_grids``
        # would make ``model(**data_batch)`` raise TypeError.  The raw
        # patch-packed modality tensors are dropped before the sample-DP slice
        # (their leading dim is the total patch count, not the sample batch).
        return build_language_forward_kwargs(
            batch,
            dp_rank=dp_rank,
            dp_size=dp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
        )

    # Vision ranks: build_vision_forward_kwargs slices the global micro-batch
    # for the vision module's DP - the patch-packed raw modality tensors are
    # sliced JOINTLY along per-image boundaries (vision DP 2 layouts), all
    # other keys by sample - and assembles ``modality_inputs``.  Videos are
    # not supported by the grid path yet - fail fast instead of producing a
    # silent embedding-count mismatch.
    return build_vision_forward_kwargs(batch, dp_rank=dp_rank, dp_size=dp_size)


def forward_step(data_iterator, model):
    """Forward training step."""
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    global stimer

    unwrapped = unwrap_model(model)
    vision_output = None
    if args.use_mimo:
        mimo_layout = getattr(args, "mimo_layout", "colocated")
        if mimo_layout == "grid":
            # Non-colocated grid path: all ranks load the same global micro-batch
            # and slice it for their module-local DP shard; the MCore MimoModel
            # dispatches by rank role (encoder forward on vision ranks, language
            # forward on language ranks) and the multi-module pipeline schedule
            # moves activations between the modules.
            with stimer(bdata=True):
                batch = get_batch(data_iterator, model=unwrapped)
            timers('batch-generator').stop()
            grid_state = getattr(unwrapped, "mimo_grid_state", None)
            assert grid_state is not None, "grid forward step requires mimo_grid_state"
            data_batch = _grid_prepare_batch(batch, unwrapped, grid_state)

            output = unwrapped(**data_batch)
            if isinstance(output, tuple):
                output_tensor, model_loss_mask = output
            else:
                output_tensor, model_loss_mask = output, None

            if grid_state.is_language_last_stage:
                loss_mask = model_loss_mask
                if loss_mask is None:
                    loss_mask = data_batch.get("loss_mask")
                if loss_mask is None:
                    raise RuntimeError(
                        "Grid language last stage requires a loss_mask for the loss "
                        "function; got None."
                    )
                return output_tensor, partial(loss_func, loss_mask, model=model)
            # Encoder ranks and non-last language PP stages return no loss.
            return output_tensor, None
        elif mimo_layout == "colocated":
            # MIMO scheduler path: the model owns the scheduler; it assembles a new
            # ViT macro batch when the current one is exhausted and returns the next
            # LLM microbatch together with its vision output.
            batch, vision_output = unwrapped.next_microbatch(data_iterator, get_batch)
            vision_data = None
            vision_grid = None
        else:
            assert False, (
                f"Unsupported --mimo-layout {mimo_layout!r}: "
                "expected 'colocated' or 'grid'"
            )
    else:
        with stimer(bdata=True):
            batch = get_batch(data_iterator, model=unwrapped)
        if getattr(args, 'enable_vision', True):
            vision_data = torch.cat([batch["imgs"], batch["videos"]], dim=0)
            vision_grid = torch.cat([batch["image_thw_grids"], batch["video_thw_grids"]], dim=0)
        else:
            vision_data = None
            vision_grid = None
    timers('batch-generator').stop()

    enable_vision = getattr(args, 'enable_vision', True)

    model_kwargs = dict(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
    )
    if enable_vision:
        model_kwargs.update(
            vision_data=vision_data,
            vision_grid_thw=vision_grid,
            video_start_index=batch["image_input_mask"].sum().cpu().item(),
            image_input_mask=batch["image_input_mask"],
            video_input_mask=batch["video_input_mask"],
        )
    if args.use_mimo:
        model_kwargs["vision_output"] = vision_output

    with stimer:
        output_tensor = model(**model_kwargs)

    return output_tensor, partial(loss_func, batch["loss_mask"], model=model)


def run_online_eval(model):
    """Run evaluation during training."""
    return []


def write_online_eval_to_tensorboard(data, iteration, writer):
    """Write online evaluation data to Tensorboard."""
    if not writer:
        return
    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)


###############################################################################
# Text-only (GPT-style) data loading — used when enable_vision=False
###############################################################################

from flagscale.train.megatron.train_gpt import (
    get_batch as get_batch_gpt,
    train_valid_test_datasets_provider as train_valid_test_datasets_provider_gpt,
    get_embedding_ranks,
)


def get_batch_text(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch for text-only training (GPT-style), with mRoPE position_ids."""
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch_gpt(
        data_iterator, vp_stage
    )

    # Expand position_ids from [B, S] to [3, B, S] for mRoPE (text-only: all 3 dims identical)
    if position_ids is not None:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def forward_step_text(data_iterator, model: Qwen35Model, return_schedule_plan: bool = False):
    """Forward training step for text-only mode."""
    args = get_args()
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch_text(
            data_iterator, vp_stage
        )
    timers('batch-generator').stop()

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        else:
            output_tensor = model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
            )

    return output_tensor, partial(loss_func, loss_mask, model=model)


###############################################################################
# Multimodal (energon) data loading — used when enable_vision=True
###############################################################################

def datasets_provider(worker_config=None):
    """Create multimodal train, validation and test datasets."""
    args = get_args()
    dname = args.data_path[0] if type(args.data_path) is list else args.data_path
    train_dataset = get_train_dataset(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        virtual_epoch_length=0,
        max_samples_per_sequence=args.max_samples_per_sequence,
        shuffle_buffer_size=args.shuffle_buffer_size,
        handler=print_error_handler,
        repeat=True,
        image_decode="pil",
    )
    val_datasets_without_source_datasets = None
    if args.eval_iters > 0:
        val_datasets = get_val_datasets(
            dname,
            batch_size=args.micro_batch_size,
            task_encoder=TaskEncoder(),
            worker_config=worker_config,
            handler=print_error_handler,
            image_decode="pil",
        )
        val_datasets_without_source_datasets = [
            LimitDataset(
                RepeatDataset(val_ds, worker_config=worker_config),
                length=args.eval_iters * get_num_microbatches(),
                worker_config=worker_config,
                reset_after_epoch=True,
            )
            for val_ds, _src_ds in val_datasets
        ]

    return train_dataset, val_datasets_without_source_datasets, None


def is_dataloader_rank(transformer_pipeline_model_parallel_size):
    """Check if we should have the dataloader on this rank."""
    is_first_rank = get_tensor_model_parallel_rank() == 0
    return is_first_rank


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
    if not is_dataloader_rank(args.transformer_pipeline_model_parallel_size):
        return None, None, None

    worker_debug_path = None
    worker_log_level = 0

    # Baseline and the colocated layout shard the sampler by the (global) DP
    # group; the grid layout instead loads the same global micro-batch on every
    # data-loading rank (module-local DP slicing happens in the forward step),
    # so the grid sampler must not shard.
    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()
    if args.use_mimo:
        mimo_layout = getattr(args, "mimo_layout", "colocated")
        if mimo_layout == "colocated":
            pass  # colocated: standard DP-sharded sampler (same as baseline)
        elif mimo_layout == "grid":
            rank = 0
            world_size = 1
            data_parallel_group = torch.distributed.group.WORLD
        else:
            assert False, (
                f"Unsupported --mimo-layout {mimo_layout!r}: "
                "expected 'colocated' or 'grid'"
            )

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )
    train_ds, valid_ds1, test_ds = datasets_provider(worker_config)

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu", weights_only=False)
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print_rank_0(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print_rank_0("loading dataloader checkpoint failed. Skipping. " + str(e))

    if valid_ds1 is not None:
        valid_dataloader = [
            EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
            for valid_ds in valid_ds1
        ]
    else:
        valid_dataloader = EnergonDataloader(None)
    test_dataloader = None

    return EnergonDataloader(train_dataloader), valid_dataloader, EnergonDataloader(test_dataloader)


class EnergonDataloader:
    """Wrapper for Megatron Energon dataloader."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        if dataloader is not None:
            self._iter = iter(cyclic_iter(dataloader))
        else:
            self._iter = iter([])

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def add_qwen35_extra_args(parser):
    """Extra arguments for Qwen3.5 training."""
    group = parser.add_argument_group(title="qwen35 arguments")
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument("--enable-vision", action="store_true", default=True,
                       help="Enable vision encoder for multimodal training (default).")
    group.add_argument("--no-enable-vision", dest="enable_vision", action="store_false",
                       help="Disable vision encoder for text-only LLM training.")
    group.add_argument("--dataloader-save", type=str, default=None)
    group.add_argument("--extra-vocab-size", type=int, default=0)
    group.add_argument("--spatial-merge-size", type=int, default=2)
    group.add_argument("--temporal-patch-size", type=int, default=2)
    group.add_argument("--patch-size", type=int, default=16)
    group.add_argument("--max-padding-length", type=int, default=2048)
    group.add_argument("--enable-variable-seq-lengths", action="store_true", default=False)
    group.add_argument("--vision-root", type=str, default=None)
    group.add_argument("--max-samples-per-sequence", type=int, default=2**31 - 1)
    group.add_argument("--shuffle-buffer-size", type=int, default=0)
    group.add_argument("--vision-ration", type=float, default=0.1)
    group.add_argument("--image-max-pixels", type=int, default=768 * 768)
    group.add_argument("--image-min-pixels", type=int, default=32 * 32)
    group.add_argument("--vision-recompute-activations", action="store_true", default=False)
    group.add_argument("--no-use-system-prompt", dest="use_system_prompt", action="store_false", default=True)
    group.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action="store_true",
        help="Convert Megatron checkpoint to Transformers checkpoint.",
    )
    group.add_argument("--freeze-LM", action="store_true", default=False)
    group.add_argument("--freeze-ViT", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint",
        action="store_true",
        default=False,
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument("--layernorm-zero-centered-gamma", action="store_true", default=True)

    # Vision encoder parameters (varies across models, no default)
    group.add_argument("--vision-num-layers", type=int, default=None)
    group.add_argument("--vision-hidden-size", type=int, default=None)
    group.add_argument("--vision-ffn-hidden-size", type=int, default=None)
    group.add_argument("--vision-num-attention-heads", type=int, default=None)

    # Profiling args required when the FlagScale runner maps YAML profiling fields to CLI.
    group.add_argument("--use-nsys-profiler", action="store_true", dest="profile", default=False)

    return parser


def add_mimo_args(parser):
    """Extra arguments for colocated MIMO training."""
    group = parser.add_argument_group(title="mimo arguments")
    group.add_argument(
        "--vision-tensor-model-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for the vision module (defaults to language TP).",
    )
    group.add_argument(
        "--vision-pipeline-model-parallel-size",
        type=int,
        default=None,
        help="Pipeline parallel size for the vision module (defaults to language PP).",
    )
    group.add_argument(
        "--vision-micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size for the vision module (defaults to language micro batch size).",
    )
    group.add_argument(
        "--mimo-fp32-grad-cache",
        action="store_true",
        default=False,
        help="Accumulate visual microbatch gradients in fp32 inside the MIMO scheduler.",
    )
    return parser


if __name__ == "__main__":
    # Determine vision mode from CLI args before megatron initialization.
    # NOTE: FlagScale's flatten_dict_to_args skips bool=False values in YAML,
    # so to disable vision in YAML, use `no_enable_vision: True` (generates --no-enable-vision).
    import argparse
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--enable-vision", action="store_true", default=True)
    _pre_parser.add_argument("--no-enable-vision", dest="enable_vision", action="store_false")
    _pre_args, _ = _pre_parser.parse_known_args()
    _enable_vision = _pre_args.enable_vision

    args = parse_and_validate_args(
        extra_args_provider=lambda parser: add_mimo_args(add_qwen35_extra_args(parser)),
        args_defaults={'tokenizer_type': 'Qwen2VLTokenizer' if _enable_vision else 'HFTokenizerFS'},
    )
    if args.use_mimo:
        mimo_layout = getattr(args, "mimo_layout", "colocated")
        if mimo_layout == "colocated":
            # Colocated layout: no global-state overrides; the colocated path
            # never touches the num-microbatches calculator.
            pass
        elif mimo_layout == "grid":
            # Non-colocated grid mode: the global parallel state and the
            # num-microbatches calculator run with TP=1/PP=1/DP=1.  The language
            # module's own TP/PP/DP come from --mimo-module-specs and are applied
            # to the language transformer config in model_provider; all module
            # communication uses the per-module grid process groups.
            validate_qwen35_grid_runtime_contract(args)
            if args.mimo_module_specs is None:
                raise ValueError(
                    "--mimo-layout=grid requires --mimo-module-specs, e.g. "
                    "'images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2'."
                )
            if args.ckpt_format != "torch_dist":
                raise ValueError(
                    "--mimo-layout=grid requires --ckpt-format=torch_dist (the "
                    "grid path saves via MCore sharded state dicts)."
                )
            if args.rampup_batch_size is not None:
                raise ValueError(
                    "--mimo-layout=grid does not support rampup_batch_size "
                    "(fail-fast)."
                )
            # Pipeline-allocation overrides are incompatible with the grid path:
            # the language module's PP layout / layer split is computed from
            # --mimo-module-specs and --num-layers (with an uneven first/last
            # stage allocation when non-divisible), so a user-supplied custom PP
            # layout or per-endpoint layer counts would silently conflict.
            if args.pipeline_model_parallel_layout is not None:
                raise ValueError(
                    "--mimo-layout=grid does not support --pipeline-model-parallel-layout: "
                    "the grid path allocates the language pipeline stages itself "
                    "(even or uneven first/last split from --num-layers and the "
                    "language PP in --mimo-module-specs) (fail-fast)."
                )
            if (
                getattr(args, "decoder_first_pipeline_num_layers", None) is not None
                or getattr(args, "decoder_last_pipeline_num_layers", None) is not None
            ):
                raise ValueError(
                    "--mimo-layout=grid does not support "
                    "--decoder-first-pipeline-num-layers / "
                    "--decoder-last-pipeline-num-layers: the grid path computes "
                    "the first/last stage layer counts itself (base+remainder / "
                    "base when the layer count is not divisible by the language "
                    "PP) (fail-fast)."
                )
            if getattr(args, "account_for_embedding_in_pipeline_split", False) or getattr(
                args, "account_for_loss_in_pipeline_split", False
            ):
                raise ValueError(
                    "--mimo-layout=grid does not support "
                    "--account-for-embedding-in-pipeline-split / "
                    "--account-for-loss-in-pipeline-split: MCore's uneven "
                    "pipeline allocation (used when the layer count is not "
                    "divisible by the language PP) is incompatible with "
                    "standalone embedding/loss stages (fail-fast)."
                )
            args.tensor_model_parallel_size = 1
            args.pipeline_model_parallel_size = 1
            args.data_parallel_size = 1
            args.context_parallel_size = 1
            args.expert_model_parallel_size = 1
            # Sequence parallelism cannot survive the forced global TP=1
            # (ModelParallelConfig rejects SP without TP: "Cannot use sequence
            # parallelism without tensor parallelism").  Preserve the user's
            # requested value in the grid-specific arg; model_provider resolves it
            # per module as ``requested && module TP > 1 && module SP-capable``,
            # and the grid SP policy marks BOTH modules SP-incapable for now (the
            # language forward does not shard embeddings and mRoPE freqs stay
            # full-length), so requested SP resolves to False for every accepted
            # layout - with an explicit rank-0 message at model build time.
            args.mimo_sequence_parallel = args.sequence_parallel
            args.sequence_parallel = False
            # The num-microbatches calculator was initialized at parse time with
            # the YAML's global data_parallel_size (e.g. 4 for TP2 on 8 ranks),
            # but grid mode runs the sampler and calculator with DP=1 (module-local
            # DP slicing happens in the forward step).  Reconfigure it to the
            # forced DP=1 before the grid batch contract and the schedule read it:
            # 48/6 yields 2 microbatches at DP=4 but 8 at DP=1 (8 * 6 == 48).
            # The colocated path never touches the calculator.
            reconfigure_grid_num_microbatches_calculator(args)
            print_rank_0(
                "> non-colocated grid MIMO: global parallel state forced to "
                "TP=1/PP=1/DP=1; module layouts from --mimo-module-specs"
            )
        else:
            assert False, (
                f"Unsupported --mimo-layout {mimo_layout!r}: "
                "expected 'colocated' or 'grid'"
            )
    full_config = pretrain_cfg_container_from_args(args)

    if _enable_vision:
        # Multimodal mode: use energon dataloaders
        train_valid_test_dataloaders_provider.is_distributed = True
        pretrain(
            full_config,
            train_valid_test_dataloaders_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            process_non_loss_data_func=write_online_eval_to_tensorboard,
            non_loss_data_func=run_online_eval,
            get_embedding_ranks=get_embedding_ranks,
        )
    else:
        # Text-only mode: use GPT-style dataset (bin/idx)
        train_valid_test_datasets_provider_gpt.is_distributed = True
        pretrain(
            full_config,
            train_valid_test_datasets_provider_gpt,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step_text,
            get_embedding_ranks=get_embedding_ranks,
        )
