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
import time
import logging
from functools import partial
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch._dynamo

from argparse import Namespace

from megatron.core import parallel_state
from megatron.training.checkpointing import get_checkpoint_name
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector

from megatron.training.utils import unwrap_model
from megatron.training.utils import is_first_or_last_pipeline_stage
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

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

from megatron.plugin.platform import get_platform
cur_platform = get_platform()

from tools.datasets.qwenvl.data.dataset_helpers import TaskEncoder, print_error_handler

IGNORE_IDX = -100


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, **kwargs
) -> Union[Qwen35Model]:
    """Provide a Qwen3.5 model instance."""
    args = get_args()
    print_rank_0("start building qwen3.5 model ...")

    # NOTE(vpp): When virtual pipeline parallelism is on, Megatron's get_model()
    # builds one chunk per virtual stage and passes vp_stage=i via kwargs. We must
    # forward it to Qwen35Model so MTP placement (mtp_on_this_rank) and language
    # layer counting land on the correct virtual stage. Defaults to None (VPP off).
    vp_stage = kwargs.get("vp_stage", None)

    # Build transformer config with Qwen35 config class
    config = core_transformer_config_from_args(args, Qwen35TransformerConfig)
    # 20260726: yaml-driven configs go through validate_yaml(), which skips
    # Megatron's validate_args() where virtual_pipeline_model_parallel_size is
    # derived from --num-layers-per-virtual-pipeline-stage; backfill it here.
    if (
        getattr(config, "virtual_pipeline_model_parallel_size", None) is None
        and getattr(args, "num_layers_per_virtual_pipeline_stage", None) is not None
    ):
        num_layers_per_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
        config.virtual_pipeline_model_parallel_size = (
            num_layers_per_stage // args.num_layers_per_virtual_pipeline_stage
        )
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
    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.first_pipeline_num_layers = None
    vision_projector_config = get_vision_projection_config(
        deepcopy(config), vision_config.hidden_size, args.spatial_merge_size
    )

    print_rank_0("building Qwen3.5 model in TE...")

    # Language model spec: hybrid GDN + Attention
    language_layer_spec = get_qwen35_language_model_spec(config, vp_stage=vp_stage)

    # Vision model spec (identical to Qwen3-VL)
    vision_model_spec = get_qwen3vl_vision_model_spec()
    vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules

    if args.enable_variable_seq_lengths:
        config.variable_seq_lengths = True

    # MTP (Multi-Token Prediction) spec
    mtp_block_spec = get_qwen35_mtp_block_spec(args, config, vp_stage=vp_stage)

    args.padded_vocab_size = args.vocab_size
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

        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )

    return model


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
    # Position ids [3 X bs X seqlen]
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


def get_batch(
    data_iterator, model: Qwen35Model = None, vp_stage: int = None
) -> Tuple:
    """Generate a batch."""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # NOTE(vpp): Under virtual pipeline parallelism the interleaved scheduler calls
    # forward_step once per local model chunk. Middle chunks (pre_process=False)
    # receive their input activations via set_input_tensor, so tokens/labels/vision
    # from the batch are unused for them -- but mRoPE position_ids and
    # attention_mask are data-dependent, so middle chunks must still pull data
    # (each chunk has its own data iterator under VPP, kept in sync by the
    # scheduler) and compute them; see the is_middle_vpp_chunk return branch
    # below. When vp_stage is None (VPP off) is_first_or_last_pipeline_stage
    # checks the physical pipeline stage, so plain-PP middle stages (e.g. pp4)
    # take the same middle-chunk path -- unlike train_gpt.py, they still need
    # position_ids because this model's rotary embedding is data-dependent.
    args = get_args()
    config = core_transformer_config_from_args(args, Qwen35TransformerConfig)
    is_middle_vpp_chunk = not is_first_or_last_pipeline_stage(
        vp_stage
    ) and not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    # NOTE(mrope-fix): do NOT early-return all-None for middle chunks without a
    # data iterator. Data loaders are built on TP rank 0 only, so non-TP0 ranks
    # of a middle pipeline stage always have data_iterator=None; they receive the
    # batch via broadcast_data() below (data=None path) and must still compute
    # the data-dependent mRoPE position_ids -- the decoder of a middle chunk
    # needs them for its rotary embedding. Returning all-None fed
    # position_ids=None into the model and crashed in rope.py
    # ('NoneType' object has no attribute 'ndim'), while rank 0 of the same
    # stage blocked in broadcast_data waiting for the TP peers that had
    # already returned. (Hit on plain pp4 without VPP, 2026-08-10.)

    cur_platform.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        # 20260804 data-pipeline probe (FLAGS_PROFILE_DATA=1): time the blocking
        # next() — queue wait for dataloader workers (decode+I/O aggregated).
        # Per-call line lets post-analysis correlate fetch stalls with iteration
        # time spikes (data-dependent shard glitches).
        _data_prof = os.getenv("FLAGS_PROFILE_DATA", "0") == "1"
        _t0 = time.perf_counter() if _data_prof else 0.0
        data = next(data_iterator)
        _fetch_ms = (time.perf_counter() - _t0) * 1e3 if _data_prof else -1.0
        pad_token_id = IGNORE_IDX
        while (data["target"] == pad_token_id).all():
            logging.getLogger(__name__).warning(
                "The current data is invalid because the target is all pad_token_id! "
                "Get next data to avoid fail, but it's better to check the data!"
            )
            data = next(data_iterator)
        if _data_prof:
            print(f"[DATA_FETCH] wait_ms={_fetch_ms:.1f}", flush=True)
    else:
        data = None

    data_text = broadcast_data(["text"], data, torch.int64)["text"]
    target = broadcast_data(["target"], data, torch.int64)["target"]
    imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]
    videos = broadcast_data(["videos"], data, torch.float32)["videos"]
    image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]

    args = get_args()
    if data_text.shape[-1] == args.max_padding_length and get_pipeline_model_parallel_rank() == 0:
        cur_platform.empty_cache()

    video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
    second_per_grid_ts = broadcast_data(['second_per_grid_ts'], data, torch.float32)['second_per_grid_ts']
    image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
    video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
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

    if is_middle_vpp_chunk:
        # mRoPE position_ids are data-dependent: every VPP chunk's decoder needs
        # them for rotary embeddings, so middle chunks must also pull data (each
        # chunk has its own data iterator under VPP, kept in sync by the
        # scheduler) and compute position_ids/attention_mask. Vision tensors and
        # labels are unused when pre_process is False.
        return (
            None,
            None,
            None,
            attention_mask,
            position_ids,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        imgs,
        videos,
        image_thw_grids,
        video_thw_grids,
        image_input_mask,
        video_input_mask,
    )


SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[Qwen35Model] = None,
):
    """Loss function."""
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):
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


def forward_step(data_iterator, model: Qwen35Model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model (Qwen35Model): The Qwen3.5 multimodal model
        return_schedule_plan (bool): Whether to return a schedule plan (for
            overlap_moe_expert_parallel_comm / combined_1f1b) instead of the
            output tensor. Mirrors train_deepseek_v4.forward_step.
    """
    args = get_args()
    timers = get_timers()

    # NOTE(vpp): The interleaved (VPP) scheduler invokes forward_step once per local
    # model chunk and tags each chunk's model with .vp_stage. Extract it so get_batch
    # can decide whether this chunk needs real data. unwrap_model peels DDP/Float16
    # wrappers; vp_stage defaults to None when VPP is off.
    vp_stage = getattr(unwrap_model(model), "vp_stage", None)

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            imgs,
            videos,
            image_thw_grids,
            video_thw_grids,
            image_input_mask,
            video_input_mask,
        ) = get_batch(data_iterator, model=unwrap_model(model), vp_stage=vp_stage)
    timers('batch-generator').stop()

    # Middle VPP chunks (pre_process=False) receive their input activation via
    # set_input_tensor and get None tokens/vision from get_batch. Skip vision
    # assembly and let Qwen35Model.forward take its non-pre_process branch
    # (vision ignored). position_ids/attention_mask are real: the decoder of a
    # middle chunk still needs them for mRoPE and variable-length attention.
    if tokens is None:
        # 20260727: combined_1f1b (EP A2A overlap) requires forward_step to
        # return a schedule plan for EVERY chunk, including middle VPP chunks
        # (pp2+vpp2 means both ranks have middle chunks). Build the plan with
        # input_ids=None; the wrapper delegates to the language model, whose
        # chunk takes its input via set_input_tensor.
        if return_schedule_plan:
            with stimer:
                schedule_plan = model.build_schedule_plan(
                    input_ids=None,
                    position_ids=position_ids,
                    vision_data=None,
                    vision_grid_thw=None,
                    attention_mask=attention_mask,
                    labels=None,
                    loss_mask=loss_mask,
                )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        with stimer:
            output_tensor = model(
                input_ids=None,
                position_ids=position_ids,
                vision_data=None,
                vision_grid_thw=None,
                attention_mask=attention_mask,
                labels=None,
            )
        return output_tensor, partial(loss_func, loss_mask, model=model)

    vision_data = torch.cat([imgs, videos], dim=0)
    vision_grid = torch.cat([image_thw_grids, video_thw_grids], dim=0)

    if return_schedule_plan:
        assert args.overlap_moe_expert_parallel_comm, (
            "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
        )
        with stimer:
            schedule_plan = model.build_schedule_plan(
                input_ids=tokens,
                position_ids=position_ids,
                vision_data=vision_data,
                vision_grid_thw=vision_grid,
                video_start_index=image_input_mask.sum().cpu().item(),
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
        return schedule_plan, partial(loss_func, loss_mask, model=model)

    with stimer:
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            vision_data=vision_data,
            vision_grid_thw=vision_grid,
            video_start_index=image_input_mask.sum().cpu().item(),
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            attention_mask=attention_mask,
            labels=labels,
        )

    return output_tensor, partial(loss_func, loss_mask, model=model)


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


def train_valid_test_dataloaders_provider(train_val_test_num_samples, vp_stage=None):
    """Build multimodal train, validation and test dataloaders.

    NOTE(vpp): vp_stage is required as a kwarg when VPP is on -- pretrain() builds
    one data iterator per virtual stage via functools.partial(provider, vp_stage=i)
    and asserts the signature accepts it. The energon multimodal stream is identical
    across virtual stages (all chunks would see the same samples), so we accept the
    argument but do not use it for data partitioning. Per-chunk data gating happens
    later in get_batch (only first/last/MTP chunks actually pull from the iterator).
    """
    args = get_args()
    if not is_dataloader_rank(args.transformer_pipeline_model_parallel_size):
        return None, None, None

    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

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
        self._iter = iter(cyclic_iter(dataloader))

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

    return parser


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'Qwen2VLTokenizer'},
        extra_args_provider=add_qwen35_extra_args,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        non_loss_data_func=run_online_eval,
    )
