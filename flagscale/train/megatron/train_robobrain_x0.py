# Mainly Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/examples/qwen2_5_vl/pretrain_qwen.py.Below is the original copyright:
# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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


import logging
import math
import os
import sys

from argparse import Namespace
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch._dynamo

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import get_checkpoint_name  # for dataloder
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

# # For pytorch 2.6
# torch.serialization.add_safe_globals([Namespace])


import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from megatron.training.training import pretrain

stimer = StragglerDetector()

#### especially for qwen2.5-vl ####
from megatron.core.num_microbatches_calculator import get_num_microbatches

torch._dynamo.config.suppress_errors = True
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
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
from megatron.training.global_vars import get_tokenizer
from megatron.training.tokenizer.tokenizer import build_tokenizer

# from tools.datasets.qwenvl.data.dataset_helpers_action import TaskEncoder, print_error_handler
# from tools.datasets.qwenvl.data.dataset_helpers_action_unified_plus_sub import TaskEncoder, print_error_handler
from tools.datasets.vla.data.dataset_helpers_vlm import TaskEncoder, print_error_handler

from flagscale.models.megatron.qwen2_5_vl.layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_mlp_module_spec,
    get_qwen2vl_vision_model_spec,
)
from flagscale.models.megatron.qwen2_5_vl.qwen2_5_vl_model import Qwen2_5VLModel
from flagscale.models.megatron.qwen2_5_vl.tensor_parallel import broadcast_data
from flagscale.models.megatron.qwen2_5_vl.transformer_config import (
    get_vision_model_config,
    get_vision_projection_config,
)

# LeRobotDataset support
from flagscale.train.datasets.lerobot_dataset import LeRobotDataset

#### especially for qwen2.5-vl ####
IGNORE_IDX = -100
FIRST_MAX_PADDING_FLAG = True
LAST_LARGE_IMG = False


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True
) -> Union[Qwen2_5VLModel]:
    args = get_args()
    build_tokenizer(args)
    print_rank_0("start building qwen2-vl model ...")

    # Config of vit, llm and projector
    config = core_transformer_config_from_args(args)
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("The Qwen2-VL model is only implemented with TransformerEngine!")

    if (
        args.rotary_seq_len_interpolation_factor is not None
        or args.rotary_seq_len_interpolation_factor != 1
    ):
        print_rank_0("Multimodal RoPE currently not support RoPE interpolation, set to None...")
        args.rotary_seq_len_interpolation_factor = None

    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.first_pipeline_num_layers = None
    vision_projector_config = get_vision_projection_config(
        deepcopy(config), vision_config.hidden_size, args.spatial_merge_size
    )

    print_rank_0("building Qwen2-5-VL model in TE...")
    # Layer Specs of vit, llm and projector
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.qk_layernorm)
    vision_model_spec = get_qwen2vl_vision_model_spec()
    vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules
    if args.enable_variable_seq_lengths:
        config.variable_seq_lengths = True

    model = Qwen2_5VLModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        drop_vision_class_token=False,  # NOTE: no class token to drop?
        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec,
        vision_projection_type="mlp",
        allow_missing_vision_projection_checkpoint=args.allow_missing_vision_projection_checkpoint,
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
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )
    print_rank_0("=" * 50)
    print_rank_0("Model Embedding Information:")
    print_rank_0("=" * 50)

    try:
        if hasattr(model, "language_model") and hasattr(model.language_model, "embedding"):
            lang_embedding = model.language_model.embedding
            print_rank_0(
                f"Language model embedding shape: {lang_embedding.word_embeddings.weight.shape}"
            )
            print_rank_0(f"Language vocab size: {args.padded_vocab_size}")
            print_rank_0(f"Language hidden size: {config.hidden_size}")
            print(f"Config vocab_size: {args.padded_vocab_size}")
            print(
                f"Actual embedding allocated: {model.language_model.embedding.word_embeddings.weight.shape[0]}"
            )
        if hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings"):
            vision_embedding = model.vision_model.embeddings
            print_rank_0(f"Vision model embedding type: {type(vision_embedding)}")
            if hasattr(vision_embedding, "patch_embedding"):
                print_rank_0(
                    f"Vision patch embedding shape: {vision_embedding.patch_embedding.weight.shape}"
                )
        if hasattr(model, "language_model") and hasattr(model.language_model, "rotary_pos_emb"):
            print_rank_0(f"Language model uses rotary position embedding")
        elif hasattr(model, "language_model") and hasattr(
            model.language_model.embedding, "position_embeddings"
        ):
            pos_emb = model.language_model.embedding.position_embeddings
            print_rank_0(f"Language position embedding shape: {pos_emb.weight.shape}")

    except Exception as e:
        print_rank_0(f"Error accessing embedding layers: {e}")

    return model


# copy from https://github.com/huggingface/transformers/blob/40a493c7ed4f19f08eadb0639cf26d49bfa5e180/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1404
def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    args = get_args()
    tokenizer = get_tokenizer()
    spatial_merge_size = args.spatial_merge_size
    image_token_id = tokenizer.image_token_id
    video_token_id = tokenizer.video_token_id
    vision_start_token_id = tokenizer.vision_start_token_id
    tokens_per_second = 2
    if second_per_grid_ts is not None:
        second_per_grid_ts = second_per_grid_ts.cpu()

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype
            )

        return position_ids, mrope_position_deltas


def get_ltor_masks_and_position_ids(
    input_ids,
    image_thw_grids,
    video_thw_grids,
    target,
    pad_token,
    second_per_grid_ts,
    ignore_index=None,
):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    position_ids, _ = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=input_ids != pad_token,
    )

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    # Attention mask.
    attention_mask = None

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        # pad_token_id = get_tokenizer().pad_token_id
        pad_token_id = IGNORE_IDX
        # while (data["target"] == pad_token_id).all() or (data["target"].shape[-1] < 986 or data["target"].shape[-1] > 1000): # for debug
        while (data["target"] == pad_token_id).all():
            logging.getLogger(__name__).warning(
                "The current data is invalid because the target is all pad_token_id! Get next data to avoid fail, but it's better to check the data!"
            )
            data = next(data_iterator)
    else:
        data = None

    data_text = broadcast_data(["text"], data, torch.int64)["text"]

    target = broadcast_data(["target"], data, torch.int64)["target"]
    # shape: num_tiles x c x h x w
    imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]

    # shape: num_tiles x c x h x w
    videos = broadcast_data(["videos"], data, torch.float32)["videos"]

    # shape: n_image_samples
    image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]

    # global LAST_LARGE_IMG
    # if LAST_LARGE_IMG:
    #     torch.cuda.empty_cache()
    #     LAST_LARGE_IMG=False
    # if image_thw_grids.prod(axis=-1).sum() // 4 > 3000:
    #     torch.cuda.empty_cache()
    #     LAST_LARGE_IMG = True
    args = get_args()
    if data_text.shape[-1] == args.max_padding_length and get_pipeline_model_parallel_rank() == 0:
        torch.cuda.empty_cache()
    # shape: n_video_samples
    video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
    # shape: n_video_samples
    second_per_grid_ts = broadcast_data(["second_per_grid_ts"], data, torch.float32)[
        "second_per_grid_ts"
    ]

    image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
    video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    # NOTE: no sequence packing in LLM inputs
    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, image_thw_grids, video_thw_grids, labels, IGNORE_IDX, second_per_grid_ts
    )
    torch.cuda.nvtx.range_pop()

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


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[Qwen2_5VLModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (Qwen2_5VLModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {"lm loss": reporting_loss})


def forward_step(data_iterator, model: Qwen2_5VLModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
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
        ) = get_batch(data_iterator)
    timers("batch-generator").stop()
    # print(f"LZY imags: {imgs.shape}, content: {imgs.sum()}, {imgs}")
    vision_data = torch.cat([imgs, videos], dim=0)
    vision_grid = torch.cat([image_thw_grids, video_thw_grids], dim=0)
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
    """Run an evaluation benchmark during training."""
    # Do nothing.
    return []


def write_online_eval_to_tensorboard(data, iteration, writer):
    """Write online evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """
    A wrapper to convert LeRobotDataset samples to the format expected by RoboBrain-X0.

    This wrapper handles:
    - Loading images/videos from LeRobotDataset
    - Converting observation and action data to the expected format
    - Building conversation tokens with action tokens
    """

    ACTION_TOKEN_START_ID = 149595
    ACTION_TOKEN_END_ID = ACTION_TOKEN_START_ID + 2048

    def __init__(self, lerobot_dataset: LeRobotDataset, args):
        self.dataset = lerobot_dataset
        self.args = args
        self.tokenizer = get_tokenizer()
        self.seq_len = args.max_padding_length
        self.temporal_patch_size = args.temporal_patch_size
        self.merge_size = args.spatial_merge_size
        self.patch_size = args.patch_size

        # Build token cache
        self._token_cache = self._build_token_cache()
        self._action_token_cache = self._build_action_token_cache()

        # Get action dimension from dataset features
        self.action_dim = None
        for key, feat in self.dataset.meta.features.items():
            if key == "action":
                self.action_dim = feat.get("shape", [7])[0]
                break
        if self.action_dim is None:
            self.action_dim = 7  # Default action dimension

    def _build_token_cache(self):
        return {
            "im_start": self.tokenizer.vocab["<|im_start|>"],
            "im_end": self.tokenizer.vocab["<|im_end|>"],
            "user": self.tokenizer.vocab["user"],
            "assistant": self.tokenizer.vocab["assistant"],
            "system": self.tokenizer.vocab["system"],
            "vision_start": self.tokenizer.vocab.get("<|vision_start|>"),
            "vision_end": self.tokenizer.vocab.get("<|vision_end|>"),
            "image_pad": self.tokenizer.vocab.get("<|image_pad|>"),
            "video_pad": self.tokenizer.vocab.get("<|video_pad|>"),
            "newline": self._safe_encode("\n")[0],
            "space": self._safe_encode(" ")[0],
            "boa": self.tokenizer.vocab.get("<boa>", 151665),
            "EOA": self.tokenizer.vocab.get("<EOA>", 151666),
            "action_split": self.tokenizer.vocab.get("<action_split>", 151667),
        }

    def _build_action_token_cache(self):
        action_cache = {}
        for action_id in range(2048):
            token_string = f"<action_token_{action_id}>"
            token_id = self.tokenizer.vocab.get(
                token_string, self.ACTION_TOKEN_START_ID + action_id
            )
            if token_id is not None:
                action_cache[action_id] = token_id
        return action_cache

    def _safe_encode(self, text):
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def _discretize_action(self, action: np.ndarray, num_bins: int = 2048) -> list:
        """Discretize continuous action values to token indices."""
        # Normalize action to [0, 1] range (assuming action is in [-1, 1])
        normalized = (action + 1) / 2
        # Clip to valid range
        normalized = np.clip(normalized, 0, 1)
        # Convert to bin indices
        bin_indices = (normalized * (num_bins - 1)).astype(int)
        return bin_indices.tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get sample from LeRobotDataset
        sample = self.dataset[idx]

        # Process images from camera keys
        imgs = []
        image_thw_grids = []

        for cam_key in self.dataset.meta.camera_keys:
            if cam_key in sample:
                img_tensor = sample[cam_key]
                # img_tensor shape: (C, H, W) or (T, C, H, W) for video
                if img_tensor.dim() == 3:
                    # Single image
                    img_np = img_tensor.numpy()
                    # Process through image processor
                    from PIL import Image

                    img_pil = Image.fromarray((img_np.transpose(1, 2, 0) * 255).astype(np.uint8))
                    imgs_info = self.tokenizer.processor.image_processor(
                        [img_pil], return_tensors="pt"
                    )
                    flattened_imgs = imgs_info["pixel_values"]
                    grid_thw = imgs_info["image_grid_thw"]
                    imgs.append(flattened_imgs)
                    image_thw_grids.extend(grid_thw.tolist())

        if len(imgs) > 0:
            imgs = np.concatenate(imgs, axis=0)
            image_thw_grids = np.array(image_thw_grids, dtype=np.int64)
        else:
            imgs = np.empty(
                [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size],
                dtype=np.float32,
            )
            image_thw_grids = np.empty([0, 3], dtype=np.int64)

        # Process action
        action = sample.get("action", None)
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            action_tokens = self._discretize_action(action)
        else:
            action_tokens = []

        # Get task description
        task = sample.get("task", "Execute the robot manipulation task.")

        # Build conversation
        num_images = len(image_thw_grids)
        image_placeholder = "<image>" * num_images

        conversation = [
            {"role": "system", "content": "You are a helpful robot assistant."},
            {
                "role": "user",
                "content": [{"type": "image", "image": "0"} for _ in range(num_images)]
                + [{"type": "text", "text": task}],
            },
            {"role": "assistant", "content": ""},
        ]

        # Build input tokens
        input_ids = self._build_conversation_tokens(conversation, [[], [], action_tokens])

        # Build target (shifted input_ids with masking)
        target = input_ids.copy()
        target = np.roll(target, shift=-1)
        target[-1] = IGNORE_IDX

        # Mask system and user turns
        system_end = self._find_turn_end(input_ids, 0)
        user_end = self._find_turn_end(input_ids, system_end)
        target[: user_end + 3] = IGNORE_IDX  # Mask up to assistant prefix

        # Expand image placeholders
        merge_length = self.merge_size**2
        image_token_id = self.tokenizer.image_token_id

        image_token_indices = np.where(input_ids == image_token_id)[0]

        target_length = (
            input_ids.shape[0]
            - len(image_token_indices)
            + image_thw_grids.prod(axis=-1).sum() // merge_length
            if len(image_thw_grids) > 0
            else input_ids.shape[0]
        )

        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_target = np.full(target_length, IGNORE_IDX, dtype=target.dtype)

        if len(image_token_indices) > 0:
            cur_x, cur_y, image_idx = 0, 0, 0
            for idx in image_token_indices:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
                final_input_ids[cur_y : cur_y + idx - cur_x] = input_ids[cur_x:idx]
                final_target[cur_y : cur_y + idx - cur_x] = target[cur_x:idx]
                cur_y += idx - cur_x
                final_input_ids[cur_y : cur_y + size] = image_token_id
                final_target[cur_y : cur_y + size] = IGNORE_IDX
                cur_y += size
                cur_x = idx + 1
            if cur_x < len(input_ids):
                final_input_ids[cur_y:] = input_ids[cur_x:]
                final_target[cur_y:] = target[cur_x:]
        else:
            final_input_ids = input_ids
            final_target = target

        image_input_mask = final_input_ids == self.tokenizer.image_token_id
        video_input_mask = np.zeros_like(final_input_ids, dtype=bool)

        return {
            "text": final_input_ids,
            "target": final_target,
            "imgs": imgs,
            "videos": np.empty(
                [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size],
                dtype=np.float32,
            ),
            "image_thw_grids": image_thw_grids,
            "video_thw_grids": np.empty([0, 3], dtype=np.int64),
            "image_input_mask": image_input_mask,
            "video_input_mask": video_input_mask,
            "second_per_grid_ts": np.zeros(0, dtype=np.float32),
        }

    def _build_conversation_tokens(self, conversation, action_tokens_list):
        """Build token sequence from conversation."""
        final_token_ids = []

        im_start_id = self._token_cache["im_start"]
        im_end_id = self._token_cache["im_end"]
        newline_id = self._token_cache["newline"]
        user_id = self._token_cache["user"]
        assistant_id = self._token_cache["assistant"]
        system_id = self._token_cache["system"]
        image_pad_id = self._token_cache["image_pad"]
        vision_start_id = self._token_cache["vision_start"]
        vision_end_id = self._token_cache.get("vision_end")

        for turn_idx, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            action_tokens = (
                action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            )

            final_token_ids.append(im_start_id)

            if role == "system":
                final_token_ids.append(system_id)
                final_token_ids.append(newline_id)
                if content.strip():
                    text_ids = self._safe_encode(content)
                    final_token_ids.extend(text_ids)

            elif role == "user":
                final_token_ids.append(user_id)
                final_token_ids.append(newline_id)
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            if item["text"].strip():
                                text_ids = self._safe_encode(item["text"])
                                final_token_ids.extend(text_ids)
                        elif item["type"] == "image":
                            if vision_start_id:
                                final_token_ids.append(vision_start_id)
                            final_token_ids.append(image_pad_id)
                            if vision_end_id:
                                final_token_ids.append(vision_end_id)
                else:
                    if content.strip():
                        text_ids = self._safe_encode(content)
                        final_token_ids.extend(text_ids)

            elif role == "assistant":
                final_token_ids.append(assistant_id)
                final_token_ids.append(newline_id)
                if content.strip():
                    text_ids = self._safe_encode(content)
                    final_token_ids.extend(text_ids)
                if action_tokens and len(action_tokens) > 0:
                    boa_id = self._token_cache["boa"]
                    for action_id in action_tokens:
                        correct_token_id = self._action_token_cache.get(action_id)
                        if correct_token_id is not None:
                            final_token_ids.append(correct_token_id)

            final_token_ids.append(im_end_id)
            final_token_ids.append(newline_id)

        return np.array(final_token_ids, dtype=np.int64)

    def _find_turn_end(self, input_ids, start_idx):
        """Find the end index of a conversation turn."""
        im_end_id = self._token_cache["im_end"]
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] == im_end_id:
                return i
        return len(input_ids) - 1


def lerobot_collate_fn(samples):
    """Collate function for LeRobotDatasetWrapper samples."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Get max sequence length
    max_seq_len = args.max_padding_length
    if args.enable_variable_seq_lengths:
        max_seq_len = max(len(s["text"]) for s in samples)
        max_seq_len = min(max_seq_len, args.max_padding_length)

    tp_size = args.tensor_model_parallel_size
    cp_size = args.context_parallel_size
    if cp_size > 1 or args.sequence_parallel:
        max_seq_len = math.ceil(max_seq_len / (tp_size * cp_size)) * (tp_size * cp_size)

    batch_size = len(samples)

    # Initialize tensors
    text_mat = np.full((batch_size, max_seq_len), tokenizer.pad_token_id, dtype=np.int64)
    target_mat = np.full((batch_size, max_seq_len), IGNORE_IDX, dtype=np.int64)
    image_input_masks = np.zeros((batch_size, max_seq_len), dtype=bool)
    video_input_masks = np.zeros((batch_size, max_seq_len), dtype=bool)

    # Collect all images and grids
    all_imgs = []
    all_videos = []
    all_image_grids = []
    all_video_grids = []
    all_second_per_grid_ts = []

    for i, s in enumerate(samples):
        text_len = min(max_seq_len, len(s["text"]))
        target_len = min(max_seq_len, len(s["target"]))

        text_mat[i, :text_len] = s["text"][:text_len]
        target_mat[i, :target_len] = s["target"][:target_len]

        if s["image_input_mask"] is not None:
            image_input_masks[i, :text_len] = s["image_input_mask"][:text_len]
        if s["video_input_mask"] is not None:
            video_input_masks[i, :text_len] = s["video_input_mask"][:text_len]

        if len(s["imgs"]) > 0:
            all_imgs.append(s["imgs"])
        if len(s["videos"]) > 0:
            all_videos.append(s["videos"])
        if len(s["image_thw_grids"]) > 0:
            all_image_grids.extend(s["image_thw_grids"].tolist())
        if len(s["video_thw_grids"]) > 0:
            all_video_grids.extend(s["video_thw_grids"].tolist())
        if len(s["second_per_grid_ts"]) > 0:
            all_second_per_grid_ts.extend(s["second_per_grid_ts"].tolist())

    # Concatenate images and videos
    if len(all_imgs) > 0:
        imgs = torch.from_numpy(np.concatenate(all_imgs, axis=0))
    else:
        temporal_patch_size = args.temporal_patch_size
        patch_size = args.patch_size
        imgs = torch.empty(
            [0, 3 * temporal_patch_size * patch_size * patch_size], dtype=torch.float32
        )

    if len(all_videos) > 0:
        videos = torch.from_numpy(np.concatenate(all_videos, axis=0))
    else:
        temporal_patch_size = args.temporal_patch_size
        patch_size = args.patch_size
        videos = torch.empty(
            [0, 3 * temporal_patch_size * patch_size * patch_size], dtype=torch.float32
        )

    if len(all_image_grids) > 0:
        image_thw_grids = torch.from_numpy(np.array(all_image_grids)).long()
    else:
        image_thw_grids = torch.empty([0, 3], dtype=torch.long)

    if len(all_video_grids) > 0:
        video_thw_grids = torch.from_numpy(np.array(all_video_grids)).long()
    else:
        video_thw_grids = torch.empty([0, 3], dtype=torch.long)

    if len(all_second_per_grid_ts) > 0:
        second_per_grid_ts = torch.from_numpy(np.array(all_second_per_grid_ts)).float()
    else:
        second_per_grid_ts = torch.empty([0], dtype=torch.float32)

    return {
        "text": torch.from_numpy(text_mat),
        "target": torch.from_numpy(target_mat),
        "imgs": imgs,
        "videos": videos,
        "image_thw_grids": image_thw_grids,
        "video_thw_grids": video_thw_grids,
        "image_input_mask": torch.from_numpy(image_input_masks),
        "video_input_mask": torch.from_numpy(video_input_masks),
        "second_per_grid_ts": second_per_grid_ts,
    }


def lerobot_datasets_provider():
    """Create train, validation and test datasets from LeRobotDataset format."""
    args = get_args()
    data_path = args.data_path[0] if isinstance(args.data_path, list) else args.data_path

    # Load LeRobotDataset
    train_dataset = LeRobotDataset(
        root=data_path,
        episodes=None,
        revision=None,
        video_backend=getattr(args, "video_backend", "pyav"),
    )

    # Wrap with our converter
    wrapped_train_dataset = LeRobotDatasetWrapper(train_dataset, args)

    # For now, we don't support separate validation/test datasets from LeRobot format
    # Users should prepare separate datasets if needed
    val_datasets = None
    test_dataset = None

    return wrapped_train_dataset, val_datasets, test_dataset


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
        max_samples_per_sequence=args.max_samples_per_sequence,  # sequential shuffle in a tar
        shuffle_buffer_size=args.shuffle_buffer_size,  # shuffle in a sequential
        handler=print_error_handler,
        repeat=True,
        image_decode="pil",
    )
    val_datasets_without_source_datasets = None
    if args.eval_iters > 0:
        val_datasets = get_val_datasets(
            dname,
            batch_size=args.micro_batch_size,
            # This is the total number over all workers
            # limit=args.eval_iters * get_num_microbatches(),
            task_encoder=TaskEncoder(),
            worker_config=worker_config,
            handler=print_error_handler,
            image_decode="pil",
        )
        val_datasets_without_source_datasets = [
            # Limit the dataset to eval_iters * num_microbatches
            LimitDataset(
                # Repeat the inner dataset in case it's too short
                RepeatDataset(val_ds, worker_config=worker_config),
                length=args.eval_iters * get_num_microbatches(),
                worker_config=worker_config,
                reset_after_epoch=True,
            )
            for val_ds, _src_ds in val_datasets
        ]

    return train_dataset, val_datasets_without_source_datasets, None


def is_first_or_last_stage(pp_size, encoder_pipeline_model_parallel_size):
    """Check if the current pipeline parallel stage is the first or last stage."""
    if pp_size == 1:  # No pipeline parallelism.
        return True

    is_valid_rank = False
    pp_rank = get_pipeline_model_parallel_rank()
    if encoder_pipeline_model_parallel_size == 0:
        # No separate pipeline stage for the vision model. Run the dataloader on the first and last pipeline stage.
        is_valid_rank = pp_rank in (0, pp_size - 1)
    elif encoder_pipeline_model_parallel_size == 1:
        # Separate pipeline stage for the vision model. Run the dataloader on the first vision and LM stage and last LM stage.
        is_valid_rank = pp_rank in (0, 1, pp_size - 1)
    else:
        raise NotImplementedError("encoder-pipeline-model-parallel-size > 1 is not supported yet")

    return is_valid_rank


def is_dataloader_rank(encoder_pipeline_model_parallel_size):
    """Check if we should have the dataloader on this tensor and pipeline parallel rank."""
    # Run dataloader only on the first tensor parallel rank (will be broadcasted to others).
    is_first_rank = get_tensor_model_parallel_rank() == 0

    # NOTE(lizhiyu): when pp_size > 2
    # pp_size = get_pipeline_model_parallel_world_size()
    # is_first_rank = is_first_rank and is_first_or_last_stage(pp_size, encoder_pipeline_model_parallel_size)

    return is_first_rank


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
    # Dataloader is only on specific ranks.
    if not is_dataloader_rank(args.transformer_pipeline_model_parallel_size):
        return None, None, None

    # Check dataset type
    dataset_type = getattr(args, "dataset_type", "energon")

    if dataset_type == "lerobot":
        # Use LeRobotDataset format
        return _build_lerobot_dataloaders(args)
    else:
        # Use default Energon/WebDataset format
        return _build_energon_dataloaders(args)


def _build_lerobot_dataloaders(args):
    """Build dataloaders for LeRobotDataset format."""
    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()

    train_ds, valid_ds, test_ds = lerobot_datasets_provider()

    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lerobot_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Wrap with LeRobotDataloader for compatibility
    wrapped_train = LeRobotDataloader(train_dataloader, train_sampler)
    wrapped_valid = LeRobotDataloader(None) if valid_ds is None else None
    wrapped_test = LeRobotDataloader(None)

    return wrapped_train, wrapped_valid, wrapped_test


def _build_energon_dataloaders(args):
    """Build dataloaders for Energon/WebDataset format."""
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
                pipeline_rank=0,  # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(
                        data_save_name, map_location="cpu", weights_only=False
                    )
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
    test_dataloader = None  # NOTE: no test

    return EnergonDataloader(train_dataloader), valid_dataloader, EnergonDataloader(test_dataloader)


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


class LeRobotDataloader:
    """A wrapper to use LeRobotDataset dataloader with the Megatron-LM training loop."""

    def __init__(self, dataloader, sampler=None):
        self._dataloader = dataloader
        self._sampler = sampler
        self._epoch = 0
        if dataloader is not None:
            self._iter = iter(self._cyclic_iter())
        else:
            self._iter = iter([])

    def _cyclic_iter(self):
        """Create a cyclic iterator that properly handles epoch changes."""
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(self._epoch)
            for x in self._dataloader:
                yield x
            self._epoch += 1

    def __next__(self):
        return next(self._iter)

    def __iter__(self):
        return self

    def save_state(self):
        """Return current state for checkpointing."""
        return {"epoch": self._epoch}

    def restore_state(self, state_dict):
        """Restore state from checkpoint."""
        if state_dict is not None and "epoch" in state_dict:
            self._epoch = state_dict["epoch"]


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title="multimodal arguments")
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        default=False,
        help="Disable vision class token",
    )
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )

    # qwen2-vl specific arguments
    group.add_argument("--extra-vocab-size", type=int, default=421)
    group.add_argument("--spatial-merge-size", type=int, default=2)
    group.add_argument("--temporal-patch-size", type=int, default=2)
    group.add_argument("--patch-size", type=int, default=14)
    group.add_argument("--max-padding-length", type=int, default=2048)
    group.add_argument(
        "--enable-variable-seq-lengths",
        action="store_true",
        default=False,
        help="Enable variable sequence lengths",
    )
    group.add_argument(
        "--vision-root", type=str, default=None, help="The vision dirctory root path."
    )
    group.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=2**31 - 1,
        help="max sequencial seqence samples in a slice",
    )
    group.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=0,
        help="the buffer size to shuffle the samples in a seqence",
    )
    # learning rate
    group.add_argument(
        "--vision-ration",
        type=float,
        default=0.1,
        help="the learning rate ration of vision(inlude merger) compared with llm",
    )
    group.add_argument(
        "--image-max-pixels",
        type=int,
        default=768 * 768,
        help="the maximum pixels of a single image",
    )
    group.add_argument(
        "--image-min-pixels", type=int, default=32 * 32, help="the minimum pixels of a single image"
    )
    group.add_argument(
        "--vision-recompute-layer-steps",
        type=int,
        default=0,
        help="the recmoute layers for vision using uniform method. 0 is disable.",
    )

    # just for checkpoint conversion
    group.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    group.add_argument(
        "--freeze-LM", action="store_true", default=False, help="Freeze the language model"
    )
    group.add_argument(
        "--freeze-ViT", action="store_true", default=False, help="Freeze the vision model"
    )
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint",
        action="store_true",
        default=False,
        help="Allow missing vision projection checkpoint",
    )
    group.add_argument(
        "--use-te", action="store_true", default=False, help="Use transformer engine"
    )

    # LeRobotDataset support
    group.add_argument(
        "--dataset-type",
        type=str,
        default="energon",
        choices=["energon", "lerobot"],
        help="Dataset format type: 'energon' for WebDataset/Energon format, 'lerobot' for LeRobotDataset format (default: energon)",
    )
    group.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec", "video_reader"],
        help="Video decoding backend for LeRobotDataset (default: pyav)",
    )
    group.add_argument(
        "--action-discretization-bins",
        type=int,
        default=2048,
        help="Number of bins for action discretization in LeRobotDataset (default: 2048)",
    )
    return parser


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "Qwen2VLTokenizer"},
        extra_args_provider=add_multimodal_extra_args,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        non_loss_data_func=run_online_eval,
    )
