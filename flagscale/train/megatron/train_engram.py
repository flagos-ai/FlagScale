# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import torch

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core import mpu
from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from megatron.training.extra_valid import extra_valid_datasets_provider
from megatron.training.training import pretrain
from megatron.plugin.hetero.parallel_context import get_parallel_context

# engram
from flagscale.models.megatron.engram.engram_builder import engram_builder

stimer = StragglerDetector()


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        assert data_iterator is not None
        data = next(data_iterator)
        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': (
                None
                if "attention_mask" not in data
                else data["attention_mask"].cuda(non_blocking=True)
            ),
            'position_ids': data["position_ids"].cuda(non_blocking=True),
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
           _broadcast(batch['tokens'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])
            ######### FlagScale Begin ########
           if mpu.get_dualpipev_pipeline_model_parallel_world_size() is not None:
                _broadcast(batch['loss_mask'])
                _broadcast(batch['labels'])
            ######### FlagScale End ########

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            _broadcast(batch['tokens'])
            if args.mtp_num_layers is not None:
                _broadcast(batch['position_ids'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

        else:
            _broadcast(batch['tokens'])

    else:

        tokens = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (args.micro_batch_size, 1, args.seq_length, args.seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        else:
            attention_mask = None
        position_ids = torch.empty(
            (args.micro_batch_size, args.seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            ######### FlagScale Modify ########
            if mpu.get_dualpipev_pipeline_model_parallel_world_size() is not None:
                _broadcast(loss_mask)
                _broadcast(labels)
            else:
                labels = None
                loss_mask = None

        elif mpu.is_pipeline_last_stage():
            # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
            # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
            # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
            _broadcast(tokens)
            if args.mtp_num_layers is not None:
                _broadcast(position_ids)
            else:
                position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        else:
            _broadcast(tokens)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

    return batch


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    # # TODO: this is pretty hacky, find a better way
    # if not is_first_or_last_pipeline_stage(vp_stage):
    #     return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

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

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
        # print(f"in train_engram, rank: {torch.distributed.get_rank()}, {tokens=}")
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    # return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0
    return parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    ########## FlagScale Begin ##########
    config = None
    para_ctx = get_parallel_context()
    if para_ctx is not None:
        config = para_ctx.get_dataset_config()
    ########## FlagScale End ##########
    if config is None:
        config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    extra_valid_datasets_provider.is_distributed = True ######## FlagScale ########

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, engram_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        extra_valid_dataset_provider=extra_valid_datasets_provider
    )
