# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import datetime
import os
import torch

from functools import partial
from typing import List, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
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
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from flagscale.train.extra_valid import extra_valid_datasets_provider
from flagscale.train.train import pretrain
from flagscale.train.global_vars import get_parallel_context
from nemo.collections.llm.gpt.data.core import create_sft_dataset
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler

import os


stimer = StragglerDetector()


def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None, is_dualpipev_first_chunk: Optional[bool] = False,
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

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
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    config = None
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        para_ctx = get_parallel_context()
        if para_ctx is not None:
            config = para_ctx.get_transformer_config()

        if config is None:
            config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage, is_dualpipev_first_chunk=is_dualpipev_first_chunk,
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        qk_l2_norm=args.qk_l2_norm,
                        use_kitchen=config.use_kitchen,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization,
                        use_kitchen=config.use_kitchen,
                    )
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=use_te, vp_stage=vp_stage
            )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            rope_scaling_factor=args.rope_scaling_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


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

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

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
            if args.mtp_num_layers is not None:
                _broadcast(batch['tokens'])
                _broadcast(batch['position_ids'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

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
            dtype=torch.int64,
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
            if args.mtp_num_layers is not None:
                _broadcast(tokens)
                _broadcast(position_ids)
            else:
                tokens = None
                position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

    return batch


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None


    # get batches based on the TP rank you are on
    # tokens, lanbels, loss_mask, position_ids, attention_mask, token_count
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

    cp_size = mpu.get_context_parallel_world_size()
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    if cp_size > 1:
        loss_all = torch.cat([losses, loss_mask])
        torch.distributed.all_reduce(loss_all, group=mpu.get_context_parallel_group())
        losses, loss_mask = torch.split(loss_all, losses.shape[0], dim=0)

    num_valid_tokens = torch.sum(loss_mask).int()
    loss = torch.sum(losses * loss_mask) / num_valid_tokens
    loss = torch.where(num_valid_tokens == 0, torch.zeros_like(loss), loss)

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
    reporting_loss = loss.clone().detach().view(1)
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    reporting_loss = reporting_loss / torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    return (loss, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = None, None, None
    if is_dataset_built_on_rank():
        train_data_path = os.path.join(args.data_path[0], "train.npy")
        validation_data_path = os.path.join(args.data_path[0], "validation.npy")
        tokenizer = get_tokenizer()
        tokenizer.eos_id = tokenizer.eod_id
        kwargs = {
            'return_cu_seqlen': False
        }
        train_ds = create_sft_dataset(
            train_data_path,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            memmap_workers=1,
            seed=args.seed,
            is_test=False,
            pack_metadata_file_path=None,
            pad_cu_seqlens=False,
            max_num_samples=train_val_test_num_samples[0],
            **kwargs,
        )
        train_batch_sampler = MegatronPretrainingSampler(
            total_samples=len(train_ds),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_ds,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=train_ds.collate_fn,
            batch_sampler=train_batch_sampler
        )

        valid_ds = create_sft_dataset(
            validation_data_path,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            memmap_workers=1,
            seed=args.seed,
            is_test=False,
            pack_metadata_file_path=None,
            pad_cu_seqlens=False,
            max_num_samples=train_val_test_num_samples[1],
            **kwargs,
        )
        valid_batch_sampler = MegatronPretrainingSampler(
            total_samples=len(train_ds),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            dataset=valid_ds,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=train_ds.collate_fn,
            batch_sampler=valid_batch_sampler
        )

        print_rank_0("> finished creating GPT datasets ...")

        return iter(train_dataloader), iter(valid_dataloader), test_ds
    return train_ds, valid_ds, test_ds

# python run.py --config-path examples/llama3/conf --config-name train_nemo_llama action=run 
if __name__ == "__main__":
    torch.backends.cudnn.deterministic=True
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    extra_valid_datasets_provider.is_distributed = True ######## FlagScale ########
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        extra_valid_dataset_provider=extra_valid_datasets_provider
    )
