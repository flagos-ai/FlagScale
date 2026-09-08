# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Batch helpers for models that consume token ids on every pipeline stage."""

import torch

from megatron.core import mpu
from megatron.core.parallel_state import get_context_parallel_group, get_hybrid_data_context_parallel_groups
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank as mtp_on_this_rank_func
from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args


BATCH_KEYS = ["attention_mask", "cu_seqlens", "cu_seqlens_padded", "hybrid_cp_group", "labels", "local_cp_size", "loss_mask", "max_seqlen", "position_ids", "tokens"]


def get_batch_for_model_with_all_stage_tokens(data_iterator, vp_stage=None, dualpipev_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)
    tp_rank = mpu.get_tensor_model_parallel_rank()
    mtp_on_this_rank = mtp_on_this_rank_func(layout=config.pipeline_model_parallel_layout, mtp_num_layers=config.mtp_num_layers, ignore_virtual=False, vp_stage=vp_stage, ignore_dualpipev=False, dualpipev_stage=dualpipev_stage)
    is_first_stage = mpu.is_pipeline_first_stage()
    is_last_stage = mpu.is_pipeline_last_stage()
    is_dualpipev = mpu.get_dualpipev_pipeline_model_parallel_world_size() is not None

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if tp_rank == 0:
        batch = next(data_iterator)
        for key in BATCH_KEYS:
            batch[key] = batch[key].cuda(non_blocking=True) if key in batch and batch[key] is not None else None
    else:
        shape = (args.micro_batch_size, args.seq_length)
        batch = {key: None for key in BATCH_KEYS}
        batch["tokens"] = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        batch["labels"] = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        batch["loss_mask"] = torch.empty(shape, dtype=torch.float32, device=torch.cuda.current_device())
        batch["position_ids"] = torch.empty(shape, dtype=torch.int64, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            batch["attention_mask"] = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool, device=torch.cuda.current_device())

    _broadcast(batch["tokens"])
    if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
        _broadcast(batch["labels"])
        _broadcast(batch["loss_mask"])
        _broadcast(batch["position_ids"])
        _broadcast(batch["attention_mask"])
    elif is_first_stage:
        if is_dualpipev:
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
        else:
            batch["labels"] = None
            batch["loss_mask"] = None
        _broadcast(batch["position_ids"])
        _broadcast(batch["attention_mask"])
    elif is_last_stage:
        _broadcast(batch["labels"])
        _broadcast(batch["loss_mask"])
        _broadcast(batch["attention_mask"])
        batch["position_ids"] = None
    else:
        batch["labels"] = None
        batch["loss_mask"] = None
        batch["position_ids"] = None
        batch["attention_mask"] = None

    batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=args.hybrid_context_parallel, cp_group=get_context_parallel_group(), hybrid_cp_group_func=get_hybrid_data_context_parallel_groups)
    return [batch[key] for key in BATCH_KEYS]
