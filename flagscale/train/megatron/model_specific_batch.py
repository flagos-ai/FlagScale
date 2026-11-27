# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Batch helpers for models that consume token ids on every pipeline stage."""

import torch

from megatron.core import mpu
from megatron.core.parallel_state import get_context_parallel_group, get_hybrid_data_context_parallel_groups
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank as mtp_on_this_rank_func
from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.plugin.platform import get_platform


def sft_loss_scale_is_non_binary(args):
    """Whether the configured ms-swift loss_scale emits per-token weights.

    The dataset is only built on tp_rank 0, so the other ranks cannot inspect the
    collated batch to learn whether a 'loss_scale' tensor is coming -- yet they
    must allocate a matching receive buffer or the TP broadcast collectives
    desynchronize. ms-swift decides this purely from the strategy spec
    (``is_binary_loss_scale``; binary strategies drop the tensor in
    ``_encode_context_list``), so every rank can derive the same answer from args.
    """
    if not getattr(args, "sft", False) or getattr(args, "sft_data_backend", "megatron") != "swift":
        return False
    spec = getattr(args, "sft_loss_scale", None)
    if not spec:
        return False
    # The Swift SFT implementation is vendored into this repository; do not
    # reintroduce an external ms-swift runtime dependency from the batch path.
    from data.swift.loss_scale import get_loss_scale

    return not get_loss_scale(spec).is_binary_loss_scale


# Kept alphabetical: callers unpack the return of
# get_batch_for_model_with_all_stage_tokens positionally in this order.
# 'loss_scale' is the optional per-token weight from the ms-swift SFT backend
# (None for every binary loss_scale strategy, which is the common case). It
# weights the loss *numerator* only -- 'loss_mask' remains the binary token
# count used as the denominator, and is what MTP reduces over internally.
BATCH_KEYS = ["attention_mask", "cu_seqlens", "cu_seqlens_padded", "hybrid_cp_group", "labels", "local_cp_size", "loss_mask", "loss_scale", "max_seqlen", "position_ids", "tokens"]
cur_platform = get_platform()


def broadcast_loss_scale_on_tp_rank(batch, args, tp_rank, src_rank, group, seq_length):
    """Broadcast 'loss_scale' across TP ranks, alongside upstream's fixed key set.

    ``megatron.core.utils.get_batch_on_this_tp_rank`` broadcasts a hardcoded set
    of keys and hands non-zero TP ranks a freshly built dict, so 'loss_scale'
    neither crosses the broadcast nor survives into that dict. Carry it here,
    gated on 'loss_mask' so it is present on exactly the stages that compute the
    loss -- that is the key upstream has already decided per PP stage.
    """
    if not sft_loss_scale_is_non_binary(args):
        batch["loss_scale"] = None
        return batch
    if batch.get("loss_mask") is None:
        # This stage does not compute the loss, so it needs no weights either.
        batch["loss_scale"] = None
        return batch
    if tp_rank != 0:
        batch["loss_scale"] = torch.empty(
            (args.micro_batch_size, seq_length), dtype=torch.float32, device=cur_platform.device()
        )
    loss_scale = batch.get("loss_scale")
    assert loss_scale is not None, (
        "a non-binary --sft-loss-scale was configured but the collated batch carries no "
        "'loss_scale'; the data backend and the loss plumbing disagree"
    )
    torch.distributed.broadcast(loss_scale, src_rank, group=group)
    batch["loss_scale"] = loss_scale
    return batch


def slice_loss_scale_on_cp_rank(
    batch, cp_group, is_hybrid_cp=False, use_contiguous_cp=False
):
    """Partition 'loss_scale' across CP ranks, in lockstep with 'loss_mask'.

    ``get_sft_batch_on_this_cp_rank`` only index-selects a hardcoded
    ``SEQUENCE_KEYS = ('tokens', 'labels', 'loss_mask', 'position_ids')``, so a
    sequence-shaped tensor added here is left un-sliced and would silently
    mismatch the sliced losses. Redo the same THD partition for 'loss_scale'.
    """
    loss_scale = batch.get("loss_scale")
    if loss_scale is None or cp_group is None:
        return batch
    cp_size = torch.distributed.get_world_size(cp_group)
    if cp_size <= 1:
        return batch
    if use_contiguous_cp:
        if loss_scale.size(1) % (2 * cp_size) != 0:
            raise ValueError(
                "loss_scale sequence length must be divisible by 2 * CP size for "
                f"contiguous DSv4 CP, got {loss_scale.size(1)} and CP size {cp_size}"
            )
        local_seq_len = loss_scale.size(1) // cp_size
        cp_rank = torch.distributed.get_rank(cp_group)
        batch["loss_scale"] = loss_scale.narrow(
            1, cp_rank * local_seq_len, local_seq_len
        ).contiguous()
        return batch
    if is_hybrid_cp:
        # Hybrid CP routes through the zigzag pretrain partitioner, whose key
        # handling differs; combining it with non-binary loss_scale is untested.
        raise NotImplementedError(
            "hybrid context parallel with a non-binary loss_scale is not supported yet; "
            "use a binary --sft-loss-scale (e.g. 'default+ignore_empty_think')"
        )
    import transformer_engine_torch as tex

    cu_seqlens = batch["cu_seqlens_padded"] if batch.get("cu_seqlens_padded") is not None \
        else batch["cu_seqlens"]
    index = tex.thd_get_partitioned_indices(
        cu_seqlens[0],
        loss_scale.size(1),
        cp_size,
        torch.distributed.get_rank(cp_group),
    )
    batch["loss_scale"] = loss_scale.index_select(1, index)
    return batch


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
            batch[key] = batch[key].to(cur_platform.device(), non_blocking=True) if key in batch and batch[key] is not None else None

    # Unlike the generic MCore broadcaster, this model-specific path keeps token
    # ids on every PP stage for DSv4 Engram/hash routing. It must still negotiate
    # the actual packed length before non-zero TP ranks allocate receive buffers.
    if args.sft:
        actual_seq_length = torch.tensor(
            batch["tokens"].shape[1] if tp_rank == 0 else 0,
            dtype=torch.int32,
            device=cur_platform.device(),
        )
        _broadcast(actual_seq_length)
        sequence_length = int(actual_seq_length.item())
    else:
        sequence_length = args.seq_length

    if tp_rank != 0:
        shape = (args.micro_batch_size, sequence_length)
        batch = {key: None for key in BATCH_KEYS}
        batch["tokens"] = torch.empty(shape, dtype=torch.int64, device=cur_platform.device())
        batch["labels"] = torch.empty(shape, dtype=torch.int64, device=cur_platform.device())
        batch["loss_mask"] = torch.empty(shape, dtype=torch.float32, device=cur_platform.device())
        batch["position_ids"] = torch.empty(shape, dtype=torch.int64, device=cur_platform.device())
        # Only allocate a receive buffer when the sender will actually broadcast
        # one; otherwise the collective counts would not match. Every rank can
        # decide this from args alone, without having built the dataset.
        if sft_loss_scale_is_non_binary(args):
            batch["loss_scale"] = torch.empty(shape, dtype=torch.float32, device=cur_platform.device())
        if args.create_attention_mask_in_dataloader:
            batch["attention_mask"] = torch.empty(
                (args.micro_batch_size, 1, sequence_length, sequence_length),
                dtype=torch.bool,
                device=cur_platform.device(),
            )

    def _broadcast_cu_seqlens(value):
        numel = torch.tensor(
            value.numel() if tp_rank == 0 and value is not None else 0,
            dtype=torch.int64,
            device=cur_platform.device(),
        )
        _broadcast(numel)
        if int(numel.item()) == 0:
            return None
        if tp_rank != 0:
            value = torch.empty(
                (1, int(numel.item())), dtype=torch.int32, device=cur_platform.device()
            )
        _broadcast(value)
        return value

    if args.sft:
        batch["cu_seqlens"] = _broadcast_cu_seqlens(batch.get("cu_seqlens"))
        if tp_rank != 0:
            batch["max_seqlen"] = torch.empty(
                1, dtype=torch.int32, device=cur_platform.device()
            )
        _broadcast(batch["max_seqlen"])
        batch["cu_seqlens_padded"] = _broadcast_cu_seqlens(
            batch.get("cu_seqlens_padded")
        )

    _broadcast(batch["tokens"])
    # 'loss_scale' is only consumed where the loss is computed, so it travels with
    # 'loss_mask' on every branch below and is dropped wherever that is dropped.
    if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
        _broadcast(batch["labels"])
        _broadcast(batch["loss_mask"])
        _broadcast(batch["loss_scale"])
        _broadcast(batch["position_ids"])
        _broadcast(batch["attention_mask"])
    elif is_first_stage:
        if is_dualpipev:
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["loss_scale"])
        else:
            batch["labels"] = None
            batch["loss_mask"] = None
            batch["loss_scale"] = None
        _broadcast(batch["position_ids"])
        _broadcast(batch["attention_mask"])
    elif is_last_stage:
        _broadcast(batch["labels"])
        _broadcast(batch["loss_mask"])
        _broadcast(batch["loss_scale"])
        _broadcast(batch["attention_mask"])
        batch["position_ids"] = None
    else:
        batch["labels"] = None
        batch["loss_mask"] = None
        batch["loss_scale"] = None
        batch["position_ids"] = None
        batch["attention_mask"] = None

    is_hybrid_cp = args.hybrid_context_parallel
    cp_group = get_context_parallel_group()
    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=is_hybrid_cp,
        cp_group=cp_group,
        hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        use_contiguous_cp=args.experimental_attention_variant == "dsv4_hybrid",
    )
    batch = slice_loss_scale_on_cp_rank(
        batch,
        cp_group,
        is_hybrid_cp=is_hybrid_cp,
        use_contiguous_cp=args.experimental_attention_variant == "dsv4_hybrid",
    )
    return [batch[key] for key in BATCH_KEYS]
