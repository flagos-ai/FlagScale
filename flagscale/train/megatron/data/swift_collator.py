# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Collate ms-swift encoded/packed rows into FlagScale Megatron batch keys.

The ms-swift template ``data_collator`` (Megatron + padding_free branch) returns
a batch shaped for the ms-swift trainer::

    input_ids   : LongTensor [1, T]         (T packed & padded tokens)
    labels      : LongTensor [1, T]         (-100 on ignored positions)
    loss_scale  : FloatTensor [1, T]        (per-token loss weight; optional)
    position_ids: LongTensor [1, T]         (restarts at 0 per packed sequence)
    seq_lens    : List[int]                 (real, unpadded length per sequence)
    attention_mask: None                    (padding_free -> varlen attention)

FlagScale's ``get_batch_for_model_with_all_stage_tokens`` consumes::

    tokens, labels, loss_mask, position_ids,
    cu_seqlens, cu_seqlens_padded, max_seqlen, attention_mask,
    hybrid_cp_group, local_cp_size

This collator bridges the two:

- ``tokens``            <- ``input_ids``
- ``labels``            <- ``labels`` with -100 remapped so masked positions do
                           not affect the loss (loss_mask carries the masking).
- ``loss_mask``         <- ``labels != -100`` (token count; the loss denominator).
- ``loss_scale``        <- per-token weights when the strategy is non-binary, else
                           ``None``. Must multiply the loss *numerator* only -- see
                           the note in :func:`swift_sft_collate`.
- ``cu_seqlens``        <- restart points of ``position_ids`` (== 0), matching
                           ms-swift's ``get_packed_seq_params``.
- ``cu_seqlens_padded`` <- same as ``cu_seqlens`` (padding_free packs to a single
                           contiguous run; the trailing pad is one extra segment).
- ``max_seqlen``        <- max segment length from cu_seqlens.
- ``attention_mask``    <- None (varlen / THD attention via packed_seq_params).
"""

from functools import partial
from typing import Any, Dict, List, Optional

import torch

IGNORE_INDEX = -100


def _cu_seqlens_from_position_ids(position_ids: torch.Tensor, total_len: int) -> torch.Tensor:
    """Reproduce ms-swift ``get_packed_seq_params`` cu_seqlens from position_ids.

    ``position_ids`` restarts at 0 at the start of every packed sub-sequence, so
    the indices where it equals 0 are the segment boundaries.
    """
    pos = position_ids.flatten()
    idx = torch.arange(pos.shape[0], device=pos.device, dtype=torch.int32)
    starts = idx[pos == 0]
    cu_seqlens = torch.cat([
        starts,
        torch.tensor([total_len], device=pos.device, dtype=torch.int32),
    ])
    return cu_seqlens


def swift_sft_collate(
    batch: List[Any],
    *,
    template,
    padding_to: Optional[int] = None,
) -> Dict[str, Any]:
    """Collate one micro-batch of (packed) rows into FlagScale batch tensors."""
    # ms-swift's data_collator handles flattening packs (list-of-rows) itself and
    # produces a single padded, packed sequence with restarting position_ids.
    res = template.data_collator(batch, padding_to=padding_to)

    input_ids = res["input_ids"]  # [1, T]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    total_len = input_ids.shape[-1]

    labels = res.get("labels")
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    elif labels.dim() == 1:
        labels = labels.unsqueeze(0)

    position_ids = res.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(total_len, dtype=torch.long).unsqueeze(0)
    elif position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    loss_scale = res.get("loss_scale")
    if loss_scale is not None and loss_scale.dim() == 1:
        loss_scale = loss_scale.unsqueeze(0)

    # Next-token alignment: ms-swift emits labels aligned to their own position
    # (labels[i] == input_ids[i] for response tokens) and shifts them at batch
    # time via torch.roll(-1) so labels[i] becomes the target for position i.
    # Megatron-core does NOT shift internally, so we replicate the roll here.
    # The circular wrap is harmless because position 0's label/loss_scale are
    # forced to IGNORE/0 upstream, so the token wrapped into position T-1 is
    # masked out of the loss.
    labels = torch.roll(labels, -1, dims=-1)
    if loss_scale is not None:
        loss_scale = torch.roll(loss_scale, -1, dims=-1)

    # ms-swift keeps the mask and the weights separate (``trainers/trainer.py``):
    #     loss_mask = labels != -100
    #     losses    = losses * loss_scale        # weights: numerator only
    #     loss      = sum(losses * loss_mask) / loss_mask.sum()
    # so the denominator counts *tokens*, never weights. Folding loss_scale into
    # loss_mask would make a non-binary scheme divide by the sum of the weights
    # and silently diverge from ms-swift, so loss_scale is passed through as its
    # own tensor and must be applied to the numerator only.
    loss_mask = (labels != IGNORE_INDEX).to(torch.float32)

    # FlagScale's model reads token ids for labels too; positions that must not
    # contribute to the loss are already zeroed in loss_mask. Replace IGNORE with
    # a valid id so downstream gather/shift ops stay in-range.
    safe_labels = labels.clone()
    safe_labels[safe_labels == IGNORE_INDEX] = 0

    cu_seqlens = _cu_seqlens_from_position_ids(position_ids, total_len)
    max_seqlen = int(cu_seqlens.diff().max().item())

    # Carry the dataloader batch dim (1, n): forward_step squeezes it back out.
    cu_seqlens = cu_seqlens.unsqueeze(0)

    out = {
        "tokens": input_ids.to(torch.int64),
        "labels": safe_labels.to(torch.int64),
        "loss_mask": loss_mask,
        # None for every binary strategy (ms-swift drops it in
        # ``_encode_context_list``), so the common case carries no extra tensor.
        "loss_scale": None if loss_scale is None else loss_scale.to(torch.float32),
        "position_ids": position_ids.to(torch.int64),
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens,
        "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int64),
        "attention_mask": None,
    }
    return out


def swift_padding_to_from_args(args) -> Optional[int]:
    """Reproduce ms-swift ``megatron.utils.get_padding_to`` from FlagScale args.

    The packed sequence length must be padded so it stays divisible by the TP
    (sequence-parallel), CP and FP8/FP4 block constraints. Keeping this identical
    to ms-swift is what makes the two runs bit-comparable.
    """
    padding_to = None
    if args.tensor_model_parallel_size > 1 and getattr(args, "sequence_parallel", False):
        padding_to = args.tensor_model_parallel_size
    if args.context_parallel_size > 1:
        padding_to = (padding_to or 1) * args.context_parallel_size
    origin_padding_to = padding_to

    fp8_format = getattr(args, "fp8_format", None) or getattr(args, "fp8", None)
    fp4_format = getattr(args, "fp4_format", None) or getattr(args, "fp4", None)
    fp8_recipe = getattr(args, "fp8_recipe", None)
    if fp8_recipe == "blockwise":
        padding_to = (padding_to or 1) * 128
    elif fp8_recipe == "mxfp8":
        padding_to = (padding_to or 1) * 32
    elif fp8_format is not None or fp4_format is not None:
        padding_to = (padding_to or 1) * 16

    if getattr(args, "attention_backend", None) == "flash":
        # ms-swift only bumps to *64 for the 'fused' backend; flash keeps the
        # base padding. Mirror that exactly (no-op branch kept for clarity).
        pass
    if getattr(args, "attention_backend", None) == "fused":
        padding_to = max(padding_to or 1, (origin_padding_to or 1) * 64)
    return padding_to


def build_swift_sft_collator(template, padding_to: Optional[int] = None):
    """Return a ``collate_fn`` bound to ``template`` and ``padding_to``."""
    return partial(swift_sft_collate, template=template, padding_to=padding_to)
