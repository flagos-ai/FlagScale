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
- ``cu_seqlens``        <- cumulative real-token lengths for each packed segment.
- ``cu_seqlens_padded`` <- cumulative physical lengths after per-segment CP
                           alignment; a trailing pack pad is a zero-real-token segment.
- ``max_seqlen``        <- max segment length from cu_seqlens.
- ``attention_mask``    <- None (varlen / THD attention via packed_seq_params).
"""

from functools import partial
from typing import Any, Dict, List, Optional

import torch

IGNORE_INDEX = -100


def _cu_seqlens_from_position_ids(
    position_ids: torch.Tensor, total_len: int, real_len: Optional[int] = None
) -> torch.Tensor:
    """Derive cu_seqlens for a packed sequence.

    ``position_ids`` restarts at 0 at the start of every packed sub-sequence, so
    the zero indices are document boundaries -- but only inside the *real* token
    region. The template pads with a repeating ramp
    (``torch.arange(cp_size*2).repeat(...)`` in ``template/base.py``), so the pad
    tail holds a zero every ``2*cp_size`` tokens. Scanning the whole sequence
    therefore invents one bogus document per 16 pad tokens: with
    ``--sft-pad-to-packing-length`` the pad tail is ~13k tokens and a 60-row
    sample reported packs with 771 and 1356 "documents", corrupting the varlen
    attention boundaries and ``max_seqlen``.

    So restrict the scan to ``real_len`` and append the pad tail as one trailing
    segment, mirroring the megatron SFT dataset (which sets ``cu_seqlens[-1]`` to
    the real end explicitly). ``real_len`` comes from the collator's
    ``seq_lens``; when unavailable this falls back to the whole-sequence scan,
    which is correct whenever there is no pad tail (e.g. padding_to=None).
    """
    pos = position_ids.flatten()
    scan_len = total_len if real_len is None else max(0, min(int(real_len), total_len))
    idx = torch.arange(scan_len, device=pos.device, dtype=torch.int32)
    starts = idx[pos[:scan_len] == 0]
    bounds = [starts]
    if 0 < scan_len < total_len:
        # Close the real region; the pad tail becomes its own segment so the
        # final boundary still equals total_len (consumers split the full tensor).
        bounds.append(torch.tensor([scan_len], device=pos.device, dtype=torch.int32))
    bounds.append(torch.tensor([total_len], device=pos.device, dtype=torch.int32))
    return torch.cat(bounds)


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

    # Real (pre-batch-pad) packed length, so the pad tail is not scanned for
    # bogus document boundaries. ms-swift reports per-row raw lengths in
    # ``seq_lens``, but before the batch-level pad the template also CP-pads each
    # row to a multiple of ``2*cp`` (``_handle_megatron_cp``). So the real region
    # is ``sum(ceil(len_i / 2cp) * 2cp)``, not ``sum(len_i)`` -- missing that CP
    # rounding left the last document's final tokens glued to the pad tail.
    seq_lens = res.get("seq_lens")
    cp = int(getattr(template, "sequence_parallel_size", 1) or 1)
    if seq_lens is not None:
        try:
            raw_lens = [int(x) for x in seq_lens]
        except TypeError:
            raw_lens = [int(seq_lens)]
        physical_lens = raw_lens
        if cp > 1:
            unit = 2 * cp
            physical_lens = [((length + unit - 1) // unit) * unit for length in raw_lens]
        physical_len = sum(physical_lens)

        if physical_len > total_len:
            raise ValueError(
                "the CP-padded Swift sequence lengths exceed the collated tensor: "
                f"{physical_len} > {total_len}"
            )

        real_lens = raw_lens
        tail_len = total_len - physical_len
        if tail_len:
            # The pack tail occupies physical rows but contains no real tokens.
            physical_lens = [*physical_lens, tail_len]
            real_lens = [*real_lens, 0]
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(real_lens, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
        )
        cu_seqlens_padded = torch.tensor(
            [0, *torch.tensor(physical_lens, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
        )
    else:
        # Custom templates without seq_lens cannot distinguish alignment padding
        # from real rows. Preserve the legacy position-id-derived metadata.
        cu_seqlens = _cu_seqlens_from_position_ids(position_ids, total_len)
        cu_seqlens_padded = cu_seqlens.clone()

    max_seqlen = int(cu_seqlens_padded.diff().max().item())

    # Carry the dataloader batch dim (1, n): forward_step squeezes it back out.
    cu_seqlens = cu_seqlens.unsqueeze(0)
    cu_seqlens_padded = cu_seqlens_padded.unsqueeze(0)

    out = {
        "tokens": input_ids.to(torch.int64),
        "labels": safe_labels.to(torch.int64),
        "loss_mask": loss_mask,
        # None for every binary strategy (ms-swift drops it in
        # ``_encode_context_list``), so the common case carries no extra tensor.
        "loss_scale": None if loss_scale is None else loss_scale.to(torch.float32),
        "position_ids": position_ids.to(torch.int64),
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        # Shape (1,) int32, matching the receive buffer non-zero TP ranks
        # allocate in megatron.core.utils.get_batch_on_this_tp_rank:
        #     max_seqlen = torch.empty(1, dtype=torch.int32, ...)
        # A 0-dim or int64 tensor here makes that broadcast disagree on both
        # shape and dtype, and NCCL hangs on the first training step rather than
        # raising -- every rank sits in the same collective waiting forever.
        "max_seqlen": torch.tensor([max_seqlen], dtype=torch.int32),
        "attention_mask": None,
    }
    return out


def swift_padding_to_from_args(args) -> Optional[int]:
    """Reproduce ms-swift ``megatron.utils.get_padding_to`` from FlagScale args.

    The packed sequence length must be padded so it stays divisible by the TP
    (sequence-parallel), CP and FP8/FP4 block constraints. Keeping this identical
    to ms-swift is what makes the two runs bit-comparable.

    With ``--sft-pad-to-packing-length`` the granularity is instead the whole
    packing length, so every pack comes out exactly that long. The template does
    ``ceil(max(seq_lens) / padding_to) * padding_to`` and a pack never exceeds the
    packing length, so that ceil collapses to 1x and the batch length is constant.
    The packed DSv4 CSA path needs this: its ``torch.compile(fullgraph=True)``
    helpers receive the derived sizes as Python ints (``l_local``, ``c_cap`` in
    ``csa_utils/cp_utils.py``), and ``capacity`` even sizes a ``torch.arange``, so
    Dynamo specializes per pack length rather than going symbolic. Variable-length
    packs then blow past ``recompile_limit`` (observed: hit 256 and abort with
    FailOnRecompileLimitHit). Trade-off: padding compute, and no longer
    bit-comparable with ms-swift.
    """
    if getattr(args, "sft_pad_to_packing_length", False):
        packing_length = getattr(args, "sft_packing_length", None) or getattr(
            args, "sft_max_length", None
        ) or args.seq_length
        return int(packing_length)

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
