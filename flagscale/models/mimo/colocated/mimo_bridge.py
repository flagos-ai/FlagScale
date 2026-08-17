# Copyright (c) 2025, BAAI. All rights reserved.

"""Colocated cross-rank utilities for MIMO module communication.

The helpers here assume the deterministic rank ordering produced by
``flagscale.models.mimo.hetero_pg_utils``.
"""

import torch.distributed as dist


def _get_language_tp_first_global_rank(language_pg):
    """Return the global rank of the first member of the language TP group."""
    if language_pg is None or language_pg.tp is None:
        return dist.get_rank()
    # Query group membership instead of assuming contiguous TP ranks.
    return dist.get_process_group_ranks(language_pg.tp)[0]


def _owner_offset(forward_idx_in_round, vit_batch_factor, language_tp_size):
    """Return the TP-group offset of the owner of the given microbatch.

    The macro batch is split into whole microbatches across the language TP
    group: TP member j owns microbatches ``[j*vbf/tp, (j+1)*vbf/tp)``.
    """
    return (forward_idx_in_round * language_tp_size // vit_batch_factor) % language_tp_size


def get_source_vision_rank(language_pg, forward_idx_in_round, vit_batch_factor):
    """Return the owner ViT rank of the given forward's microbatch slice.

    Within a language TP group, every member is also a ViT rank because vision
    TP size is 1.  The group's macro batch is split into whole microbatches
    (see ``get_my_microbatch_range``): TP member j computes and supplies the
    microbatches in its slice.

    Example (language TP=2, vit_batch_factor=4): forwards 0,1 belong to
    TP member 0; forwards 2,3 to TP member 1.
    """
    assert forward_idx_in_round >= 0, (
        f"forward_idx_in_round must be non-negative, got {forward_idx_in_round}"
    )
    assert vit_batch_factor >= 1, f"vit_batch_factor must be >= 1, got {vit_batch_factor}"
    tp_first = _get_language_tp_first_global_rank(language_pg)
    language_tp_size = max(1, dist.get_world_size(language_pg.tp))
    assert vit_batch_factor % language_tp_size == 0, (
        f"vit_batch_factor ({vit_batch_factor}) must be divisible by language_tp_size "
        f"({language_tp_size}) for colocated MIMO source-rank cycling"
    )
    src_rank = tp_first + _owner_offset(forward_idx_in_round, vit_batch_factor, language_tp_size)
    assert src_rank >= 0, f"computed source vision rank {src_rank} is negative"
    return src_rank


def get_my_microbatch_range(language_pg, num_micro):
    """Return this rank's ``[lo, hi)`` microbatch slice within the macro batch.

    The language TP group splits the macro batch into whole microbatches:
    TP rank j owns ``[j*num/tp, (j+1)*num/tp)``.  Each rank's ViT forward
    therefore covers exactly ``vision_micro_batch_size`` samples (vbs) and
    the TP group jointly covers the ``vbf * mbs`` samples of one entity's
    macro batch.
    """
    tp_size = max(1, dist.get_world_size(language_pg.tp))
    assert num_micro % tp_size == 0, (
        f"vit_batch_factor ({num_micro}) must be divisible by language TP size "
        f"({tp_size}) so each TP rank owns whole microbatches"
    )
    per_rank = num_micro // tp_size
    tp_rank = dist.get_rank(language_pg.tp) if tp_size > 1 else 0
    lo = tp_rank * per_rank
    return lo, lo + per_rank


def exchange_macro_outputs(entries, language_pg, vit_batch_factor, mark_requires_grad=True):
    """Exchange per-microbatch vision outputs inside the language TP group.

    Each microbatch entry is computed by its owner rank (see
    ``get_my_microbatch_range``) and broadcast to the other group members;
    non-owned entries must be preallocated receive buffers.  After the
    exchange every served tensor is marked requires_grad so the scheduler
    registers a grad hook per microbatch — its macro-batch completion trigger
    relies on a hook firing for *every* microbatch.  Set after the broadcast
    to avoid in-place writes into requires-grad leaves.  Gradients captured
    for microbatches this rank does not own are unused.

    Args:
        entries: List of ``{"main": Tensor, "aux": list[Tensor] | None}``
            dicts, one per microbatch in the macro batch; modified in place.
        language_pg: Language module process group collection.
        vit_batch_factor: Microbatches per macro batch (owner mapping).
        mark_requires_grad: Whether to mark served tensors requires_grad.
            Pass False when the ViT is frozen: without hooks no gradients are
            expected, the exhausted macro batch is dropped silently, and the
            delayed ViT backward never triggers.
    """
    for forward_idx, entry in enumerate(entries):
        src_rank = get_source_vision_rank(language_pg, forward_idx, vit_batch_factor)
        if entry["main"] is not None:
            entry["main"] = broadcast_to_language_tp(entry["main"], language_pg, src_rank)
        if entry["aux"] is not None:
            entry["aux"] = [
                broadcast_to_language_tp(f, language_pg, src_rank) for f in entry["aux"]
            ]
    for entry in entries:
        if not mark_requires_grad:
            break
        if entry["main"] is not None:
            entry["main"].requires_grad_(True)
        if entry["aux"] is not None:
            for f in entry["aux"]:
                f.requires_grad_(True)


def broadcast_to_language_tp(tensor, language_pg, src_rank):
    """Broadcast ``tensor`` from ``src_rank`` to all ranks in the language TP group.

    Args:
        tensor: Tensor on the current rank.  Only meaningful on ``src_rank``
            before the call; after the call every rank in the TP group holds
            the same data.
        language_pg: Language module process group collection.
        src_rank: Global rank of the source within the language TP group.

    Returns:
        Tensor after broadcast.
    """
    if language_pg is None or language_pg.tp is None:
        return tensor
    if dist.get_world_size(language_pg.tp) <= 1:
        return tensor
    tensor = tensor.contiguous()
    dist.broadcast(tensor, src=src_rank, group=language_pg.tp)
    return tensor
