# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Utilities for real and physical packed-sequence boundaries."""

from typing import Optional

import torch


def padding_mask_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Build a 1-D physical-layout padding mask from dual THD boundaries."""
    if cu_seqlens_padded is None or torch.equal(cu_seqlens, cu_seqlens_padded):
        return None
    if cu_seqlens.shape != cu_seqlens_padded.shape:
        raise ValueError("cu_seqlens and cu_seqlens_padded must have the same shape")

    real_lengths = cu_seqlens.diff()
    physical_lengths = cu_seqlens_padded.diff()
    if bool(torch.any(real_lengths < 0)) or bool(torch.any(real_lengths > physical_lengths)):
        raise ValueError("each real packed-sequence length must be in [0, physical_length]")

    total_length = int(cu_seqlens_padded[-1].item())
    positions = torch.arange(
        total_length,
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    segment_ids = torch.bucketize(positions, cu_seqlens_padded[1:], right=True)
    offsets = positions - cu_seqlens_padded[:-1][segment_ids]
    return offsets >= real_lengths[segment_ids]

