# Modified for the HorizonKV inference artifact; see NOTICE.
"""Dispatch pooled block selection to the uniform/packed CUDA attention backend."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .block_selection import _select_pooled_block_cols
from .sparse_backend import rows_blocksparse_attention as _rows_blocksparse_attention
from .sparse_backend import rows_blocksparse_attention_varlen as _rows_blocksparse_attention_varlen


@triton.jit
def _block_cols_to_descending_rowmask_kernel(
    block_cols,
    row_blockmask,
    n_k_blocks,
    max_k_blocks,
    BLOCK_KEEP: tl.constexpr,
    BLOCK_MASK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    keep_offsets = tl.arange(0, BLOCK_KEEP)
    cols = tl.load(
        block_cols + row * max_k_blocks + keep_offsets,
        mask=keep_offsets < max_k_blocks,
        other=-1,
    )
    cols = tl.sort(cols, descending=True)
    mask_base = row * n_k_blocks
    tl.store(row_blockmask + mask_base + keep_offsets, cols, mask=keep_offsets < max_k_blocks)
    fill_offsets = max_k_blocks + tl.arange(0, BLOCK_MASK)
    tl.store(row_blockmask + mask_base + fill_offsets, -1, mask=fill_offsets < n_k_blocks)


def _block_cols_to_rowmask(block_cols, *, n_k_blocks):
    rows, n_q_blocks, keep = block_cols.shape
    mask = torch.empty((rows, n_q_blocks, n_k_blocks), device=block_cols.device, dtype=torch.int32)
    _block_cols_to_descending_rowmask_kernel[rows * n_q_blocks,](
        block_cols,
        mask,
        n_k_blocks,
        keep,
        BLOCK_KEEP=triton.next_power_of_2(keep),
        BLOCK_MASK=triton.next_power_of_2(n_k_blocks),
        num_warps=4,
        num_stages=1,
    )
    return mask


def blocksparse_attention_pooled_triton(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float,
    block_m: int = 64,
    block_n: int = 64,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    k_lens: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    rows, Lq, _, head_dim = q.shape
    packed_varlen_k = k.ndim == 3
    Lk = int(max_seqlen_k) if packed_varlen_k else int(k.shape[1])
    if Lk <= block_n:
        return None
    block_cols = _select_pooled_block_cols(
        q,
        k,
        keep_ratio=1 - sparsity,
        block_m=block_m,
        block_n=block_n,
        softmax_scale=softmax_scale,
        k_lens=k_lens if packed_varlen_k else None,
        cu_seqlens_k=cu_seqlens_k if packed_varlen_k else None,
        max_seqlen_k=Lk,
    )
    if block_cols is None:
        return None
    scale = softmax_scale if softmax_scale is not None else head_dim ** (-0.5)
    mask = _block_cols_to_rowmask(block_cols, n_k_blocks=(Lk + block_n - 1) // block_n)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        row_blockmask=mask,
        block_m=block_m,
        block_n=block_n,
        softmax_scale=scale,
        return_lse=return_lse,
    )
    if packed_varlen_k:
        return _rows_blocksparse_attention_varlen(
            **kwargs,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=Lk,
        )
    return _rows_blocksparse_attention(**kwargs)
