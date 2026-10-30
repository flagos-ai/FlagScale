# Modified for the HorizonKV inference artifact; see NOTICE.
"""64x64 CUDA block-sparse attention for uniform and variable-length rows."""
from __future__ import annotations
import math
import torch

_BLOCK_SPARSE_ATTN_FORWARD = None
_ROW_METADATA_CACHE: dict[
    tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}
_VARLEN_ROW_METADATA_CACHE: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}
from block_sparse_attn.block_sparse_attn_interface import (
    _wrapped_block_sparse_attn_forward as _BLOCK_SPARSE_ATTN_FORWARD,
)


def rows_blocksparse_backend_is_available() -> bool:
    return _BLOCK_SPARSE_ATTN_FORWARD is not None


def _rows_blocksparse_metadata(
    device: torch.device, *, rows: int, lq: int, lk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    key = (device_index, int(rows), int(lq), int(lk))
    cached = _ROW_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    cu_q = torch.tensor([0, int(lq)], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, int(lk)], device=device, dtype=torch.int32)
    head_mask_type = torch.arange(1, int(rows) + 1, device=device, dtype=torch.int32)
    streaming_info = torch.zeros(int(rows) * 2, device=device, dtype=torch.int32)
    result = (cu_q, cu_k, head_mask_type, streaming_info)
    _ROW_METADATA_CACHE[key] = result
    return result


@torch.no_grad()
def rows_blocksparse_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_blockmask: torch.Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if _BLOCK_SPARSE_ATTN_FORWARD is None:
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None
    if tuple(k.shape) != tuple(v.shape):
        return None
    if (
        q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or (q.dtype != k.dtype)
        or (q.dtype != v.dtype)
        or (q.stride(-1) != 1)
        or (k.stride(-1) != 1)
        or (v.stride(-1) != 1)
    ):
        return None
    rows, lq, one_head, head_dim = q.shape
    lk = int(k.shape[1])
    if (
        int(one_head) != 1
        or int(k.shape[0]) != int(rows)
        or int(k.shape[2]) != 1
        or (int(k.shape[-1]) != int(head_dim))
    ):
        return None
    n_q_blocks = math.ceil(int(lq) / int(block_m))
    n_k_blocks = math.ceil(int(lk) / int(block_n))
    if (
        row_blockmask.device != q.device
        or row_blockmask.dtype != torch.int32
        or (not row_blockmask.is_contiguous())
        or (tuple(row_blockmask.shape) != (int(rows), n_q_blocks, n_k_blocks))
    ):
        return None
    cu_q, cu_k, head_mask_type, streaming_info = _rows_blocksparse_metadata(
        q.device, rows=int(rows), lq=int(lq), lk=int(lk)
    )
    q_rows = q.squeeze(2).permute(1, 0, 2)
    k_rows = k.squeeze(2).permute(1, 0, 2)
    v_rows = v.squeeze(2).permute(1, 0, 2)
    forward_result = _BLOCK_SPARSE_ATTN_FORWARD(
        q_rows,
        k_rows,
        v_rows,
        cu_q,
        cu_k,
        int(block_m),
        int(block_n),
        head_mask_type,
        streaming_info,
        row_blockmask.unsqueeze(0),
        int(lq),
        int(lk),
        0.0,
        float(softmax_scale),
        False,
        False,
        False,
        -1,
        -1,
    )
    out = forward_result[0]
    out = out.permute(1, 0, 2).unsqueeze(2).contiguous()
    if not return_lse:
        return out
    softmax_lse = forward_result[1]
    if (
        not isinstance(softmax_lse, torch.Tensor)
        or softmax_lse.ndim != 3
        or int(softmax_lse.shape[0]) != 1
        or (int(softmax_lse.shape[1]) != int(rows))
        or (int(softmax_lse.shape[2]) < int(lq))
    ):
        return None
    return (out, softmax_lse[0, :, : int(lq)].unsqueeze(1).contiguous())


def _varlen_rows_blocksparse_metadata(
    device: torch.device, *, rows: int, lq: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = (
        int(device.index) if device.index is not None else int(torch.cuda.current_device())
    )
    key = (device_index, int(rows), int(lq))
    cached = _VARLEN_ROW_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    cu_q = torch.arange(
        0, (int(rows) + 1) * int(lq), step=int(lq), device=device, dtype=torch.int32
    )
    head_mask_type = torch.ones(1, device=device, dtype=torch.int32)
    streaming_info = torch.zeros(2, device=device, dtype=torch.int32)
    result = (cu_q, head_mask_type, streaming_info)
    _VARLEN_ROW_METADATA_CACHE[key] = result
    return result


@torch.no_grad()
def rows_blocksparse_attention_varlen(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    row_blockmask: torch.Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if _BLOCK_SPARSE_ATTN_FORWARD is None:
        return None
    if q.ndim != 4 or k.ndim != 3 or v.ndim != 3 or (tuple(k.shape) != tuple(v.shape)):
        return None
    if (
        q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or (q.dtype != k.dtype)
        or (q.dtype != v.dtype)
        or (q.stride(-1) != 1)
        or (k.stride(-1) != 1)
        or (v.stride(-1) != 1)
    ):
        return None
    rows, lq, one_head, head_dim = q.shape
    max_seqlen_k = int(max_seqlen_k)
    if (
        int(one_head) != 1
        or int(k.shape[1]) != 1
        or int(k.shape[2]) != int(head_dim)
        or (max_seqlen_k <= 0)
        or (cu_seqlens_k.device != q.device)
        or (cu_seqlens_k.dtype != torch.int32)
        or (not cu_seqlens_k.is_contiguous())
        or (int(cu_seqlens_k.numel()) != int(rows) + 1)
    ):
        return None
    n_q_blocks = math.ceil(int(lq) / int(block_m))
    n_k_blocks = math.ceil(max_seqlen_k / int(block_n))
    if (
        row_blockmask.device != q.device
        or row_blockmask.dtype != torch.int32
        or (not row_blockmask.is_contiguous())
        or (tuple(row_blockmask.shape) != (int(rows), n_q_blocks, n_k_blocks))
    ):
        return None
    cu_q, head_mask_type, streaming_info = _varlen_rows_blocksparse_metadata(
        q.device, rows=int(rows), lq=int(lq)
    )
    q_packed = q.reshape(int(rows) * int(lq), 1, int(head_dim))
    forward_result = _BLOCK_SPARSE_ATTN_FORWARD(
        q_packed,
        k,
        v,
        cu_q,
        cu_seqlens_k,
        int(block_m),
        int(block_n),
        head_mask_type,
        streaming_info,
        row_blockmask.unsqueeze(1),
        int(lq),
        max_seqlen_k,
        0.0,
        float(softmax_scale),
        False,
        False,
        False,
        -1,
        -1,
    )
    out = forward_result[0].view(int(rows), int(lq), 1, int(head_dim)).contiguous()
    if not return_lse:
        return out
    softmax_lse = forward_result[1]
    if (
        not isinstance(softmax_lse, torch.Tensor)
        or softmax_lse.ndim != 3
        or int(softmax_lse.shape[0]) != int(rows)
        or (int(softmax_lse.shape[1]) != 1)
        or (int(softmax_lse.shape[2]) < int(lq))
    ):
        return None
    return (out, softmax_lse[:, :, : int(lq)].contiguous())
