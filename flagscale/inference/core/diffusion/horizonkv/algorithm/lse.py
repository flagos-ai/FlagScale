# Modified for the HorizonKV inference artifact; see NOTICE.
"""Stable LSE merge for hit and miss attention partials."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _merge_attention_lse_parts_kernel(
    out_a,
    lse_a,
    out_b,
    lse_b,
    out,
    total_elements: tl.constexpr,
    Lq: tl.constexpr,
    q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    token_linear = offsets // head_dim
    token_in_row = token_linear % (Lq * q_heads)
    lq = token_in_row // q_heads
    qh = token_in_row - lq * q_heads
    row = token_linear // (Lq * q_heads)
    lse_offset = (row * q_heads + qh) * Lq + lq
    la = tl.load(lse_a + lse_offset, mask=mask, other=-3.4028234663852886e38)
    lb = tl.load(lse_b + lse_offset, mask=mask, other=-3.4028234663852886e38)
    m = tl.maximum(la, lb)
    both_invalid = m < -3e38
    safe_m = tl.where(both_invalid, 0.0, m)
    wa = tl.where(la < -3e38, 0.0, tl.exp(la - safe_m))
    wb = tl.where(lb < -3e38, 0.0, tl.exp(lb - safe_m))
    denom = wa + wb
    va = tl.load(out_a + offsets, mask=mask, other=0.0).to(tl.float32)
    vb = tl.load(out_b + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (va * wa + vb * wb) / tl.maximum(denom, 1e-20)
    result = tl.where(both_invalid, 0.0, result)
    tl.store(out + offsets, result, mask=mask)


def merge_attention_lse_parts_triton(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> torch.Tensor | None:
    Lq = int(out_a.shape[1])
    q_heads = int(out_a.shape[2])
    head_dim = int(out_a.shape[3])
    out = torch.empty_like(out_a)
    total_elements = int(out.numel())
    if total_elements <= 0:
        return out
    grid = (triton.cdiv(total_elements, 256),)
    _merge_attention_lse_parts_kernel[grid](
        out_a,
        lse_a,
        out_b,
        lse_b,
        out,
        total_elements,
        Lq,
        q_heads,
        head_dim,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out


def _merge_attention_partials(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    try:
        merged = merge_attention_lse_parts_triton(
            out_a.contiguous(), lse_a.contiguous(), out_b.contiguous(), lse_b.contiguous()
        )
    except Exception:
        merged = None
    if merged is not None:
        return merged.to(dtype=out_dtype)
    merge_max = torch.maximum(lse_a, lse_b)
    both_invalid = torch.isneginf(merge_max)
    safe_merge_max = torch.where(both_invalid, torch.zeros_like(merge_max), merge_max)
    weight_a = torch.exp(lse_a - safe_merge_max).transpose(1, 2).unsqueeze(-1)
    weight_b = torch.exp(lse_b - safe_merge_max).transpose(1, 2).unsqueeze(-1)
    invalid_mask = both_invalid.transpose(1, 2).unsqueeze(-1)
    weight_a = torch.where(invalid_mask, torch.zeros_like(weight_a), weight_a)
    weight_b = torch.where(invalid_mask, torch.zeros_like(weight_b), weight_b)
    denom = (weight_a + weight_b).clamp_min_(1e-20)
    merged = (
        out_a.to(dtype=torch.float32) * weight_a + out_b.to(dtype=torch.float32) * weight_b
    ) / denom
    return merged.to(dtype=out_dtype)
