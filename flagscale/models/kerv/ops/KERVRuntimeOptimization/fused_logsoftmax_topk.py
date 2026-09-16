# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KERV-specialized fused LogSoftmax+TopK for wide, short-batch logits."""

from __future__ import annotations

import torch

_TRITON_AVAILABLE = True
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on non-Triton platforms.
    _TRITON_AVAILABLE = False

    class _TritonStub:
        @staticmethod
        def jit(function=None, **_kwargs):
            if function is not None:
                return function
            return lambda candidate: candidate

    class _LanguageStub:
        constexpr = object()

    triton = _TritonStub()
    tl = _LanguageStub()


@triton.jit
def _chunk_logsumexp_topk(
    logits,
    partial_max,
    partial_sum,
    candidate_values,
    candidate_indices,
    rows: tl.constexpr,
    cols,
    chunks: tl.constexpr,
    stride_row,
    CHUNK: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * CHUNK + tl.arange(0, CHUNK)
    valid = offsets < cols
    values = tl.load(
        logits + row * stride_row + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    chunk_max = tl.max(values, axis=0)
    chunk_sum = tl.sum(tl.exp(values - chunk_max), axis=0)
    partial_offset = row * chunks + chunk
    tl.store(partial_max + partial_offset, chunk_max)
    tl.store(partial_sum + partial_offset, chunk_sum)

    candidate_offset = partial_offset * TOPK
    mutable = values
    lane = tl.arange(0, CHUNK)
    for rank in tl.static_range(TOPK):
        top_value, top_lane = tl.max(mutable, axis=0, return_indices=True)
        tl.store(candidate_values + candidate_offset + rank, top_value)
        tl.store(candidate_indices + candidate_offset + rank, chunk * CHUNK + top_lane)
        mutable = tl.where(lane == top_lane, -float("inf"), mutable)


@triton.jit
def _merge_logsumexp_topk(
    partial_max,
    partial_sum,
    candidate_values,
    candidate_indices,
    output_values,
    output_indices,
    chunks: tl.constexpr,
    CANDIDATES: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk_lanes = tl.arange(0, triton.next_power_of_2(chunks))
    chunk_mask = chunk_lanes < chunks
    max_values = tl.load(
        partial_max + row * chunks + chunk_lanes,
        mask=chunk_mask,
        other=-float("inf"),
    )
    sums = tl.load(
        partial_sum + row * chunks + chunk_lanes,
        mask=chunk_mask,
        other=0.0,
    )
    global_max = tl.max(max_values, axis=0)
    global_sum = tl.sum(sums * tl.exp(max_values - global_max), axis=0)
    log_normalizer = global_max + tl.log(global_sum)

    lanes = tl.arange(0, CANDIDATES)
    valid = lanes < chunks * TOPK
    base = row * chunks * TOPK
    values = tl.load(candidate_values + base + lanes, mask=valid, other=-float("inf"))
    indices = tl.load(candidate_indices + base + lanes, mask=valid, other=0)
    mutable = values
    output_base = row * TOPK
    for rank in tl.static_range(TOPK):
        top_value, top_lane = tl.max(mutable, axis=0, return_indices=True)
        top_index = tl.sum(tl.where(lanes == top_lane, indices, 0), axis=0)
        tl.store(output_values + output_base + rank, top_value - log_normalizer)
        tl.store(output_indices + output_base + rank, top_index)
        mutable = tl.where(lanes == top_lane, -float("inf"), mutable)


def fused_logsoftmax_topk(
    logits: torch.Tensor,
    k: int = 8,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the same interface as ``torch.topk(log_softmax(logits), k)``.

    This path is intentionally narrow: inference-only, contiguous FP32 CUDA
    logits, K=8, and vocabulary width between 16K and 64K. BF16/FP16 logits
    use PyTorch so quantized ties preserve KERV's original candidate order.
    """
    if not _TRITON_AVAILABLE or logits.dtype != torch.float32:
        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        result = torch.topk(probabilities, k, dim=-1)
        return result.values, result.indices
    if not (
        logits.is_cuda
        and logits.is_contiguous()
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and int(k) == 8
        and int(chunk_size) == 1024
        and 16384 <= int(logits.shape[1]) <= 65536
    ):
        raise ValueError("unsupported fused LogSoftmax+TopK input")

    rows, cols = map(int, logits.shape)
    chunks = triton.cdiv(cols, chunk_size)
    partial_max = torch.empty((rows, chunks), device=logits.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    candidate_values = torch.empty((rows, chunks, k), device=logits.device, dtype=torch.float32)
    candidate_indices = torch.empty((rows, chunks, k), device=logits.device, dtype=torch.int32)
    output_values = torch.empty((rows, k), device=logits.device, dtype=logits.dtype)
    output_indices = torch.empty((rows, k), device=logits.device, dtype=torch.int64)

    _chunk_logsumexp_topk[(rows, chunks)](
        logits,
        partial_max,
        partial_sum,
        candidate_values,
        candidate_indices,
        rows,
        cols,
        chunks,
        logits.stride(0),
        CHUNK=chunk_size,
        TOPK=k,
        num_warps=8,
    )
    candidates = triton.next_power_of_2(chunks * k)
    _merge_logsumexp_topk[(rows,)](
        partial_max,
        partial_sum,
        candidate_values,
        candidate_indices,
        output_values,
        output_indices,
        chunks=chunks,
        CANDIDATES=candidates,
        TOPK=k,
        num_warps=4,
    )
    return output_values, output_indices


__all__ = ["fused_logsoftmax_topk"]
