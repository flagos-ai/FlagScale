"""Pure tensor transformations used by the BAGEL checkpoint converter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionLayout:
    """Architecture values required to pack separate Q/K/V projections."""

    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int | None = None

    @property
    def head_size(self) -> int:
        return self.kv_channels or self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.num_attention_heads <= 0 or self.num_query_groups <= 0:
            raise ValueError("attention head and query-group counts must be positive")
        if self.num_attention_heads % self.num_query_groups:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} is not divisible by "
                f"num_query_groups={self.num_query_groups}"
            )
        if self.kv_channels is None and self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size={self.hidden_size} is not divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: AttentionLayout) -> torch.Tensor:
    """Pack external Q/K/V into MCore interleaved GQA order."""

    layout.validate()
    if q.ndim not in (1, 2) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError(f"Q/K/V must all be rank 1 or rank 2, got {q.ndim}, {k.ndim}, {v.ndim}")
    if k.shape != v.shape:
        raise ValueError(f"K and V shapes differ: {tuple(k.shape)} != {tuple(v.shape)}")
    if q.ndim == 2 and (q.shape[1] != layout.hidden_size or k.shape[1] != layout.hidden_size):
        raise ValueError(
            f"Q/K/V input dimension must be hidden_size={layout.hidden_size}: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    head_size = layout.head_size
    q_expected = layout.num_attention_heads * head_size
    kv_expected = layout.num_query_groups * head_size
    if q.shape[0] != q_expected or k.shape[0] != kv_expected:
        raise ValueError(
            f"unexpected Q/K/V output dimensions: expected q={q_expected}, kv={kv_expected}; "
            f"got q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )

    tail = () if q.ndim == 1 else (layout.hidden_size,)
    q_heads = q.reshape(layout.num_attention_heads, head_size, *tail)
    k_heads = k.reshape(layout.num_query_groups, head_size, *tail)
    v_heads = v.reshape(layout.num_query_groups, head_size, *tail)
    heads_per_group = layout.num_attention_heads // layout.num_query_groups
    packed = []
    for group in range(layout.num_query_groups):
        start = group * heads_per_group
        packed.extend(
            [
                q_heads[start : start + heads_per_group],
                k_heads[group : group + 1],
                v_heads[group : group + 1],
            ]
        )
    return torch.cat(packed, dim=0).reshape(-1, *tail)


def merge_gated_mlp(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Pack gate/up projections as the MCore ``[gate; up]`` FC1 tensor."""

    if gate.shape != up.shape:
        raise ValueError(f"gate and up shapes differ: {tuple(gate.shape)} != {tuple(up.shape)}")
    if gate.ndim not in (1, 2):
        raise ValueError(f"gate/up tensors must be rank 1 or rank 2, got {gate.ndim}")
    return torch.cat((gate, up), dim=0)

