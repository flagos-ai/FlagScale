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

"""Attention QKV merge/split helpers shared across directions and model types."""

import torch


def merge_attention_qkv(q_proj, k_proj, v_proj, cfg):
    """Merge HF's separate q/k/v projections into Megatron's fused linear_qkv."""
    hidden = cfg.hidden_size
    num_qg = cfg.num_query_groups
    kv_ch = cfg.kv_channels
    heads_per_group = cfg.num_attention_heads // num_qg

    if cfg.attention_output_gate:
        # q_proj: (num_heads * 2 * kv_ch, hidden) -> (num_heads, 2, kv_ch, hidden)
        qz = q_proj.view(cfg.num_attention_heads, 2, kv_ch, hidden)
        q_heads = qz[:, 0]  # (num_heads, kv_ch, hidden)
        z_heads = qz[:, 1]  # (num_heads, kv_ch, hidden)

        # Group by query group: (num_qg, heads_per_group, kv_ch, hidden)
        q_grouped = q_heads.view(num_qg, heads_per_group, kv_ch, hidden)
        z_grouped = z_heads.view(num_qg, heads_per_group, kv_ch, hidden)
        k_grouped = k_proj.view(num_qg, 1, kv_ch, hidden)
        v_grouped = v_proj.view(num_qg, 1, kv_ch, hidden)

        qkv = torch.cat([q_grouped, z_grouped, k_grouped, v_grouped], dim=1)
    else:
        q_grouped = q_proj.view(num_qg, heads_per_group, kv_ch, hidden)
        k_grouped = k_proj.view(num_qg, 1, kv_ch, hidden)
        v_grouped = v_proj.view(num_qg, 1, kv_ch, hidden)
        qkv = torch.cat([q_grouped, k_grouped, v_grouped], dim=1)

    return qkv.view(-1, hidden)


def split_attention_qkv(linear_qkv, cfg):
    """Split Megatron fused linear_qkv back into HF's q/k/v projections."""
    hidden = cfg.hidden_size
    num_qg = cfg.num_query_groups
    kv_ch = cfg.kv_channels
    heads_per_group = cfg.num_attention_heads // num_qg

    if cfg.attention_output_gate:
        total_hpg = 2 * heads_per_group + 2
    else:
        total_hpg = heads_per_group + 2

    qkv = linear_qkv.view(num_qg, total_hpg, kv_ch, hidden)
    if cfg.attention_output_gate:
        q_heads = qkv[:, :heads_per_group]
        z_heads = qkv[:, heads_per_group : 2 * heads_per_group]
        k_heads = qkv[:, -2:-1]
        v_heads = qkv[:, -1:]
        q_flat = q_heads.reshape(-1, kv_ch, hidden)
        z_flat = z_heads.reshape(-1, kv_ch, hidden)
        q_proj = torch.cat([q_flat, z_flat], dim=1).reshape(-1, hidden)
    else:
        q, k_heads, v_heads = torch.split(qkv, [heads_per_group, 1, 1], dim=1)
        q_proj = q.reshape(-1, hidden)

    k_proj = k_heads.reshape(-1, hidden)
    v_proj = v_heads.reshape(-1, hidden)
    return q_proj, k_proj, v_proj
