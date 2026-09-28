from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..adasparse_args import (
    get_height,
    get_width,
    get_frames,
    get_num_layers,
    get_num_steps,
    get_sparsity_modes,
    get_sparsity,
    get_block_size,
    get_search_steps,
    get_min_recall,
    get_enable_log,
)
from block_sparse_attn import block_sparse_attn_func
from .fused_fa2_search_kernel import _flash_attn_triton_search
from .fused_fa2_search_kernel_cache_lse import _flash_attn_triton_search_cache_lse
from ..adaspa_handler import get_model_name

import os
from pathlib import Path
from typing import List, Optional, Union


class AdaptiveSparseAttention(nn.Module):
    def forward(self, *args, **kwargs):
        return adaptive_sparse_attn(*args, **kwargs)


mask_cache = {}
lse_cache = {}
global_counter = 0


def get_step_and_layer():
    global global_counter
    num_layers = get_num_layers()
    num_steps = get_num_steps()
    step = global_counter // num_layers % num_steps
    layer = global_counter % num_layers
    global_counter += 1
    return step, layer


def truncate_padding(tensor, attn_mask):
    if attn_mask is None:
        return tensor

    mask = attn_mask.squeeze()

    true_indices = torch.where(mask)[0]
    if true_indices.numel() > 0:
        last_true_pos = true_indices[-1].item() + 1
    else:
        last_true_pos = 1

    return tensor[..., :last_true_pos, :]


def move_text_to_back(tensor, text_length):
    B, H, S, D = tensor.shape

    text_part = tensor[:, :, :text_length, :]
    video_part = tensor[:, :, text_length:, :]
    return torch.cat([video_part, text_part], dim=2)


def move_text_to_front(tensor, text_length):
    B, H, S, D = tensor.shape

    video_part = tensor[:, :, :-text_length, :]
    text_part = tensor[:, :, -text_length:, :]
    return torch.cat([text_part, video_part], dim=2)


def adaptive_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Computes sparse attention with optional masking and dropout.

    Args:
        query:  Query tensor of shape [batch_size, seq_len, num_heads, head_dim].
        key:    Key tensor of shape [batch_size, seq_len, num_heads, head_dim].
        value:  Value tensor of shape [batch_size, seq_len, num_heads, head_dim].
        attn_mask: Optional mask tensor of shape [batch_size, 1, seq_len, seq_len] 
                  where `-inf` indicates masked positions. Defaults to None.
        dropout_p: Dropout probability for attention weights. Defaults to 0.0.
        is_causal: If True, applies causal masking. Defaults to False.

    Returns:
        Output tensor of shape [batch_size, seq_len, num_heads, head_dim].
    """
    step, layer = get_step_and_layer()

    model_name = get_model_name()

    if step >= get_search_steps()[0]:
        if model_name in ("HunyuanVideo", "Wan2.1"):
            _, _, S_q, _ = query.shape
            query = truncate_padding(query, attn_mask)
            key = truncate_padding(key, attn_mask)
            value = truncate_padding(value, attn_mask)

        if model_name == "CogVideoX":
            query = move_text_to_back(query, 226)
            key = move_text_to_back(key, 226)
            value = move_text_to_back(value, 226)

        B, H, S, D = query.shape
        device = query.device
        cu_seqlens = torch.arange(0, (B + 1) * S, step=S, dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(0, (B + 1) * S, step=S, dtype=torch.int32, device=device)
        head_mask_type = torch.tensor([1] * H, dtype=torch.int32, device=device)

        if "cache_lse" in get_sparsity_modes():
            if step == get_search_steps()[0]:
                block_mask, lse = search_block_mask(
                    query,
                    key,
                    value,
                    sparsity=get_sparsity(),
                    recall=get_min_recall(),
                    sparsity_modes=get_sparsity_modes(),
                )
                lse_cache[layer] = lse

            elif step in get_search_steps():
                block_mask, _ = search_block_mask(
                    query,
                    key,
                    value,
                    cached_lse=lse_cache[layer],
                    sparsity=get_sparsity(),
                    recall=get_min_recall(),
                    sparsity_modes=get_sparsity_modes(),
                )

        else:
            if step in get_search_steps():
                block_mask, _ = search_block_mask(
                    query,
                    key,
                    value,
                    sparsity=get_sparsity(),
                    recall=get_min_recall(),
                    sparsity_modes=get_sparsity_modes(),
                )

        if step in get_search_steps():
            mask_cache[layer] = block_mask

            if get_enable_log():
                print(f"real_sparsity = {1 - block_mask.sum().item() / block_mask.numel()}")

        block_mask = mask_cache[layer]
        block_size = get_block_size()
        expected_blocks = (S + block_size - 1) // block_size

        if (
            block_mask.ndim != 4
            or block_mask.shape[0] != B
            or block_mask.shape[2] != expected_blocks
            or block_mask.shape[3] != expected_blocks
        ):
            raise RuntimeError(
                "Invalid block_mask shape before block_sparse_attn_func: "
                f"got {tuple(block_mask.shape)}, expected ({B}, H_or_subset, {expected_blocks}, {expected_blocks})"
            )

        query = query.permute(0, 2, 1, 3).reshape((B * S, H, -1))
        key = key.permute(0, 2, 1, 3).reshape((B * S, H, -1))
        value = value.permute(0, 2, 1, 3).reshape((B * S, H, -1))
        out = block_sparse_attn_func(
                query, key, value,
                cu_seqlens, cu_seqlens,
                head_mask_type,
                None,
                block_mask,
                S, S,
                dropout_p,
                deterministic=False,
                softmax_scale=None,
                is_causal=is_causal,
                exact_streaming=False,
                return_attn_probs=False,
                # sparse_block_size=get_block_size(),
            )

        if model_name in ("HunyuanVideo", "Wan2.1"):
            truncated_out = out
            out = torch.zeros(B, H, S_q, D, dtype=query.dtype, device=query.device)
            out[:, :, :S, :] = truncated_out.reshape(B, S, H, -1).permute(0, 2, 1, 3)

        else:
            out = out.reshape(B, S, H, -1).permute(0, 2, 1, 3)

        if model_name == "CogVideoX":
            out = move_text_to_front(out, 226)

    else:
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    return out


def search_block_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cached_lse: Optional[torch.Tensor] = None,
    sparsity: float = 0.8,
    recall: float = 0.9,
    sparsity_modes: List[str] = [
        "random_select",
        "head_adaptive",
        "cache_lse",
        "row_wise",
        "text_sink",
        "first_frame_sink",
    ],
) -> torch.Tensor:
    B, H, S, D = query.shape

    block_size = get_block_size()
    fa_block_size = get_block_size()

    # ====== 1. Compute num_blocks and padding ====== #
    num_blocks_final = (S + block_size - 1) // block_size
    pad_len = (fa_block_size - (S % fa_block_size)) % fa_block_size

    # ====== 2. Pad the S dimension ====== #
    if pad_len > 0:
        pad_shape = (B, H, pad_len, D)
        q_pad = torch.zeros(pad_shape, dtype=query.dtype, device=query.device)
        k_pad = torch.zeros(pad_shape, dtype=key.dtype, device=key.device)
        v_pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)

        query_padded = torch.cat([query, q_pad], dim=2)
        key_padded   = torch.cat([key,   k_pad], dim=2)
        value_padded = torch.cat([value, v_pad], dim=2)

    else:
        query_padded = query
        key_padded   = key
        value_padded = value

    # ====== 3. Call search pattern kernel, return (attn_output, lse_now, block_sums) ====== #
    if cached_lse is not None:
        _, lse_now, block_sums = _flash_attn_triton_search_cache_lse(
            query_padded, key_padded, value_padded,
            cached_lse,
            causal=False, sm_scale=None,
            BLOCK_M=fa_block_size, BLOCK_N=fa_block_size,
            sparse_block_size=block_size
        )

    else:
        _, lse_now, block_sums = _flash_attn_triton_search(
            query_padded, key_padded, value_padded,
            causal=False, sm_scale=None,
            BLOCK_M=fa_block_size, BLOCK_N=fa_block_size,
            sparse_block_size=block_size
        )

    # ====== 4. Adjust block_sums size according to actual block count M ====== #
    M = num_blocks_final
    if block_sums.shape[2] != M:
        block_sums = block_sums[:, :, :M, :M].contiguous()  # [B, H, M, M]

    def row_wise_topk_mask_all(block_sums_3d, local_sparsity):
        """
        For all heads [BH, M, M], perform row-wise top-k and return a bool mask [BH, M, M].
        Each row keeps round((1.0 - local_sparsity) * M) blocks.
        """
        BH_, M_, _ = block_sums_3d.shape
        k_per_row = round((1.0 - local_sparsity) * M_)
        k_per_row = max(k_per_row, 0)
        k_per_row = min(k_per_row, M_)

        mask_all = torch.zeros((BH_, M_, M_), dtype=torch.bool, device=block_sums_3d.device)
        if k_per_row == 0:
            return mask_all

        if "row_wise" in sparsity_modes:
            # Perform top-k on dim=-1 in one shot
            _, top_idx = block_sums_3d.topk(k_per_row, dim=-1)  # [BH_, M_, k_per_row]
            # scatter
            src = torch.ones_like(top_idx, dtype=torch.bool)
            mask_all.scatter_(dim=-1, index=top_idx, src=src)
        else:
            k = round((1.0 - local_sparsity) * M_ * M_)
            k = max(k, 0)
            k = min(k, M_ * M_)

            # Find top-k over the flattened [BH_, M_ * M_] dimension
            _, top_idx = block_sums_3d.view(BH_, -1).topk(k, dim=-1)  # [BH_, k]

            # Efficiently compute row/col indices with PyTorch
            row_idx = torch.div(top_idx, M_, rounding_mode='floor')  # row index
            col_idx = torch.remainder(top_idx, M_)  # column index

            # Construct mask
            mask_all[torch.arange(BH_)[:, None], row_idx, col_idx] = True

        return mask_all

    def compute_recall_for_sparsity_all(block_sums_3d, sum_all_1d, local_sparsity):
        """
        For all heads [BH, M, M], perform row-wise top-k, compute recall (chosen_sum / sum_all),
        and return a recall vector of shape [BH].
        """
        BH_, M_, _ = block_sums_3d.shape
        k_per_row = round((1.0 - local_sparsity) * M_)
        k_per_row = max(k_per_row, 0)
        k_per_row = min(k_per_row, M_)

        if k_per_row == 0:
            return torch.zeros(BH_, dtype=torch.float, device=block_sums_3d.device)

        if "row_wise" in sparsity_modes:
            # [BH_, M_, k_per_row]
            topvals, _ = block_sums_3d.topk(k_per_row, dim=-1)
            chosen_sum_1d = topvals.sum(dim=-1).sum(dim=-1)  # [BH_]
        else:
            k = round((1.0 - local_sparsity) * M_ * M_)
            k = max(k, 0)
            k = min(k, M_ * M_)

            # Find top-k over the flattened [BH_, M_ * M_] dimension
            topvals, _ = block_sums_3d.view(BH_, -1).topk(k, dim=-1)  # [BH_, k]
            chosen_sum_1d = topvals.sum(dim=-1)  # [BH_]

        recall_1d = chosen_sum_1d / (sum_all_1d + 1e-9)

        return recall_1d

    # ====== 6. Three-group head strategy: s_up, s_down, sparsity, ensuring overall avg sparsity = sparsity ====== #
    if "head_adaptive" in sparsity_modes:
        # Unfold to [BH, M, M]
        BH = B * H
        block_sums_3d = block_sums.view(BH, M, M)
        # Compute total score for each head
        sum_all_1d = block_sums_3d.sum(dim=-1).sum(dim=-1)  # [BH]

        # Define three sparsities
        s_up = (1.0 + sparsity) / 2.0
        s_down = (3.0 * sparsity - 1.0) / 2.0
        # If needed, clamp to [0,1]
        # s_up = max(0.0, min(1.0, s_up))
        # s_down = max(0.0, min(1.0, s_down))

        # 1) Compute recall_up for all heads
        recall_up = compute_recall_for_sparsity_all(block_sums_3d, sum_all_1d, s_up)
        # Find heads with recall >= target
        up_candidates = torch.nonzero((recall_up >= recall), as_tuple=True)[0]  # 1D indices
        # Sort by recall_up descending
        up_cand_vals = recall_up[up_candidates]
        sorted_ids_up = torch.argsort(up_cand_vals, descending=True)
        # max_n = size of upper group, can't exceed up_candidates count nor BH//2
        raw_n = len(up_candidates)  # number of heads >= recall
        max_n_up = min(raw_n, BH // 2)
        heads_up = up_candidates[sorted_ids_up[:max_n_up]]

        # 2) From leftovers, pick s_down
        leftover_mask_up = torch.ones(BH, dtype=torch.bool, device=block_sums.device)
        leftover_mask_up[heads_up] = False
        leftover_indices_up = leftover_mask_up.nonzero(as_tuple=True)[0]  # indices

        recall_down = compute_recall_for_sparsity_all(block_sums_3d, sum_all_1d, s_down)
        leftover_recall_down = recall_down[leftover_indices_up]
        # Sort ascending (smaller first)
        sorted_ids_down = torch.argsort(leftover_recall_down, descending=False)
        # Also choose max_n_up
        max_n_down = min(len(sorted_ids_down), max_n_up)
        heads_down = leftover_indices_up[sorted_ids_down[:max_n_down]]

        # 3) Remaining heads use mid (i.e., original sparsity)
        leftover_mask_down = leftover_mask_up.clone()
        leftover_mask_down[heads_down] = False
        heads_mid = leftover_mask_down.nonzero(as_tuple=True)[0]

        # Compute three masks
        mask_up_all = row_wise_topk_mask_all(block_sums_3d, s_up)
        mask_down_all = row_wise_topk_mask_all(block_sums_3d, s_down)
        mask_mid_all = row_wise_topk_mask_all(block_sums_3d, sparsity)

        # Merge to final mask
        block_mask_3d_final = torch.zeros_like(mask_up_all, dtype=torch.bool)  # [BH, M, M]
        block_mask_3d_final[heads_up]   = mask_up_all[heads_up]
        block_mask_3d_final[heads_down] = mask_down_all[heads_down]
        block_mask_3d_final[heads_mid]  = mask_mid_all[heads_mid]

        # ====== 7. Restore to [B, H, M, M] & set last row/col to True ====== #
        block_mask = block_mask_3d_final.view(B, H, M, M)

    else:
        k = round(M * M * (1 - sparsity))  # proportion to keep
        # Flatten and select top-k in each [b, h]
        block_sums_2d = block_sums.view(B * H, -1)  # [B*H, M*M]

        # Get top-k indices in each row
        _, topk_indices = torch.topk(block_sums_2d, k, dim=-1, largest=True)

        # Create an all-zero mask then set top-k positions to True
        block_mask_2d = torch.zeros_like(block_sums_2d, dtype=torch.bool)  # [B*H, M*M]
        row_idx = torch.arange(B * H, device=block_sums_2d.device).unsqueeze(-1)  # [B*H, 1]
        block_mask_2d[row_idx, topk_indices] = True

        # Restore to [B, H, M, M]
        block_mask = block_mask_2d.view(B, H, M, M)

    temporal_dict = {
        "HunyuanVideo": 4,
        "CogVideoX": 8,
        "Wan2.1": 4,
    }
    spatial_dict = {
        "HunyuanVideo": 16,
        "CogVideoX": 16,
        "Wan2.1": 16,
    }

    model_name = get_model_name()
    if model_name not in temporal_dict or model_name not in spatial_dict:
        model_name = "Wan2.1"

    frame_len = ((get_height() // spatial_dict[model_name]) *
                    (get_width() // spatial_dict[model_name]))
    frame_num = (get_frames() - 1) // temporal_dict[model_name] + 1

    if model_name == "CogVideoX":
        frame_num *= 2

    if "first_frame_sink" in sparsity_modes:
        frame_block_len = (frame_len - 1) // block_size + 1
        block_mask[..., : frame_block_len, :] = True
        block_mask[..., :, : frame_block_len] = True

    if "text_sink" in sparsity_modes:
        video_len = frame_len * frame_num
        video_block_len = video_len // block_size
        block_mask[..., video_block_len : , :] = True
        block_mask[..., :, video_block_len : ] = True

    # ====== 8. Randomly set some non-True elements to True (example) ====== #
    if "random_select" in sparsity_modes:
        random_ratio = 0.01
        num_false_elements = (~block_mask).sum().item()
        num_random = min(num_false_elements, int(block_mask.numel() * random_ratio))

        false_indices = (~block_mask).nonzero(as_tuple=False)  # [num_false_elements, 4]
        random_perm = torch.randperm(num_false_elements, device=block_mask.device)
        random_indices = false_indices[random_perm[:num_random]]
        block_mask[random_indices.split(1, dim=-1)] = True

    if "head_adaptive" in sparsity_modes:
        # ====== 9. Report head counts for each sparsity and overall avg sparsity ====== #
        heads_up_count = heads_up.shape[0]
        heads_down_count = heads_down.shape[0]
        heads_mid_count = heads_mid.shape[0]

        total_blocks = block_mask.numel()
        true_blocks = block_mask.sum().item()
        avg_true_ratio = true_blocks / total_blocks if total_blocks > 0 else 0.0

        # ====== 10. Compute and report final recall for each (batch, head) ====== #
        chosen_sum_3d = torch.where(block_mask_3d_final, block_sums_3d, torch.zeros_like(block_sums_3d))
        chosen_sum_1d = chosen_sum_3d.sum(dim=-1).sum(dim=-1)  # [BH]
        final_recall_1d = chosen_sum_1d / (sum_all_1d + 1e-9)
        final_recall_2d = final_recall_1d.view(B, H)

        if get_enable_log():
            print(f"[Info] Totally {BH} (batch, head): ")
            print(f"   - The number of heads using s_up={s_up:.4f}: {heads_up_count}")
            print(f"   - The number of heads using  s_down={s_down:.4f}: {heads_down_count}")
            print(f"   - The number of heads using  sparsity={sparsity:.4f}: {heads_mid_count}")
            print(f"[Info] Overall average sparsity (true block proportion) = {avg_true_ratio:.4f}")
            print(f"[Info] Every (batch, head)'s final recall: ")
            for b_idx in range(B):
                for h_idx in range(H):
                    val = final_recall_2d[b_idx, h_idx].item()
                    print(f"  - (batch={b_idx}, head={h_idx}), recall={val:.4f}")

    return block_mask, lse_now