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

"""TP/PP/EP sharding helpers for Megatron checkpoints."""

import os
import re

import torch


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
def _tp_rank_key(name):
    """Extract tp_rank from a Megatron state_dict key suffix like '_tp0'."""
    m = re.search(r"_tp(\d+)$", name)
    return int(m.group(1)) if m else None


def split_tp_column_parallel(weight, tp_size):
    """Split a 2-D weight along dim 0 for TP column-parallel shards."""
    return torch.chunk(weight, tp_size, dim=0)


def split_tp_row_parallel(weight, tp_size):
    """Split a 2-D weight along dim 1 for TP row-parallel shards."""
    return torch.chunk(weight, tp_size, dim=1)


def merge_tp_column_parallel(shards):
    """Merge TP column-parallel shards along dim 0."""
    return torch.cat(shards, dim=0)


def merge_tp_row_parallel(shards):
    """Merge TP row-parallel shards along dim 1."""
    return torch.cat(shards, dim=1)


# -----------------------------------------------------------------------------
# PP helpers
# -----------------------------------------------------------------------------
def split_pp_layers(state_dict, cfg):
    """Split a full state dict into per-PP-rank chunks.

    Megatron PP checkpoints use per-stage local layer indices: each PP rank
    stores its transformer layers as ``layers.0``, ``layers.1``, ... regardless
    of the global position.  Non-transformer tensors follow the Megatron
    placement convention:

    * First PP rank:  word embeddings and the full vision model.
    * Last PP rank:   final layernorm, output layer, and MTP modules.

    Returns a list of dicts indexed by pp_rank.
    """
    pp_size = cfg.pp
    if pp_size == 1:
        return [state_dict]

    layer_counts = cfg.pp_layer_counts
    # Precompute global offset for each PP rank
    offsets = [0]
    for c in layer_counts[:-1]:
        offsets.append(offsets[-1] + c)

    per_pp = [dict() for _ in range(pp_size)]

    for key, value in state_dict.items():
        # Vision model lives on the first PP rank and is not split.
        if key.startswith("vision_model."):
            per_pp[0][key] = value
            continue

        # Word embeddings are placed on the first and last PP ranks
        # (first for pre_process, last for output_layer weight tying).
        if key.startswith("language_model.embedding."):
            per_pp[0][key] = value
            per_pp[-1][key] = value
            continue

        # Final norm, output layer, and MTP live on the last PP rank.
        if (
            key.startswith("language_model.decoder.final_layernorm.")
            or key.startswith("language_model.output_layer.")
            or key.startswith("language_model.mtp.")
        ):
            per_pp[-1][key] = value
            continue

        # Transformer layers are partitioned and renamed to local indices.
        m = re.search(r"language_model\.decoder\.layers\.(\d+)\.", key)
        if m:
            layer_idx = int(m.group(1))
            # Find which PP rank this global layer belongs to
            pp_rank = pp_size - 1
            for r in range(pp_size):
                if layer_idx < offsets[r] + layer_counts[r]:
                    pp_rank = r
                    break
            local_idx = layer_idx - offsets[pp_rank]
            local_key = (
                key[: m.start()] + f"language_model.decoder.layers.{local_idx}." + key[m.end() :]
            )
            per_pp[pp_rank][local_key] = value
            continue

        # Any unexpected keys default to the last PP rank.
        per_pp[-1][key] = value

    return per_pp


def merge_pp_layers(pp_state_dicts, cfg):
    """Merge per-PP-rank state dicts into a single state_dict.

    Accepts either a dict mapping pp_rank -> state_dict or a list ordered by
    pp_rank.  Local transformer layer indices are converted back to global
    indices; other keys keep their original names.
    """
    merged = {}
    if isinstance(pp_state_dicts, dict):
        ranks = sorted(pp_state_dicts.keys())
    else:
        ranks = range(len(pp_state_dicts))

    layer_counts = cfg.pp_layer_counts
    # Precompute global offset for each PP rank
    offsets = [0]
    for c in layer_counts[:-1]:
        offsets.append(offsets[-1] + c)

    for r in ranks:
        sd = pp_state_dicts[r]
        global_offset = offsets[r]
        for key, value in sd.items():
            m = re.search(r"language_model\.decoder\.layers\.(\d+)\.", key)
            if not m:
                # Embeddings (replicated), vision, final norm, output layer, MTP, etc.
                # Keep the first occurrence; replicated tensors are identical across ranks.
                if key not in merged:
                    merged[key] = value
                continue
            local_idx = int(m.group(1))
            global_idx = global_offset + local_idx
            global_key = (
                key[: m.start()] + f"language_model.decoder.layers.{global_idx}." + key[m.end() :]
            )
            merged[global_key] = value
    return merged


# -----------------------------------------------------------------------------
# EP helpers
# -----------------------------------------------------------------------------
def split_ep_experts(state_dict, cfg):
    """Split full MoE expert weights per EP rank using contiguous blocks.

    Operates on already-TP-sharded expert weights. Returns a list of dicts
    indexed by ep_rank. Non-expert keys are replicated across EP ranks.
    """
    ep_size = cfg.ep
    if ep_size == 1:
        return [state_dict]

    experts_per_ep = cfg.num_experts // ep_size
    per_ep = [dict() for _ in range(ep_size)]
    expert_re = re.compile(r"^(.*\.mlp\.experts\.linear_fc[12]\.(?:weight|bias))(\d+)$")

    for key, value in state_dict.items():
        m = expert_re.match(key)
        if not m:
            for rank in range(ep_size):
                per_ep[rank][key] = value
            continue

        global_idx = int(m.group(2))
        ep_rank = global_idx // experts_per_ep
        local_idx = global_idx % experts_per_ep
        new_key = f"{m.group(1)}{local_idx}"
        per_ep[ep_rank][new_key] = value

    return per_ep


def merge_ep_experts(ep_state_dicts, cfg):
    """Merge per-EP-rank MoE expert weights into a single state_dict."""
    merged = {}
    experts_per_ep = cfg.num_experts // cfg.ep
    expert_re = re.compile(r"^(.*\.mlp\.experts\.linear_fc[12]\.(?:weight|bias))(\d+)$")

    for ep_rank, sd in enumerate(ep_state_dicts):
        for key, value in sd.items():
            m = expert_re.match(key)
            if not m:
                if key not in merged:
                    merged[key] = value
                continue

            local_idx = int(m.group(2))
            global_idx = ep_rank * experts_per_ep + local_idx
            new_key = f"{m.group(1)}{global_idx}"
            merged[new_key] = value

    return merged


# -----------------------------------------------------------------------------
# Megatron release checkpoint naming
# -----------------------------------------------------------------------------
def megatron_shard_path(save_dir, tp_rank=0, pp_rank=0, ep_rank=None, release=True):
    """Build path for a single Megatron checkpoint shard."""
    if release:
        if ep_rank is not None:
            rank_dir = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_ep_{ep_rank:02d}"
        else:
            rank_dir = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
        return os.path.join(save_dir, "release", rank_dir, "model_optim_rng.pt")
    else:
        raise NotImplementedError("Non-release checkpoint layout is not supported yet.")


def iter_release_shard_dirs(checkpoint_dir):
    """Yield (tp_rank, pp_rank, ep_rank, dir_path) for release checkpoint shards."""
    release_dir = os.path.join(checkpoint_dir, "release")
    if not os.path.isdir(release_dir):
        return

    pattern = re.compile(r"mp_rank_(\d+)_(\d+)(?:_ep_(\d+))?")
    for name in sorted(os.listdir(release_dir)):
        m = pattern.match(name)
        if not m:
            continue
        tp_rank = int(m.group(1))
        pp_rank = int(m.group(2))
        ep_rank = int(m.group(3)) if m.group(3) is not None else None
        yield tp_rank, pp_rank, ep_rank, os.path.join(release_dir, name)
