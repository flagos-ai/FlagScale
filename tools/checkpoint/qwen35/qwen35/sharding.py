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
    """Split a full LLM state_dict into per-PP-rank chunks.

    Returns a list of dicts indexed by pp_rank.
    """
    num_layers = cfg.num_layers
    pp_size = cfg.pp
    if pp_size == 1:
        return [state_dict]

    layers_per_rank = num_layers // pp_size
    per_pp = [dict() for _ in range(pp_size)]

    for key, value in state_dict.items():
        m = re.search(r"\.layers\.(\d+)\.", key)
        if not m:
            # Non-layer keys (embeddings/final norm/MTP) go to the last PP rank by convention.
            per_pp[-1][key] = value
            continue

        layer_idx = int(m.group(1))
        pp_rank = layer_idx // layers_per_rank
        if pp_rank >= pp_size:
            pp_rank = pp_size - 1
        per_pp[pp_rank][key] = value

    return per_pp


def merge_pp_layers(pp_state_dicts, cfg):
    """Merge per-PP-rank state dicts into a single state_dict.

    Accepts either a dict mapping pp_rank -> state_dict or a list ordered by
    pp_rank.
    """
    merged = {}
    if isinstance(pp_state_dicts, dict):
        ranks = sorted(pp_state_dicts.keys())
    else:
        ranks = range(len(pp_state_dicts))
    for r in ranks:
        merged.update(pp_state_dicts[r])
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
