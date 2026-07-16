"""Checkpoint I/O helpers for HF and Megatron formats."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_hf_weights(hf_dir):
    """Load all HF weights from safetensors files in the directory."""
    sd = {}
    for st_file in sorted(Path(hf_dir).glob("*.safetensors")):
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    return sd


def load_hf_config(hf_dir):
    """Load HF config.json."""
    config_path = Path(hf_dir) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def load_megatron_shard(path):
    """Load a single Megatron shard and return its 'model' state dict."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("model", sd)


def find_megatron_shard(meg_dir, tp_rank=0, pp_rank=0, ep_rank=None):
    """Find a single Megatron shard file under release/ or iter_*/ subdirs."""
    candidates = [meg_dir]
    release_dir = os.path.join(meg_dir, "release")
    if os.path.isdir(release_dir):
        candidates.append(release_dir)

    iter_dirs = sorted(
        d
        for d in os.listdir(meg_dir)
        if d.startswith("iter_") and os.path.isdir(os.path.join(meg_dir, d))
    )
    for d in iter_dirs:
        candidates.append(os.path.join(meg_dir, d))
        iter_release = os.path.join(meg_dir, d, "release")
        if os.path.isdir(iter_release):
            candidates.append(iter_release)

    ep_rank = ep_rank if ep_rank is not None else 0
    suffixes = [
        # Standard PP>1 and EP>1 naming: tp_pp_ep
        f"{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}",
    ]
    # PP>1 EP=1 naming (only valid when ep_rank==0, otherwise ambiguous)
    if ep_rank == 0:
        suffixes.append(f"{tp_rank:02d}_{pp_rank:03d}")
    # PP=1 EP>1 naming: tp_ep
    suffixes.append(f"{tp_rank:02d}_{ep_rank:03d}")
    # PP=1 EP=1 naming (only valid when both are 0)
    if pp_rank == 0 and ep_rank == 0:
        suffixes.append(f"{tp_rank:02d}")

    for base in candidates:
        for suffix in suffixes:
            path = os.path.join(base, f"mp_rank_{suffix}", "model_optim_rng.pt")
            if os.path.exists(path):
                return path
    return None


def save_megatron_release_checkpoint(shards_dict, save_dir, cfg):
    """Save a {(pp, tp, ep?): state_dict} mapping as a Megatron release checkpoint."""
    release_dir = os.path.join(save_dir, "release")
    os.makedirs(release_dir, exist_ok=True)

    for rank_tuple, shard in shards_dict.items():
        # Support both (pp, tp) and (pp, tp, ep) tuple layouts regardless of cfg.ep.
        if len(rank_tuple) == 3:
            pp_rank, tp_rank, ep_rank = rank_tuple
        else:
            pp_rank, tp_rank = rank_tuple
            ep_rank = None
        use_ep = ep_rank is not None and ep_rank >= 0 and cfg.ep > 1

        name = f"mp_rank_{tp_rank:02d}"
        if cfg.pp > 1:
            name = f"{name}_{pp_rank:03d}"
        if use_ep:
            name = f"{name}_{ep_rank:03d}"

        ckpt_dir = os.path.join(release_dir, name)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, "model_optim_rng.pt")
        torch.save({"model": shard}, save_path)

    tracker_path = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    with open(tracker_path, "w") as f:
        f.write("release\n")

    return release_dir


def ensure_hf_path(hf_path):
    """Return a local HF checkpoint path.

    If ``hf_path`` is an existing directory containing safetensors files, return
    it as-is. Otherwise treat it as a ModelScope model ID and download the
    checkpoint (only supported for hf2meg).
    """
    if os.path.isdir(hf_path) and any(Path(hf_path).glob("*.safetensors")):
        return hf_path

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "modelscope is not installed. Please install it (pip install modelscope) "
            "or provide a local HF checkpoint directory via --hf-path."
        ) from exc

    print(f"HF path not found locally; downloading from ModelScope: {hf_path}")
    local_dir = snapshot_download(hf_path)
    print(f"Downloaded to: {local_dir}")
    return local_dir


def save_hf_checkpoint(hf_sd, save_dir, filename="model.safetensors"):
    """Save a HF state dict as safetensors."""
    os.makedirs(save_dir, exist_ok=True)
    save_file(hf_sd, os.path.join(save_dir, filename))
    return os.path.join(save_dir, filename)
