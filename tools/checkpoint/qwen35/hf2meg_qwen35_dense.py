#!/usr/bin/env python3
"""
Convert HuggingFace checkpoint to Megatron TP/PP format for Qwen3.5.

Usage:
    python hf2meg_qwen35_dense.py \
        --yaml /path/to/.yaml \
        --hf-dir /path/to/hf/qwen35 \
        --save-dir /path/to/output \
        [--ref-ckpt-dir /path/to/ref/megatron/checkpoint]
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

# Layer norm adjustment for zero-centered gamma.
# Controls LLM and MTP layernorms only. Vision layernorms are never adjusted.
# Enabled by default; use --no-adjust-ln to disable.
LN_ADJUSTMENT = True


def adjust_ln_weight(weight):
    """Apply layer norm adjustment (subtract 1.0) for zero-centered gamma."""
    return weight - 1.0 if LN_ADJUSTMENT else weight


# ============================================================
# Config
# ============================================================


class Config:
    """Flat config built from training yaml."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        system = raw.get("system", {})
        model = raw.get("model", {})

        self.tp = system.get("tensor_model_parallel_size", 1)
        self.pp = system.get("pipeline_model_parallel_size", 1)

        self.num_layers = model["num_layers"]
        self.hidden_size = model["hidden_size"]
        self.ffn_hidden_size = model["ffn_hidden_size"]
        self.num_attention_heads = model["num_attention_heads"]
        self.num_query_groups = model["num_query_groups"]
        self.kv_channels = model["kv_channels"]
        self.untie = model.get("untie_embeddings_and_output_weights", False)
        self.attention_output_gate = model.get("attention_output_gate", False)

        self.linear_attention_freq = model.get("linear_attention_freq", 4)
        self.linear_key_head_dim = model.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = model.get("linear_value_head_dim", 128)
        self.linear_num_key_heads = model.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = model.get("linear_num_value_heads", 16)
        self.qk_dim = self.linear_key_head_dim * self.linear_num_key_heads
        self.v_dim = self.linear_value_head_dim * self.linear_num_value_heads

        self.vision_num_layers = model.get("vision_num_layers", 24)
        self.vision_hidden_size = model.get("vision_hidden_size", 1024)
        self.vision_num_attention_heads = model.get("vision_num_attention_heads", 16)
        self.vision_ffn_hidden_size = model.get("vision_ffn_hidden_size", 4096)

        self.patch_size = model.get("patch_size", 16)
        self.temporal_patch_size = 2  # hardcoded in get_vision_model_config
        self.use_linear_proj = model.get("vision_patch_embed_linear", False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to training yaml config")
    p.add_argument("--hf-dir", required=True, help="Path to HF checkpoint directory")
    p.add_argument(
        "--save-dir",
        required=True,
        help="Output base directory (e.g. xxx/checkpoints). "
        "Script creates xxx/checkpoints/release/mp_rank_*/ automatically.",
    )
    p.add_argument(
        "--ref-ckpt-dir",
        default=None,
        help="Reference Megatron checkpoint dir to match embedding shape (optional)",
    )
    p.add_argument(
        "--adjust-embedding",
        action="store_true",
        help="Adjust embedding vocab size to match reference checkpoint "
        "(default: keep HF original size)",
    )
    p.add_argument(
        "--no-adjust-ln",
        action="store_true",
        help="Disable layer norm adjustment (zero-centered gamma)",
    )
    return p.parse_args()


# ============================================================
# Load HF weights
# ============================================================


def load_hf_weights(hf_dir):
    """Load all HF weights from safetensors files."""
    hf_dir = Path(hf_dir)
    sd = {}
    for st_file in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
    return sd


def load_hf_config(hf_dir):
    """Load HF config.json."""
    with open(Path(hf_dir) / "config.json") as f:
        return json.load(f)


# ============================================================
# Convert HF -> Megatron (full, unsharded)
# ============================================================


def is_gdn_layer(idx, freq):
    return (idx % freq) != (freq - 1)


def merge_gdn_in_proj(qkv, z, b, a, cfg):
    """Merge HF's separate qkv, z, b, a into Megatron's fused in_proj.

    Megatron expects TP-rank-grouped layout for ColumnParallel TP sharding:
        [tp0_q, tp0_k, tp0_v, tp0_z, tp0_b, tp0_a,
         tp1_q, tp1_k, tp1_v, tp1_z, tp1_b, tp1_a, ...]

    Each TP rank gets a contiguous block: [rank_q, rank_k, rank_v, rank_z, rank_b, rank_a]
    which matches the forward split: qkv, gate, beta, alpha.
    """
    hidden = cfg.hidden_size
    qk_head_dim = cfg.linear_key_head_dim
    v_head_dim = cfg.linear_value_head_dim
    num_qk_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads
    tp_size = cfg.tp

    # Split flat QKV into Q, K, V
    q_flat, k_flat, v_flat = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)

    # Reshape to (num_qk_heads, per_head_dim, hidden)
    q_g = q_flat.reshape(num_qk_heads, qk_head_dim, hidden)
    k_g = k_flat.reshape(num_qk_heads, qk_head_dim, hidden)
    v_g = v_flat.reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    z_g = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    b_g = b.reshape(num_qk_heads, v_per_group, hidden)
    a_g = a.reshape(num_qk_heads, v_per_group, hidden)

    # Reorder to TP-rank-grouped layout
    q, k, v, z, b, a = [w.reshape(tp_size, -1, hidden) for w in [q_g, k_g, v_g, z_g, b_g, a_g]]
    in_proj = torch.cat([q, k, v, z, b, a], dim=1).reshape(-1, hidden)
    return in_proj


def merge_attention_qkv(q_proj, k_proj, v_proj, cfg):
    """Merge HF's separate q/k/v projections into Megatron's fused linear_qkv.

    Megatron order (per group, attention_output_gate=True):
        [q*hpg, z*hpg, k, v]
    """
    hidden = cfg.hidden_size
    num_qg = cfg.num_query_groups
    kv_ch = cfg.kv_channels
    heads_per_group = cfg.num_attention_heads // num_qg

    if cfg.attention_output_gate:
        # q_proj contains Q+Z interleaved: (num_qg * 2*hpg * kv_ch, hidden)
        q_combined = q_proj.view(num_qg, 2 * heads_per_group, kv_ch, hidden)
        q_heads = q_combined[:, :heads_per_group]
        z_heads = q_combined[:, heads_per_group:]
        k_heads = k_proj.view(num_qg, 1, kv_ch, hidden)
        v_heads = v_proj.view(num_qg, 1, kv_ch, hidden)
        qkv = torch.cat([q_heads, z_heads, k_heads, v_heads], dim=1)
    else:
        q_heads = q_proj.view(num_qg, heads_per_group, kv_ch, hidden)
        k_heads = k_proj.view(num_qg, 1, kv_ch, hidden)
        v_heads = v_proj.view(num_qg, 1, kv_ch, hidden)
        qkv = torch.cat([q_heads, k_heads, v_heads], dim=1)

    return qkv.view(-1, hidden)


def convert_llm(hf_sd, meg_sd, cfg):
    """Convert LLM parameters from HF to Megatron naming."""
    freq = cfg.linear_attention_freq

    # Embedding
    hf_key = "model.language_model.embed_tokens.weight"
    if hf_key in hf_sd:
        meg_sd["language_model.embedding.word_embeddings.weight"] = hf_sd[hf_key]

    # LLM layers
    for layer_idx in range(cfg.num_layers):
        hf_pfx = f"model.language_model.layers.{layer_idx}"
        mg_pfx = f"language_model.decoder.layers.{layer_idx}"

        if is_gdn_layer(layer_idx, freq):
            # --- GDN layer ---
            mk = f"{mg_pfx}.self_attention.in_proj.layer_norm_weight"
            hk = f"{hf_pfx}.input_layernorm.weight"
            if hk in hf_sd:
                meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

            # in_proj: merge qkv, z, b, a
            mk = f"{mg_pfx}.self_attention.in_proj.weight"
            qkv = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_qkv.weight")
            z = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_z.weight")
            b = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_b.weight")
            a = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_a.weight")
            if qkv is not None and z is not None and b is not None and a is not None:
                meg_sd[mk] = merge_gdn_in_proj(qkv, z, b, a, cfg)

            # conv1d
            mk = f"{mg_pfx}.self_attention.conv1d.weight"
            hk = f"{hf_pfx}.linear_attn.conv1d.weight"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

            # out_proj
            mk = f"{mg_pfx}.self_attention.out_proj.weight"
            hk = f"{hf_pfx}.linear_attn.out_proj.weight"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

            mk = f"{mg_pfx}.self_attention.out_norm.weight"
            hk = f"{hf_pfx}.linear_attn.norm.weight"
            if hk in hf_sd:
                meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

            # A_log, dt_bias
            for suffix in ["A_log", "dt_bias"]:
                mk = f"{mg_pfx}.self_attention.{suffix}"
                hk = f"{hf_pfx}.linear_attn.{suffix}"
                if hk in hf_sd:
                    meg_sd[mk] = hf_sd[hk]

        else:
            # --- Standard Attention layer ---
            mk = f"{mg_pfx}.self_attention.linear_qkv.layer_norm_weight"
            hk = f"{hf_pfx}.input_layernorm.weight"
            if hk in hf_sd:
                meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

            # QKV merge
            mk = f"{mg_pfx}.self_attention.linear_qkv.weight"
            q = hf_sd.get(f"{hf_pfx}.self_attn.q_proj.weight")
            k = hf_sd.get(f"{hf_pfx}.self_attn.k_proj.weight")
            v = hf_sd.get(f"{hf_pfx}.self_attn.v_proj.weight")
            if q is not None and k is not None and v is not None:
                meg_sd[mk] = merge_attention_qkv(q, k, v, cfg)

            # o_proj
            mk = f"{mg_pfx}.self_attention.linear_proj.weight"
            hk = f"{hf_pfx}.self_attn.o_proj.weight"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

            # q/k norm
            for suffix in ["q_layernorm", "k_layernorm"]:
                hf_sfx = suffix.replace("layernorm", "norm")
                mk = f"{mg_pfx}.self_attention.{suffix}.weight"
                hk = f"{hf_pfx}.self_attn.{hf_sfx}.weight"
                if hk in hf_sd:
                    meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

        # --- MLP (same for both layer types) ---
        mk = f"{mg_pfx}.mlp.linear_fc1.layer_norm_weight"
        hk = f"{hf_pfx}.post_attention_layernorm.weight"
        if hk in hf_sd:
            meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

        # gate_proj + up_proj -> linear_fc1
        mk = f"{mg_pfx}.mlp.linear_fc1.weight"
        gate = hf_sd.get(f"{hf_pfx}.mlp.gate_proj.weight")
        up = hf_sd.get(f"{hf_pfx}.mlp.up_proj.weight")
        if gate is not None and up is not None:
            meg_sd[mk] = torch.cat([gate, up], dim=0)

        # down_proj
        mk = f"{mg_pfx}.mlp.linear_fc2.weight"
        hk = f"{hf_pfx}.mlp.down_proj.weight"
        if hk in hf_sd:
            meg_sd[mk] = hf_sd[hk]

    # Final layernorm
    mk = "language_model.decoder.final_layernorm.weight"
    hk = "model.language_model.norm.weight"
    if hk in hf_sd:
        meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

    # Output layer (only if untied)
    if cfg.untie:
        mk = "language_model.output_layer.weight"
        hk = "lm_head.weight"
        if hk in hf_sd:
            meg_sd[mk] = hf_sd[hk]


def convert_vision(hf_sd, meg_sd, cfg):
    """Convert vision model parameters from HF to Megatron."""
    mg_pfx = "vision_model"
    hf_pfx = "model.visual"
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads  # MHA

    # patch_embed
    for s in ["weight", "bias"]:
        hk = f"{hf_pfx}.patch_embed.proj.{s}"
        mk = f"{mg_pfx}.patch_embed.proj.{s}"
        if hk in hf_sd:
            val = hf_sd[hk]
            if s == "weight" and cfg.use_linear_proj:
                # HF uses nn.Conv3d, Megatron uses nn.Linear: flatten spatial dims
                val = val.view(cfg.vision_hidden_size, -1)
            meg_sd[mk] = val

    # pos_embed
    hk = f"{hf_pfx}.pos_embed.weight"
    mk = f"{mg_pfx}.pos_embed.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

    # Vision blocks
    for i in range(cfg.vision_num_layers):
        hf_blk = f"{hf_pfx}.blocks.{i}"
        mg_blk = f"{mg_pfx}.decoder.layers.{i}"

        # norm1 (from linear_qkv layernorm) — vision encoder does NOT use zero-centered gamma
        for s in ["weight", "bias"]:
            hk = f"{hf_blk}.norm1.{s}"
            mk = f"{mg_blk}.self_attention.linear_qkv.layer_norm_{s}"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

        # norm2 (from linear_fc1 layernorm) — vision encoder does NOT use zero-centered gamma
        for s in ["weight", "bias"]:
            hk = f"{hf_blk}.norm2.{s}"
            mk = f"{mg_blk}.mlp.linear_fc1.layer_norm_{s}"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

        # QKV: HF [3*vis_h, vis_h] -> Megatron [num_qg, 3, head_dim, vis_h]
        hk = f"{hf_blk}.attn.qkv.weight"
        mk = f"{mg_blk}.self_attention.linear_qkv.weight"
        if hk in hf_sd:
            qkv = hf_sd[hk].view(3, vis_qg, vis_head_dim, vis_h)
            qkv = qkv.transpose(0, 1)
            meg_sd[mk] = qkv.reshape(-1, vis_h).contiguous()

        hk = f"{hf_blk}.attn.qkv.bias"
        mk = f"{mg_blk}.self_attention.linear_qkv.bias"
        if hk in hf_sd:
            qkv_b = hf_sd[hk].view(3, vis_qg, vis_head_dim)
            qkv_b = qkv_b.transpose(0, 1)
            meg_sd[mk] = qkv_b.reshape(-1).contiguous()

        # attn.proj
        for s in ["weight", "bias"]:
            hk = f"{hf_blk}.attn.proj.{s}"
            mk = f"{mg_blk}.self_attention.linear_proj.{s}"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

        # mlp
        for s in ["weight", "bias"]:
            for layer in ["linear_fc1", "linear_fc2"]:
                hk = f"{hf_blk}.mlp.{layer}.{s}"
                mk = f"{mg_blk}.mlp.{layer}.{s}"
                if hk in hf_sd:
                    meg_sd[mk] = hf_sd[hk]

    # Merger
    for s in ["weight", "bias"]:
        hk = f"{hf_pfx}.merger.linear_fc1.{s}"
        mk = f"{mg_pfx}.projection.encoder.linear_fc1.{s}"
        if hk in hf_sd:
            meg_sd[mk] = hf_sd[hk]
        hk = f"{hf_pfx}.merger.linear_fc2.{s}"
        mk = f"{mg_pfx}.projection.encoder.linear_fc2.{s}"
        if hk in hf_sd:
            meg_sd[mk] = hf_sd[hk]

    # Vision final layernorm -> merger norm — vision encoder does NOT use zero-centered gamma
    for s in ["weight", "bias"]:
        hk = f"{hf_pfx}.merger.norm.{s}"
        mk = f"{mg_pfx}.decoder.final_layernorm.{s}"
        if hk in hf_sd:
            meg_sd[mk] = hf_sd[hk]


def convert_mtp(hf_sd, meg_sd, cfg):
    """Convert MTP parameters from HF to Megatron."""
    # Direct mappings (reverse of meg2hf)
    mtp_direct = {
        "mtp.fc.weight": "language_model.mtp.layers.0.eh_proj.weight",
        "mtp.pre_fc_norm_embedding.weight": "language_model.mtp.layers.0.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": "language_model.mtp.layers.0.hnorm.weight",
        "mtp.norm.weight": "language_model.mtp.layers.0.final_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight": "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight",
        "mtp.layers.0.mlp.down_proj.weight": "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight",
        "mtp.layers.0.input_layernorm.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight",
        "mtp.layers.0.self_attn.q_norm.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight",
        "mtp.layers.0.self_attn.o_proj.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight",
    }
    for hk, mk in mtp_direct.items():
        if hk in hf_sd:
            meg_sd[mk] = adjust_ln_weight(hf_sd[hk])

    # MTP QKV
    mk = "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"
    q = hf_sd.get("mtp.layers.0.self_attn.q_proj.weight")
    k = hf_sd.get("mtp.layers.0.self_attn.k_proj.weight")
    v = hf_sd.get("mtp.layers.0.self_attn.v_proj.weight")
    if q is not None and k is not None and v is not None:
        meg_sd[mk] = merge_attention_qkv(q, k, v, cfg)

    # MTP MLP
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"
    gate = hf_sd.get("mtp.layers.0.mlp.gate_proj.weight")
    up = hf_sd.get("mtp.layers.0.mlp.up_proj.weight")
    if gate is not None and up is not None:
        meg_sd[mk] = torch.cat([gate, up], dim=0)


def convert_to_megatron(hf_sd, cfg):
    """Convert HF state dict to full (unsharded) Megatron state dict."""
    meg_sd = {}
    convert_llm(hf_sd, meg_sd, cfg)
    convert_vision(hf_sd, meg_sd, cfg)
    convert_mtp(hf_sd, meg_sd, cfg)
    return meg_sd


# ============================================================
# Split by PP and TP
# ============================================================


def _find_ref_shard(ref_ckpt_dir, tp_rank, pp_rank):
    """Find reference shard file, supporting both naming conventions and iter subdirs."""
    candidates = [ref_ckpt_dir]
    # If no direct mp_rank found, check for iter_*/ subdirs
    iter_dirs = sorted(
        [
            d
            for d in os.listdir(ref_ckpt_dir)
            if d.startswith("iter_") and os.path.isdir(os.path.join(ref_ckpt_dir, d))
        ]
    )
    for d in iter_dirs:
        candidates.append(os.path.join(ref_ckpt_dir, d))

    for base in candidates:
        # Format with PP suffix (TP+PP > 1)
        path_pp = os.path.join(base, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")
        if os.path.exists(path_pp):
            return path_pp
        # Format without PP suffix (PP == 1)
        if pp_rank == 0:
            path_no_pp = os.path.join(base, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
            if os.path.exists(path_no_pp):
                return path_no_pp
    return None


def get_ref_embedding_shape(ref_ckpt_dir):
    """Get reference embedding shape from an existing Megatron checkpoint."""
    if ref_ckpt_dir is None:
        return None
    path = _find_ref_shard(ref_ckpt_dir, 0, 0)
    if path is None:
        return None
    sd = torch.load(path, map_location="cpu", weights_only=False)
    emb = sd["model"].get("language_model.embedding.word_embeddings.weight")
    if emb is not None:
        return tuple(emb.shape)
    return None


def adjust_embedding(meg_sd, ref_shape, tp_size):
    """Adjust embedding to match reference shape if needed."""
    if ref_shape is None:
        return meg_sd

    emb_key = "language_model.embedding.word_embeddings.weight"
    if emb_key not in meg_sd:
        return meg_sd

    current = meg_sd[emb_key]
    target_vocab_per_rank = ref_shape[0]
    target_vocab = target_vocab_per_rank * tp_size

    if current.shape[0] == target_vocab:
        return meg_sd

    print(f"  Adjusting embedding vocab: {current.shape[0]} -> {target_vocab}")
    if current.shape[0] < target_vocab:
        # Pad with zeros
        pad_size = target_vocab - current.shape[0]
        padding = torch.zeros(pad_size, current.shape[1], dtype=current.dtype)
        meg_sd[emb_key] = torch.cat([current, padding], dim=0)
    else:
        # Truncate
        meg_sd[emb_key] = current[:target_vocab]

    return meg_sd


def split_tp(meg_sd, cfg):
    """Split a full Megatron state dict into TP shards.

    Returns list of state dicts, one per TP rank.
    """
    tp = cfg.tp
    shards = [{} for _ in range(tp)]

    # Vision config
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads

    # LLM config
    num_qg = cfg.num_query_groups
    kv_ch = cfg.kv_channels
    heads_per_group = cfg.num_attention_heads // num_qg
    if cfg.attention_output_gate:
        total_hpg = 2 * heads_per_group + 2
    else:
        total_hpg = heads_per_group + 2

    for k, v in meg_sd.items():
        if not isinstance(v, torch.Tensor):
            for r in range(tp):
                shards[r][k] = v
            continue

        # ---- Embedding ----
        if k == "language_model.embedding.word_embeddings.weight":
            chunks = v.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r]
            continue

        # ---- Output layer ----
        if k == "language_model.output_layer.weight":
            chunks = v.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r]
            continue

        # ---- Vision model ----
        if "vision_model" in k:
            if "patch_embed" in k or "pos_embed" in k or "final_layernorm" in k:
                # Replicated or not sharded
                for r in range(tp):
                    shards[r][k] = v
            elif "linear_qkv.weight" in k:
                viewed = v.view(vis_qg, 3, vis_head_dim, vis_h)
                chunks = viewed.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r].reshape(-1, vis_h)
            elif "linear_qkv.bias" in k:
                viewed = v.view(vis_qg, 3, vis_head_dim)
                chunks = viewed.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r].reshape(-1)
            elif "linear_proj.weight" in k or "linear_fc2.weight" in k:
                # RowParallel: split dim 1
                chunks = v.chunk(tp, dim=1)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "linear_fc1.weight" in k and "projection" not in k:
                # ColumnParallel: split dim 0
                chunks = v.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "linear_fc1.bias" in k and "projection" not in k:
                chunks = v.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "linear_proj.bias" in k or "linear_fc2.bias" in k:
                # RowParallel bias: not sharded
                for r in range(tp):
                    shards[r][k] = v
            elif "layer_norm" in k:
                for r in range(tp):
                    shards[r][k] = v
            elif "projection.encoder" in k:
                if "linear_fc1" in k:
                    chunks = v.chunk(tp, dim=0)
                    for r in range(tp):
                        shards[r][k] = chunks[r]
                elif "linear_fc2.weight" in k:
                    chunks = v.chunk(tp, dim=1)
                    for r in range(tp):
                        shards[r][k] = chunks[r]
                else:
                    for r in range(tp):
                        shards[r][k] = v
            else:
                for r in range(tp):
                    shards[r][k] = v
            continue

        # ---- LLM ----
        if "layer_norm_weight" in k or "layer_norm_bias" in k:
            for r in range(tp):
                shards[r][k] = v
        elif "final_layernorm" in k:
            for r in range(tp):
                shards[r][k] = v
        # GDN layers
        elif "in_proj.weight" in k:
            # merge_gdn_in_proj produces TP-rank-grouped layout:
            # [tp0_q, tp0_k, tp0_v, tp0_z, tp0_b, tp0_a, tp1_q, ...]
            # Each TP rank is already a contiguous block [q_local, k_local, ...].
            # Just chunk along dim=0 to distribute to shards.
            chunks = v.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r]
        elif "conv1d.weight" in k:
            chunks = v.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r]
        elif "A_log" in k or "dt_bias" in k:
            chunks = v.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r]
        elif "out_norm" in k:
            for r in range(tp):
                shards[r][k] = v
        elif "out_proj.weight" in k:
            chunks = v.chunk(tp, dim=1)
            for r in range(tp):
                shards[r][k] = chunks[r]
        # Attention layers
        elif "linear_qkv.weight" in k:
            viewed = v.view(num_qg, total_hpg, kv_ch, cfg.hidden_size)
            chunks = viewed.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r].reshape(-1, cfg.hidden_size)
        elif "linear_proj.weight" in k:
            chunks = v.chunk(tp, dim=1)
            for r in range(tp):
                shards[r][k] = chunks[r]
        elif "q_layernorm" in k or "k_layernorm" in k:
            for r in range(tp):
                shards[r][k] = v
        # MLP
        elif "linear_fc1.weight" in k:
            # Gated MLP: [gate; up], each is (ffn_hidden, hidden)
            # View as (2, ffn_hidden, hidden) then chunk on dim=1
            viewed = v.view(2, cfg.ffn_hidden_size, cfg.hidden_size)
            chunks = viewed.chunk(tp, dim=1)
            for r in range(tp):
                shards[r][k] = chunks[r].reshape(-1, cfg.hidden_size)
        elif "linear_fc2.weight" in k:
            chunks = v.chunk(tp, dim=1)
            for r in range(tp):
                shards[r][k] = chunks[r]
        # MTP eh_proj (ColumnParallel)
        elif "eh_proj.weight" in k:
            chunks = v.chunk(tp, dim=0)
            for r in range(tp):
                shards[r][k] = chunks[r]
        else:
            for r in range(tp):
                shards[r][k] = v

    return shards


def split_pp(meg_sd, cfg):
    """Split full Megatron state dict by PP stages.

    Returns dict {pp_rank: state_dict}.
    """
    pp = cfg.pp
    layers_per_pp = cfg.num_layers // pp
    pp_stages = {r: {} for r in range(pp)}

    for k, v in meg_sd.items():
        if "decoder.layers." in k and "vision" not in k:
            # Extract layer index
            parts = k.split("decoder.layers.")
            rest = parts[1]
            layer_idx = int(rest.split(".")[0])
            pp_rank = layer_idx // layers_per_pp
            local_idx = layer_idx % layers_per_pp
            new_k = (
                parts[0] + "decoder.layers." + str(local_idx) + "." + ".".join(rest.split(".")[1:])
            )
            pp_stages[pp_rank][new_k] = v
        else:
            # Non-layer params: decide which PP stage they belong to
            if k.startswith("language_model.output_layer") or "mtp." in k:
                # Output layer and MTP go to last PP stage
                pp_stages[pp - 1][k] = v
            elif k.startswith("language_model.decoder.final_layernorm"):
                # Final layernorm goes to last PP stage
                pp_stages[pp - 1][k] = v
            elif k.startswith("language_model.embedding"):
                # Embedding replicated to all PP stages
                for r in range(pp):
                    pp_stages[r][k] = v
            elif k.startswith("vision_model"):
                # Vision model only in first PP stage
                pp_stages[0][k] = v
            else:
                # Other params (shouldn't happen often)
                pp_stages[0][k] = v

    return pp_stages


# ============================================================
# Add _extra_state
# ============================================================

EXTRA_STATE_KEYS = {
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "self_attention.in_proj",
    "self_attention.out_proj",
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "self_attention.core_attention",
    "self_attention.q_layernorm",
    "self_attention.k_layernorm",
    "vision_model.projection.encoder.linear_fc1",
    "vision_model.projection.encoder.linear_fc2",
}


def add_extra_states(sd):
    """Add _extra_state tensors for TE layers."""
    result = dict(sd)
    extra = torch.empty(0, dtype=torch.uint8)

    # Find all prefix patterns that need _extra_state
    prefixes = set()
    for k in list(result.keys()):
        for pattern in EXTRA_STATE_KEYS:
            if pattern in k:
                # Get the base prefix (everything up to the pattern)
                idx = k.find(pattern)
                if idx >= 0:
                    base = k[: idx + len(pattern)]
                    prefixes.add(base)

    for base in prefixes:
        es_key = f"{base}._extra_state"
        if es_key not in result:
            result[es_key] = extra.clone()

    return result


# ============================================================
# Validate
# ============================================================


def validate(shards_dict, cfg, ref_ckpt_dir):
    """Compare generated shards with reference checkpoint."""
    if ref_ckpt_dir is None:
        return True

    print("\n" + "=" * 80)
    print("Validation: Comparing with reference checkpoint")
    print("=" * 80)

    all_ok = True
    for pp_rank in range(cfg.pp):
        for tp_rank in range(cfg.tp):
            ref_path = _find_ref_shard(ref_ckpt_dir, tp_rank, pp_rank)
            if ref_path is None:
                print(f"  Skip: reference not found for PP={pp_rank}, TP={tp_rank}")
                continue

            ref_sd = torch.load(ref_path, map_location="cpu", weights_only=False)["model"]
            gen_sd = shards_dict[(pp_rank, tp_rank)]

            ref_keys = set(k for k in ref_sd.keys() if "_extra_state" not in k)
            gen_keys = set(k for k in gen_sd.keys() if "_extra_state" not in k)

            # If untie=false, output_layer may not exist in HF-derived checkpoint
            if not cfg.untie:
                ref_keys.discard("language_model.output_layer.weight")
                gen_keys.discard("language_model.output_layer.weight")

            missing = ref_keys - gen_keys
            extra = gen_keys - ref_keys

            if missing:
                print(f"  PP={pp_rank}, TP={tp_rank}: Missing keys ({len(missing)}):")
                for k in sorted(missing)[:5]:
                    print(f"    {k}")
                all_ok = False
            if extra:
                print(f"  PP={pp_rank}, TP={tp_rank}: Extra keys ({len(extra)}):")
                for k in sorted(extra)[:5]:
                    print(f"    {k}")
                all_ok = False

            # Check shapes
            mismatches = 0
            emb_mismatch = False
            for k in ref_keys & gen_keys:
                if isinstance(ref_sd[k], torch.Tensor) and isinstance(gen_sd[k], torch.Tensor):
                    if ref_sd[k].shape != gen_sd[k].shape:
                        if "embedding.word_embeddings" in k:
                            emb_mismatch = True
                            print(
                                "  Embedding shape differs (expected if not using --adjust-embedding):"
                            )
                            print(
                                f"    ref: {tuple(ref_sd[k].shape)}, gen: {tuple(gen_sd[k].shape)}"
                            )
                        else:
                            mismatches += 1
                            if mismatches <= 3:
                                print(f"  Shape mismatch: {k}")
                                print(
                                    f"    ref: {tuple(ref_sd[k].shape)}, gen: {tuple(gen_sd[k].shape)}"
                                )
            if mismatches > 0:
                print(f"  Total shape mismatches: {mismatches}")
                all_ok = False

            if not missing and not extra and mismatches == 0:
                if emb_mismatch:
                    print(f"  PP={pp_rank}, TP={tp_rank}: OK (embedding shape differs)")
                else:
                    print(f"  PP={pp_rank}, TP={tp_rank}: OK")

    print("=" * 80)
    if all_ok:
        print("Validation PASSED")
    else:
        print("Validation FAILED")
    print("=" * 80)
    return all_ok


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()
    cfg = Config(args.yaml)

    # Apply CLI override for layer norm adjustment
    global LN_ADJUSTMENT
    if args.no_adjust_ln:
        LN_ADJUSTMENT = False

    print(
        f"Config: TP={cfg.tp}, PP={cfg.pp}, layers={cfg.num_layers}, "
        f"hidden={cfg.hidden_size}, ffn={cfg.ffn_hidden_size}"
    )
    print(f"GDN: freq={cfg.linear_attention_freq}, qk_dim={cfg.qk_dim}, v_dim={cfg.v_dim}")
    print(f"Vision: layers={cfg.vision_num_layers}, hidden={cfg.vision_hidden_size}")
    print(f"LN adjustment: {LN_ADJUSTMENT}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load HF weights
    print(f"\nLoading HF weights from {args.hf_dir}...")
    hf_sd = load_hf_weights(args.hf_dir)
    print(f"Loaded {len(hf_sd)} HF parameters")

    # Convert to Megatron naming
    print("\nConverting to Megatron format...")
    meg_sd = convert_to_megatron(hf_sd, cfg)
    del hf_sd
    print(f"Converted to {len(meg_sd)} Megatron parameters")

    # Adjust embedding if requested
    if args.adjust_embedding and args.ref_ckpt_dir is not None:
        ref_emb_shape = get_ref_embedding_shape(args.ref_ckpt_dir)
        if ref_emb_shape is not None:
            meg_sd = adjust_embedding(meg_sd, ref_emb_shape, cfg.tp)

    # Split by PP
    print(f"\nSplitting by PP (stages={cfg.pp})...")
    pp_stages = split_pp(meg_sd, cfg)
    del meg_sd

    # Split by TP and save under release/ (release checkpoint format)
    ckpt_base = os.path.join(args.save_dir, "release")
    os.makedirs(ckpt_base, exist_ok=True)
    print(f"Splitting by TP (ranks={cfg.tp}) and saving to {ckpt_base}...")
    shards_dict = {}
    for pp_rank in range(cfg.pp):
        tp_shards = split_tp(pp_stages[pp_rank], cfg)
        for tp_rank in range(cfg.tp):
            shard = add_extra_states(tp_shards[tp_rank])
            shards_dict[(pp_rank, tp_rank)] = shard

            # Save
            if cfg.pp == 1:
                ckpt_dir = os.path.join(ckpt_base, f"mp_rank_{tp_rank:02d}")
            else:
                ckpt_dir = os.path.join(ckpt_base, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_path = os.path.join(ckpt_dir, "model_optim_rng.pt")
            torch.save({"model": shard}, save_path)
            print(f"  Saved: {save_path} ({len(shard)} keys)")

    # Create tracker file for release checkpoint
    tracker_path = os.path.join(args.save_dir, "latest_checkpointed_iteration.txt")
    with open(tracker_path, "w") as f:
        f.write("release\n")
    print(f"  Created tracker: {tracker_path}")

    # Validate (optional, only if ref provided)
    if args.ref_ckpt_dir:
        success = validate(shards_dict, cfg, args.ref_ckpt_dir)
        import sys

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
