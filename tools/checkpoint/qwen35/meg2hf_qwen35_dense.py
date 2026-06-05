#!/usr/bin/env python3
"""
Convert Megatron TP/PP checkpoint to HuggingFace format for Qwen3.5 dense models.

Reads model config from the training yaml, loads raw checkpoint shards,
merges TP/PP, converts to HF naming/shapes, saves as safetensors.
Optionally compares with a reference HF model when --hf-ref-dir is provided.

Usage:
    python meg2hf_qwen35_dense.py \
        --yaml /path/to/4b.yaml \
        --meg-ckpt-dir /path/to/checkpoints/iter_0000001 \
        --save-dir /path/to/output
        [--hf-ref-dir /path/to/hf/qwen35_4b]
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

# Layer norm adjustment for zero-centered gamma.
# Controls LLM and MTP layernorms only. Vision layernorms are never adjusted.
# By default enabled. Use --no-adjust-ln CLI flag to disable.
LN_ADJUSTMENT = True


def restore_ln_weight(weight):
    """Restore layer norm weight (add 1.0) from zero-centered gamma format."""
    return weight + 1.0 if LN_ADJUSTMENT else weight


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

        self.vision_num_layers = model.get("vision_num_layers", 12)
        self.vision_hidden_size = model.get("vision_hidden_size", 768)
        self.vision_num_attention_heads = model.get("vision_num_attention_heads", 12)
        self.vision_ffn_hidden_size = model.get("vision_ffn_hidden_size", 3072)

        self.patch_size = model.get("patch_size", 16)
        self.temporal_patch_size = 2  # hardcoded in get_vision_model_config
        self.use_linear_proj = model.get("vision_patch_embed_linear", False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to training yaml config")
    p.add_argument("--meg-ckpt-dir", required=True, help="Path to Megatron checkpoint directory")
    p.add_argument(
        "--hf-ref-dir", default=None, help="Reference HF model dir for shape validation (optional)"
    )
    p.add_argument("--save-dir", required=True, help="Output directory for HF checkpoint")
    p.add_argument(
        "--no-adjust-ln",
        action="store_true",
        help="Disable layer norm adjustment (zero-centered gamma)",
    )
    return p.parse_args()


# ============================================================
# Load & merge
# ============================================================


def load_shard(ckpt_dir, tp_rank, pp_rank):
    # Try PP suffix format first (TP+PP > 1)
    path_pp = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")
    if os.path.exists(path_pp):
        sd = torch.load(path_pp, map_location="cpu", weights_only=False)
        return sd["model"]
    # Fallback to no-PP format (PP == 1)
    if pp_rank == 0:
        path_no_pp = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        if os.path.exists(path_no_pp):
            sd = torch.load(path_no_pp, map_location="cpu", weights_only=False)
            return sd["model"]
    raise FileNotFoundError(f"Cannot find checkpoint for TP={tp_rank}, PP={pp_rank} in {ckpt_dir}")


def is_gdn_layer(idx, freq):
    return (idx % freq) != (freq - 1)


def merge_tp_shards(shards, cfg):
    """Merge TP shards into one state dict for a single PP stage."""
    tp = cfg.tp
    merged = {}
    all_keys = set()
    for s in shards:
        all_keys.update(s.keys())

    # Vision config
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads  # vision uses num_query_groups = num_heads (MHA)
    vis_gps = vis_qg // tp

    for k in sorted(all_keys):
        if "extra_state" in k:
            continue
        vals = [s[k] for s in shards if k in s]
        if not isinstance(vals[0], torch.Tensor):
            continue

        # ---- Vision model (ETP sharded) ----
        if "vision_model" in k:
            if "patch_embed" in k or "pos_embed" in k:
                # Replicated
                merged[k] = vals[0]
            elif "linear_qkv.weight" in k:
                viewed = [x.view(vis_gps, -1, vis_head_dim, vis_h) for x in vals]
                merged[k] = torch.cat(viewed, dim=0).view(-1, vis_h)
            elif "linear_qkv.bias" in k:
                viewed = [x.view(vis_gps, -1, vis_head_dim) for x in vals]
                merged[k] = torch.cat(viewed, dim=0).view(-1)
            elif "linear_proj.weight" in k or "linear_fc2.weight" in k:
                merged[k] = torch.cat(vals, dim=1)
            elif "linear_fc1.weight" in k and "projection" not in k:
                # ColumnParallel: cat dim 0
                merged[k] = torch.cat(vals, dim=0)
            elif "linear_fc1.bias" in k and "projection" not in k:
                merged[k] = torch.cat(vals, dim=0)
            elif "linear_proj.bias" in k or "linear_fc2.bias" in k:
                # RowParallel bias: not sharded
                merged[k] = vals[0]
            elif "layer_norm" in k or "final_layernorm" in k:
                merged[k] = vals[0]
            elif "projection.encoder" in k:
                # Vision merger/projection: fc1 is ColumnParallel, fc2 is RowParallel
                if "linear_fc1" in k:
                    merged[k] = torch.cat(vals, dim=0)
                elif "linear_fc2.weight" in k:
                    merged[k] = torch.cat(vals, dim=1)
                else:
                    # fc2 bias (RowParallel: not sharded), layernorm, etc.
                    merged[k] = vals[0]
            else:
                merged[k] = vals[0]
            continue

        # ---- LLM ----
        if "embedding" in k:
            merged[k] = torch.cat(vals, dim=0)
        elif "final_layernorm" in k:
            merged[k] = vals[0]
        elif "layer_norm_weight" in k or "layer_norm_bias" in k:
            merged[k] = vals[0]
        # GDN layers
        elif "in_proj.weight" in k:
            # Each rank has [q_local, k_local, v_local, z_local, beta_local, alpha_local]
            # Reconstruct total: [q_all, k_all, v_all, z_all, beta_all, alpha_all]
            qk_dim = cfg.qk_dim
            v_dim = cfg.v_dim
            num_v = cfg.linear_num_value_heads
            q_per = qk_dim // tp
            k_per = qk_dim // tp
            v_per = v_dim // tp
            z_per = v_dim // tp
            b_per = num_v // tp
            a_per = num_v // tp

            q_all, k_all, v_all, z_all, b_all, a_all = [], [], [], [], [], []
            for val in vals:
                off = 0
                q_all.append(val[off : off + q_per])
                off += q_per
                k_all.append(val[off : off + k_per])
                off += k_per
                v_all.append(val[off : off + v_per])
                off += v_per
                z_all.append(val[off : off + z_per])
                off += z_per
                b_all.append(val[off : off + b_per])
                off += b_per
                a_all.append(val[off : off + a_per])

            merged[k] = torch.cat(
                [
                    torch.cat(q_all, dim=0),
                    torch.cat(k_all, dim=0),
                    torch.cat(v_all, dim=0),
                    torch.cat(z_all, dim=0),
                    torch.cat(b_all, dim=0),
                    torch.cat(a_all, dim=0),
                ],
                dim=0,
            )
        elif "conv1d.weight" in k:
            merged[k] = torch.cat(vals, dim=0)
        elif "A_log" in k or "dt_bias" in k:
            merged[k] = torch.cat(vals, dim=0)
        elif "out_norm" in k:
            merged[k] = vals[0]
        elif "out_proj.weight" in k:
            merged[k] = torch.cat(vals, dim=1)
        # Attention layers
        elif "linear_qkv.weight" in k:
            gps = cfg.num_query_groups // tp
            heads_per_group = cfg.num_attention_heads // cfg.num_query_groups
            if cfg.attention_output_gate:
                total_hpg = 2 * heads_per_group + 2
            else:
                total_hpg = heads_per_group + 2
            viewed = [x.view(gps, total_hpg, cfg.kv_channels, cfg.hidden_size) for x in vals]
            merged[k] = torch.cat(viewed, dim=0).view(-1, cfg.hidden_size)
        elif "linear_proj.weight" in k:
            merged[k] = torch.cat(vals, dim=1)
        elif "q_layernorm" in k or "k_layernorm" in k:
            merged[k] = vals[0]
        # MLP
        elif "linear_fc1.weight" in k:
            viewed = [x.view(2, -1, cfg.hidden_size) for x in vals]
            merged[k] = torch.cat(viewed, dim=1).view(-1, cfg.hidden_size)
        elif "linear_fc2.weight" in k:
            merged[k] = torch.cat(vals, dim=1)
        # MTP eh_proj (ColumnParallel)
        elif "eh_proj.weight" in k:
            merged[k] = torch.cat(vals, dim=0)
        else:
            merged[k] = vals[0]

    return merged


def merge_pp_stages(pp_merged, cfg):
    """Merge PP stages into a single state dict with global layer indices."""
    layers_per_pp = cfg.num_layers // cfg.pp
    full_sd = {}

    for pp_rank in range(cfg.pp):
        sd = pp_merged[pp_rank]
        for k, v in sd.items():
            if "decoder.layers." in k and "vision" not in k:
                # Remap local layer index to global
                parts = k.split("decoder.layers.")
                rest = parts[1]
                local_idx = int(rest.split(".")[0])
                global_idx = pp_rank * layers_per_pp + local_idx
                new_k = (
                    parts[0]
                    + "decoder.layers."
                    + str(global_idx)
                    + "."
                    + ".".join(rest.split(".")[1:])
                )
                full_sd[new_k] = v
            else:
                full_sd[k] = v

    return full_sd


# ============================================================
# Convert Megatron -> HF
# ============================================================


def split_gdn_in_proj(in_proj_weight, cfg):
    """Split Megatron's fused in_proj into HF's separate qkv, z, b, a.

    Megatron in_proj order (after TP merge): [tp0_q, tp0_k, tp0_v, tp0_z, tp0_b, tp0_a,
                                               tp1_q, tp1_k, tp1_v, tp1_z, tp1_b, tp1_a, ...]

    Returns: (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)
    """
    hidden = cfg.hidden_size
    num_qk_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    qk_head_dim = cfg.linear_key_head_dim
    v_head_dim = cfg.linear_value_head_dim
    v_per_group = num_v_heads // num_qk_heads
    tp_size = cfg.tp
    heads_per_tp = num_qk_heads // tp_size

    # After TP merge, in_proj is [tp_size * per_rank_dim, hidden]
    # Reshape to (tp_size, per_rank_dim, hidden)
    per_rank_dim = in_proj_weight.shape[0] // tp_size
    rank_weights = in_proj_weight.reshape(tp_size, per_rank_dim, hidden)

    # Split each rank into q, k, v, z, b, a
    q_parts, k_parts, v_parts, z_parts, b_parts, a_parts = [], [], [], [], [], []
    for rank in range(tp_size):
        rw = rank_weights[rank]
        q_end = heads_per_tp * qk_head_dim
        k_end = q_end + heads_per_tp * qk_head_dim
        v_end = k_end + heads_per_tp * v_per_group * v_head_dim
        z_end = v_end + heads_per_tp * v_per_group * v_head_dim
        b_end = z_end + heads_per_tp * v_per_group
        a_end = b_end + heads_per_tp * v_per_group
        q_parts.append(rw[:q_end])
        k_parts.append(rw[q_end:k_end])
        v_parts.append(rw[k_end:v_end])
        z_parts.append(rw[v_end:z_end])
        b_parts.append(rw[z_end:b_end])
        a_parts.append(rw[b_end:a_end])

    # Cat all parts and reshape to head-grouped form
    q_g = torch.cat(q_parts, dim=0).reshape(num_qk_heads, qk_head_dim, hidden)
    k_g = torch.cat(k_parts, dim=0).reshape(num_qk_heads, qk_head_dim, hidden)
    v_g = torch.cat(v_parts, dim=0).reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    z_g = torch.cat(z_parts, dim=0).reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    b_g = torch.cat(b_parts, dim=0).reshape(num_qk_heads, v_per_group, hidden)
    a_g = torch.cat(a_parts, dim=0).reshape(num_qk_heads, v_per_group, hidden)

    # Flatten back to HF format
    qkv = torch.cat(
        [
            q_g.reshape(-1, hidden),
            k_g.reshape(-1, hidden),
            v_g.reshape(-1, hidden),
        ],
        dim=0,
    )
    z_flat = z_g.reshape(-1, hidden)
    b_flat = b_g.reshape(-1, hidden)
    a_flat = a_g.reshape(-1, hidden)

    return qkv, z_flat, b_flat, a_flat


def convert_to_hf(full_sd, cfg):
    """Convert merged Megatron state dict to HF naming convention."""
    hf_sd = {}
    hidden = cfg.hidden_size
    freq = cfg.linear_attention_freq

    # ---- Embedding (no pad/truncate — keep original shape for honest comparison) ----
    emb_key = "language_model.embedding.word_embeddings.weight"
    if emb_key in full_sd:
        hf_sd["model.language_model.embed_tokens.weight"] = full_sd[emb_key]

    # ---- LLM layers ----
    for layer_idx in range(cfg.num_layers):
        mg_pfx = f"language_model.decoder.layers.{layer_idx}"
        hf_pfx = f"model.language_model.layers.{layer_idx}"

        if is_gdn_layer(layer_idx, freq):
            # --- GDN layer ---
            # input_layernorm
            mk = f"{mg_pfx}.self_attention.in_proj.layer_norm_weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.input_layernorm.weight"] = restore_ln_weight(full_sd[mk])

            # in_proj -> qkv, z, b, a
            mk = f"{mg_pfx}.self_attention.in_proj.weight"
            if mk in full_sd:
                qkv, z, b, a = split_gdn_in_proj(full_sd[mk], cfg)
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_qkv.weight"] = qkv
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_z.weight"] = z
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_b.weight"] = b
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_a.weight"] = a

            # conv1d
            mk = f"{mg_pfx}.self_attention.conv1d.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.conv1d.weight"] = full_sd[mk]

            # out_proj
            mk = f"{mg_pfx}.self_attention.out_proj.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.out_proj.weight"] = full_sd[mk]

            # out_norm -> norm
            mk = f"{mg_pfx}.self_attention.out_norm.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.norm.weight"] = restore_ln_weight(full_sd[mk])

            # A_log, dt_bias
            mk = f"{mg_pfx}.self_attention.A_log"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.A_log"] = full_sd[mk]
            mk = f"{mg_pfx}.self_attention.dt_bias"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.dt_bias"] = full_sd[mk]

        else:
            # --- Standard Attention layer ---
            # input_layernorm
            mk = f"{mg_pfx}.self_attention.linear_qkv.layer_norm_weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.input_layernorm.weight"] = restore_ln_weight(full_sd[mk])

            # QKV split (with optional attention_output_gate)
            mk = f"{mg_pfx}.self_attention.linear_qkv.weight"
            if mk in full_sd:
                num_qg = cfg.num_query_groups
                kv_ch = cfg.kv_channels
                heads_per_group = cfg.num_attention_heads // num_qg
                if cfg.attention_output_gate:
                    total_hpg = 2 * heads_per_group + 2
                else:
                    total_hpg = heads_per_group + 2
                qkv = full_sd[mk].view(num_qg, total_hpg, kv_ch, hidden)
                if cfg.attention_output_gate:
                    # Interleaved: [q*hpg, z*hpg, k, v] per group
                    q_heads = qkv[:, :heads_per_group]
                    z_heads = qkv[:, heads_per_group : 2 * heads_per_group]
                    k_heads = qkv[:, -2:-1]
                    v_heads = qkv[:, -1:]
                    # HF q_proj = cat([Q, Z], dim=1) per group then flatten
                    q_combined = torch.cat([q_heads, z_heads], dim=1)
                    hf_sd[f"{hf_pfx}.self_attn.q_proj.weight"] = q_combined.reshape(-1, hidden)
                else:
                    q, k_heads, v_heads = torch.split(qkv, [heads_per_group, 1, 1], dim=1)
                    hf_sd[f"{hf_pfx}.self_attn.q_proj.weight"] = q.reshape(-1, hidden)
                hf_sd[f"{hf_pfx}.self_attn.k_proj.weight"] = k_heads.reshape(-1, hidden)
                hf_sd[f"{hf_pfx}.self_attn.v_proj.weight"] = v_heads.reshape(-1, hidden)

            # o_proj
            mk = f"{mg_pfx}.self_attention.linear_proj.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.self_attn.o_proj.weight"] = full_sd[mk]

            # q/k norm
            mk = f"{mg_pfx}.self_attention.q_layernorm.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.self_attn.q_norm.weight"] = restore_ln_weight(full_sd[mk])
            mk = f"{mg_pfx}.self_attention.k_layernorm.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.self_attn.k_norm.weight"] = restore_ln_weight(full_sd[mk])

        # --- MLP (same for both layer types) ---
        # post_attention_layernorm
        mk = f"{mg_pfx}.mlp.linear_fc1.layer_norm_weight"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.post_attention_layernorm.weight"] = restore_ln_weight(full_sd[mk])

        mk = f"{mg_pfx}.mlp.linear_fc1.weight"
        if mk in full_sd:
            gate, up = torch.split(full_sd[mk], cfg.ffn_hidden_size, dim=0)
            hf_sd[f"{hf_pfx}.mlp.gate_proj.weight"] = gate
            hf_sd[f"{hf_pfx}.mlp.up_proj.weight"] = up

        mk = f"{mg_pfx}.mlp.linear_fc2.weight"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.mlp.down_proj.weight"] = full_sd[mk]

    # ---- Final layernorm ----
    mk = "language_model.decoder.final_layernorm.weight"
    if mk in full_sd:
        hf_sd["model.language_model.norm.weight"] = restore_ln_weight(full_sd[mk])

    # ---- Output layer (lm_head) ----
    if cfg.untie:
        mk = "language_model.output_layer.weight"
        if mk in full_sd:
            hf_sd["lm_head.weight"] = full_sd[mk]

    # ---- Vision model ----
    convert_vision(full_sd, hf_sd, cfg)

    # ---- MTP (Multi-Token Prediction) ----
    convert_mtp(full_sd, hf_sd, cfg)

    return hf_sd


def convert_vision(full_sd, hf_sd, cfg):
    """Convert vision model parameters."""
    mg_pfx = "vision_model"
    hf_pfx = "model.visual"
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads  # MHA: num_query_groups = num_heads

    # patch_embed
    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.patch_embed.proj.{s}"
        if mk in full_sd:
            val = full_sd[mk]
            if s == "weight" and cfg.use_linear_proj:
                # Megatron uses nn.Linear, HF expects nn.Conv3d
                val = val.view(
                    cfg.vision_hidden_size,
                    3,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                )
            hf_sd[f"{hf_pfx}.patch_embed.proj.{s}"] = val

    # pos_embed
    mk = f"{mg_pfx}.pos_embed.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.pos_embed.weight"] = full_sd[mk]

    # Vision blocks
    for i in range(cfg.vision_num_layers):
        mg_blk = f"{mg_pfx}.decoder.layers.{i}"
        hf_blk = f"{hf_pfx}.blocks.{i}"

        # norm1 (from linear_qkv layernorm) — vision encoder does NOT use zero-centered gamma
        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.self_attention.linear_qkv.layer_norm_{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.norm1.{s}"] = full_sd[mk]

        # norm2 (from linear_fc1 layernorm) — vision encoder does NOT use zero-centered gamma
        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.mlp.linear_fc1.layer_norm_{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.norm2.{s}"] = full_sd[mk]

        # QKV: Megatron grouped [num_qg, (q+k+v)*head_dim, vis_h] -> HF [3*vis_h, vis_h]
        mk = f"{mg_blk}.self_attention.linear_qkv.weight"
        if mk in full_sd:
            qkv_w = full_sd[mk].view(vis_qg, 3, vis_head_dim, vis_h)
            converted = qkv_w.transpose(0, 1).reshape(-1, vis_h).contiguous()
            hf_sd[f"{hf_blk}.attn.qkv.weight"] = converted

        mk = f"{mg_blk}.self_attention.linear_qkv.bias"
        if mk in full_sd:
            qkv_b = full_sd[mk].view(vis_qg, 3, vis_head_dim)
            converted = qkv_b.transpose(0, 1).reshape(-1).contiguous()
            hf_sd[f"{hf_blk}.attn.qkv.bias"] = converted

        # attn.proj
        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.self_attention.linear_proj.{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.attn.proj.{s}"] = full_sd[mk]

        # mlp
        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.mlp.linear_fc1.{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.mlp.linear_fc1.{s}"] = full_sd[mk]
            mk = f"{mg_blk}.mlp.linear_fc2.{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.mlp.linear_fc2.{s}"] = full_sd[mk]

    # Merger
    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.projection.encoder.linear_fc1.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.linear_fc1.{s}"] = full_sd[mk]
        mk = f"{mg_pfx}.projection.encoder.linear_fc2.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.linear_fc2.{s}"] = full_sd[mk]

    # Vision final layernorm -> merger norm — vision encoder does NOT use zero-centered gamma
    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.decoder.final_layernorm.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.norm.{s}"] = full_sd[mk]


def convert_mtp(full_sd, hf_sd, cfg):
    """Convert MTP (Multi-Token Prediction) parameters."""
    hidden = cfg.hidden_size

    # Direct mappings
    mtp_direct = {
        "language_model.mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
        "language_model.mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "language_model.mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "language_model.mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight": "mtp.layers.0.post_attention_layernorm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight": "mtp.layers.0.mlp.down_proj.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
    }
    for mk, hk in mtp_direct.items():
        if mk in full_sd:
            hf_sd[hk] = restore_ln_weight(full_sd[mk])

    # MTP QKV (standard attention with gated output)
    mk = "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"
    if mk in full_sd:
        num_qg = cfg.num_query_groups
        kv_ch = cfg.kv_channels
        heads_per_group = cfg.num_attention_heads // num_qg
        if cfg.attention_output_gate:
            total_hpg = 2 * heads_per_group + 2
        else:
            total_hpg = heads_per_group + 2
        qkv = full_sd[mk].view(num_qg, total_hpg, kv_ch, hidden)
        if cfg.attention_output_gate:
            q_heads = qkv[:, :heads_per_group]
            z_heads = qkv[:, heads_per_group : 2 * heads_per_group]
            k_heads = qkv[:, -2:-1]
            v_heads = qkv[:, -1:]
            q_combined = torch.cat([q_heads, z_heads], dim=1)
            hf_sd["mtp.layers.0.self_attn.q_proj.weight"] = q_combined.reshape(-1, hidden)
        else:
            q, k_heads, v_heads = torch.split(qkv, [heads_per_group, 1, 1], dim=1)
            hf_sd["mtp.layers.0.self_attn.q_proj.weight"] = q.reshape(-1, hidden)
        hf_sd["mtp.layers.0.self_attn.k_proj.weight"] = k_heads.reshape(-1, hidden)
        hf_sd["mtp.layers.0.self_attn.v_proj.weight"] = v_heads.reshape(-1, hidden)

    # MTP MLP (gated)
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"
    if mk in full_sd:
        gate, up = torch.split(full_sd[mk], cfg.ffn_hidden_size, dim=0)
        hf_sd["mtp.layers.0.mlp.gate_proj.weight"] = gate
        hf_sd["mtp.layers.0.mlp.up_proj.weight"] = up


# ============================================================
# Compare
# ============================================================


def load_hf_shapes(hf_dir):
    if hf_dir is None:
        return None
    shapes = {}
    for st_file in sorted(Path(hf_dir).glob("*.safetensors")):
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = list(f.get_tensor(key).shape)
    return shapes


def compare_models(hf_sd, ref_shapes):
    """Compare converted model with reference HF model and return PASS/FAIL."""
    print("\n" + "=" * 100)
    print("Shape Comparison: Converted vs Reference HF Model")
    print("=" * 100)

    if ref_shapes is None or len(ref_shapes) == 0:
        print("No reference shapes provided, skipping comparison.")
        return True

    # Include MTP in comparison (don't filter it out)
    all_keys = sorted(set(list(hf_sd.keys()) + list(ref_shapes.keys())))
    mismatches, missing, extra, matched = [], [], [], 0

    for k in all_keys:
        in_conv = k in hf_sd
        in_ref = k in ref_shapes
        if in_conv and in_ref:
            cs = list(hf_sd[k].shape)
            rs = ref_shapes[k]
            if cs == rs:
                matched += 1
            else:
                mismatches.append((k, cs, rs))
        elif in_conv:
            extra.append(k)
        else:
            missing.append(k)

    print(f"\nMatched: {matched}/{len(all_keys)}")

    # Report issues
    if mismatches:
        print(f"\n❌ Shape mismatches ({len(mismatches)}):")
        for k, cs, rs in mismatches:
            print(f"  {k:80s} converted={cs} ref={rs}")
    if missing:
        print(f"\n❌ Missing in converted ({len(missing)}):")
        for k in missing:
            print(f"  {k:80s} ref_shape={ref_shapes[k]}")
    if extra:
        print(f"\n❌ Extra in converted ({len(extra)}):")
        for k in extra:
            print(f"  {k:80s} shape={list(hf_sd[k].shape)}")

    # Param counts
    conv_total = sum(t.numel() for t in hf_sd.values())
    ref_total = sum(__import__("math").prod(shape) if shape else 1 for shape in ref_shapes.values())

    print(f"\nConverted total params: {conv_total:>15,}")
    print(f"Reference total params: {ref_total:>15,}")
    print(f"Difference:             {conv_total - ref_total:>15,}")

    # Final verdict
    print("\n" + "=" * 100)
    if mismatches or missing or extra:
        print("❌ VALIDATION FAILED")
        print(f"   - {len(mismatches)} shape mismatches")
        print(f"   - {len(missing)} missing params")
        print(f"   - {len(extra)} extra params")
        success = False
    else:
        print("✅ VALIDATION PASSED")
        print(f"   - All {matched} parameters match")
        print(f"   - Total params: {conv_total:,}")
        success = True
    print("=" * 100)

    return success


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
    print(f"LN adjustment: {LN_ADJUSTMENT}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load and merge TP shards per PP stage
    pp_merged = {}
    for pp_rank in range(cfg.pp):
        print(f"\nLoading PP stage {pp_rank}...")
        shards = []
        for tp_rank in range(cfg.tp):
            print(f"  Loading TP shard {tp_rank}...")
            shards.append(load_shard(args.meg_ckpt_dir, tp_rank, pp_rank))
        pp_merged[pp_rank] = merge_tp_shards(shards, cfg)
        del shards

    # Merge PP stages
    print("\nMerging PP stages...")
    full_sd = merge_pp_stages(pp_merged, cfg)
    del pp_merged

    print(f"\nMerged Megatron keys: {len(full_sd)}")
    for k in sorted(full_sd.keys()):
        if isinstance(full_sd[k], torch.Tensor):
            print(f"  {k:90s} {list(full_sd[k].shape)}")

    # Convert to HF
    print("\nConverting to HF format...")
    hf_sd = convert_to_hf(full_sd, cfg)
    del full_sd

    print(f"\nConverted HF keys: {len(hf_sd)}")
    for k in sorted(hf_sd.keys()):
        print(f"  {k:80s} {list(hf_sd[k].shape)}")

    # Save
    print(f"\nSaving to {args.save_dir}...")
    save_file(hf_sd, os.path.join(args.save_dir, "model.safetensors"))

    # Compare (optional, only if reference provided)
    if args.hf_ref_dir:
        print("\nComparing with reference HF model...")
        ref_shapes = load_hf_shapes(args.hf_ref_dir)
        success = compare_models(hf_sd, ref_shapes)
        import sys

        sys.exit(0 if success else 1)
    else:
        print("\nConversion complete (no reference provided for validation).")


if __name__ == "__main__":
    main()
