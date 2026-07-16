"""Unified dense / MoE Qwen3.5 checkpoint converter."""

import os

import torch

from qwen35 import constants
from qwen35.attention import merge_attention_qkv, split_attention_qkv
from qwen35.config import Config
from qwen35.gdn import (
    _merge_gdn_conv1d,
    _shard_gdn_conv1d,
    is_gdn_layer,
    merge_gdn_in_proj,
    split_gdn_in_proj,
)
from qwen35.io import (
    ensure_hf_path,
    find_megatron_shard,
    load_hf_weights,
    load_megatron_shard,
    save_hf_checkpoint,
    save_megatron_release_checkpoint,
)
from qwen35.mlp import (
    convert_layer_mlp_hf2meg,
    convert_layer_mlp_meg2hf,
)
from qwen35.mtp import convert_mtp_hf2meg, convert_mtp_meg2hf
from qwen35.sharding import (
    merge_ep_experts,
    merge_pp_layers,
    split_ep_experts,
    split_pp_layers,
)
from qwen35.validation import validate_hf2meg_against_ref, validate_meg2hf_against_ref
from qwen35.vision import convert_vision_hf2meg, convert_vision_meg2hf


# -----------------------------------------------------------------------------
# Layer-norm helpers
# -----------------------------------------------------------------------------
def _adjust_ln(weight):
    """Subtract 1.0 when legacy LN adjustment is enabled."""
    return weight - 1.0 if constants.LN_ADJUSTMENT else weight


def _restore_ln(weight):
    """Add 1.0 when legacy LN adjustment is enabled."""
    return weight + 1.0 if constants.LN_ADJUSTMENT else weight


# -----------------------------------------------------------------------------
# Base converter
# -----------------------------------------------------------------------------
class BaseConverter:
    """Common conversion pipeline for Qwen3.5 dense and MoE models."""

    def __init__(self, cfg: Config, adjust_embedding=False):
        self.cfg = cfg
        self.adjust_embedding = adjust_embedding

    # -------------------------------------------------------------------------
    # HF -> Megatron naming conversion
    # -------------------------------------------------------------------------
    def _convert_llm_hf2meg(self, hf_sd, meg_sd):
        cfg = self.cfg
        freq = cfg.linear_attention_freq

        hf_key = "model.language_model.embed_tokens.weight"
        if hf_key in hf_sd:
            meg_sd["language_model.embedding.word_embeddings.weight"] = hf_sd[hf_key]

        for layer_idx in range(cfg.num_layers):
            hf_pfx = f"model.language_model.layers.{layer_idx}"
            mg_pfx = f"language_model.decoder.layers.{layer_idx}"

            if is_gdn_layer(layer_idx, freq):
                # GDN layer
                mk = f"{mg_pfx}.self_attention.in_proj.layer_norm_weight"
                hk = f"{hf_pfx}.input_layernorm.weight"
                if hk in hf_sd:
                    meg_sd[mk] = _adjust_ln(hf_sd[hk])

                mk = f"{mg_pfx}.self_attention.in_proj.weight"
                qkv = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_qkv.weight")
                z = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_z.weight")
                b = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_b.weight")
                a = hf_sd.get(f"{hf_pfx}.linear_attn.in_proj_a.weight")
                if qkv is not None and z is not None and b is not None and a is not None:
                    meg_sd[mk] = merge_gdn_in_proj(qkv, z, b, a, cfg)

                mk = f"{mg_pfx}.self_attention.conv1d.weight"
                hk = f"{hf_pfx}.linear_attn.conv1d.weight"
                if hk in hf_sd:
                    meg_sd[mk] = hf_sd[hk]

                mk = f"{mg_pfx}.self_attention.out_proj.weight"
                hk = f"{hf_pfx}.linear_attn.out_proj.weight"
                if hk in hf_sd:
                    meg_sd[mk] = hf_sd[hk]

                mk = f"{mg_pfx}.self_attention.out_norm.weight"
                hk = f"{hf_pfx}.linear_attn.norm.weight"
                if hk in hf_sd:
                    meg_sd[mk] = hf_sd[hk] - 1.0

                for suffix in ["A_log", "dt_bias"]:
                    mk = f"{mg_pfx}.self_attention.{suffix}"
                    hk = f"{hf_pfx}.linear_attn.{suffix}"
                    if hk in hf_sd:
                        meg_sd[mk] = hf_sd[hk]
            else:
                # Standard attention layer
                mk = f"{mg_pfx}.self_attention.linear_qkv.layer_norm_weight"
                hk = f"{hf_pfx}.input_layernorm.weight"
                if hk in hf_sd:
                    meg_sd[mk] = _adjust_ln(hf_sd[hk])

                mk = f"{mg_pfx}.self_attention.linear_qkv.weight"
                q = hf_sd.get(f"{hf_pfx}.self_attn.q_proj.weight")
                k = hf_sd.get(f"{hf_pfx}.self_attn.k_proj.weight")
                v = hf_sd.get(f"{hf_pfx}.self_attn.v_proj.weight")
                if q is not None and k is not None and v is not None:
                    meg_sd[mk] = merge_attention_qkv(q, k, v, cfg)

                mk = f"{mg_pfx}.self_attention.linear_proj.weight"
                hk = f"{hf_pfx}.self_attn.o_proj.weight"
                if hk in hf_sd:
                    meg_sd[mk] = hf_sd[hk]

                for suffix in ["q_layernorm", "k_layernorm"]:
                    hf_sfx = suffix.replace("layernorm", "norm")
                    mk = f"{mg_pfx}.self_attention.{suffix}.weight"
                    hk = f"{hf_pfx}.self_attn.{hf_sfx}.weight"
                    if hk in hf_sd:
                        meg_sd[mk] = _adjust_ln(hf_sd[hk])

            # MLP (dense or MoE)
            convert_layer_mlp_hf2meg(hf_sd, meg_sd, layer_idx, hf_pfx, mg_pfx, cfg)

        # Final layernorm
        mk = "language_model.decoder.final_layernorm.weight"
        hk = "model.language_model.norm.weight"
        if hk in hf_sd:
            meg_sd[mk] = _adjust_ln(hf_sd[hk])

        # Output layer (only if untied)
        if cfg.untie:
            mk = "language_model.output_layer.weight"
            hk = "lm_head.weight"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

    def convert_to_megatron(self, hf_sd):
        """Convert HF state dict to full (unsharded) Megatron state dict."""
        meg_sd = {}
        self._convert_llm_hf2meg(hf_sd, meg_sd)
        convert_vision_hf2meg(hf_sd, meg_sd, self.cfg)
        convert_mtp_hf2meg(hf_sd, meg_sd, self.cfg)
        return meg_sd

    # -------------------------------------------------------------------------
    # Megatron -> HF naming conversion
    # -------------------------------------------------------------------------
    def _convert_llm_meg2hf(self, full_sd, hf_sd):
        cfg = self.cfg
        freq = cfg.linear_attention_freq
        emb_key = "language_model.embedding.word_embeddings.weight"
        if emb_key in full_sd:
            hf_sd["model.language_model.embed_tokens.weight"] = full_sd[emb_key]

        for layer_idx in range(cfg.num_layers):
            mg_pfx = f"language_model.decoder.layers.{layer_idx}"
            hf_pfx = f"model.language_model.layers.{layer_idx}"

            if is_gdn_layer(layer_idx, freq):
                mk = f"{mg_pfx}.self_attention.in_proj.layer_norm_weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.input_layernorm.weight"] = _restore_ln(full_sd[mk])

                mk = f"{mg_pfx}.self_attention.in_proj.weight"
                if mk in full_sd:
                    qkv, z, b, a = split_gdn_in_proj(full_sd[mk], cfg)
                    hf_sd[f"{hf_pfx}.linear_attn.in_proj_qkv.weight"] = qkv
                    hf_sd[f"{hf_pfx}.linear_attn.in_proj_z.weight"] = z
                    hf_sd[f"{hf_pfx}.linear_attn.in_proj_b.weight"] = b
                    hf_sd[f"{hf_pfx}.linear_attn.in_proj_a.weight"] = a

                mk = f"{mg_pfx}.self_attention.conv1d.weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.linear_attn.conv1d.weight"] = full_sd[mk]

                mk = f"{mg_pfx}.self_attention.out_proj.weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.linear_attn.out_proj.weight"] = full_sd[mk]

                mk = f"{mg_pfx}.self_attention.out_norm.weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.linear_attn.norm.weight"] = full_sd[mk] + 1.0

                for suffix in ["A_log", "dt_bias"]:
                    mk = f"{mg_pfx}.self_attention.{suffix}"
                    if mk in full_sd:
                        hf_sd[f"{hf_pfx}.linear_attn.{suffix}"] = full_sd[mk]
            else:
                mk = f"{mg_pfx}.self_attention.linear_qkv.layer_norm_weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.input_layernorm.weight"] = _restore_ln(full_sd[mk])

                mk = f"{mg_pfx}.self_attention.linear_qkv.weight"
                if mk in full_sd:
                    q, k, v = split_attention_qkv(full_sd[mk], cfg)
                    hf_sd[f"{hf_pfx}.self_attn.q_proj.weight"] = q
                    hf_sd[f"{hf_pfx}.self_attn.k_proj.weight"] = k
                    hf_sd[f"{hf_pfx}.self_attn.v_proj.weight"] = v

                mk = f"{mg_pfx}.self_attention.linear_proj.weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.self_attn.o_proj.weight"] = full_sd[mk]

                mk = f"{mg_pfx}.self_attention.q_layernorm.weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.self_attn.q_norm.weight"] = _restore_ln(full_sd[mk])
                mk = f"{mg_pfx}.self_attention.k_layernorm.weight"
                if mk in full_sd:
                    hf_sd[f"{hf_pfx}.self_attn.k_norm.weight"] = _restore_ln(full_sd[mk])

            convert_layer_mlp_meg2hf(full_sd, hf_sd, layer_idx, hf_pfx, mg_pfx, cfg)

        mk = "language_model.decoder.final_layernorm.weight"
        if mk in full_sd:
            hf_sd["model.language_model.norm.weight"] = _restore_ln(full_sd[mk])

        if cfg.untie:
            mk = "language_model.output_layer.weight"
            if mk in full_sd:
                hf_sd["lm_head.weight"] = full_sd[mk]

    def convert_to_hf(self, full_sd):
        """Convert merged Megatron state dict to HF naming convention."""
        hf_sd = {}
        self._convert_llm_meg2hf(full_sd, hf_sd)
        convert_vision_meg2hf(full_sd, hf_sd, self.cfg)
        convert_mtp_meg2hf(full_sd, hf_sd, self.cfg)
        return hf_sd

    # -------------------------------------------------------------------------
    # TP / PP / EP sharding
    # -------------------------------------------------------------------------
    def _split_tp(self, meg_sd):
        """Split a full Megatron state dict into TP shards."""
        cfg = self.cfg
        tp = cfg.tp
        shards = [{} for _ in range(tp)]

        vis_h = cfg.vision_hidden_size
        vis_heads = cfg.vision_num_attention_heads
        vis_head_dim = vis_h // vis_heads
        vis_qg = vis_heads

        for k, v in meg_sd.items():
            if not isinstance(v, torch.Tensor):
                for r in range(tp):
                    shards[r][k] = v
                continue

            # Embedding / output layer
            if k in (
                "language_model.embedding.word_embeddings.weight",
                "language_model.output_layer.weight",
            ):
                chunks = v.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r]
                continue

            # Vision model
            if "vision_model" in k:
                if "patch_embed" in k or "pos_embed" in k or "final_layernorm" in k:
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
                    chunks = v.chunk(tp, dim=1)
                    for r in range(tp):
                        shards[r][k] = chunks[r]
                elif "linear_fc1.weight" in k and "projection" not in k:
                    chunks = v.chunk(tp, dim=0)
                    for r in range(tp):
                        shards[r][k] = chunks[r]
                elif "linear_fc1.bias" in k and "projection" not in k:
                    chunks = v.chunk(tp, dim=0)
                    for r in range(tp):
                        shards[r][k] = chunks[r]
                elif "linear_proj.bias" in k or "linear_fc2.bias" in k:
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

            # LLM
            if "layer_norm_weight" in k or "layer_norm_bias" in k:
                for r in range(tp):
                    shards[r][k] = v
            elif "final_layernorm" in k:
                for r in range(tp):
                    shards[r][k] = v
            elif "pre_mlp_layernorm" in k:
                for r in range(tp):
                    shards[r][k] = v
            elif "in_proj.weight" in k:
                chunks = v.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "conv1d.weight" in k:
                conv_shards = _shard_gdn_conv1d(v, cfg)
                for r in range(tp):
                    shards[r][k] = conv_shards[r]
            elif "conv1d.bias" in k:
                conv_shards = _shard_gdn_conv1d(v, cfg)
                for r in range(tp):
                    shards[r][k] = conv_shards[r]
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
            elif "linear_qkv.weight" in k:
                # Simple dim-0 chunk / cat for TP sharding of the fused QKV weight.
                # This is valid for any TP size, including cases where num_query_groups < tp.
                chunks = v.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "linear_proj.weight" in k:
                chunks = v.chunk(tp, dim=1)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "q_layernorm" in k or "k_layernorm" in k:
                for r in range(tp):
                    shards[r][k] = v
            elif "mlp.experts.linear_fc1.weight" in k:
                viewed = v.view(2, cfg.moe_ffn_hidden_size, cfg.hidden_size)
                chunks = viewed.chunk(tp, dim=1)
                for r in range(tp):
                    shards[r][k] = chunks[r].reshape(-1, cfg.hidden_size)
            elif "shared_experts.linear_fc1.weight" in k:
                viewed = v.view(2, cfg.moe_shared_expert_intermediate_size, cfg.hidden_size)
                chunks = viewed.chunk(tp, dim=1)
                for r in range(tp):
                    shards[r][k] = chunks[r].reshape(-1, cfg.hidden_size)
            elif "linear_fc1.weight" in k:
                viewed = v.view(2, cfg.ffn_hidden_size, cfg.hidden_size)
                chunks = viewed.chunk(tp, dim=1)
                for r in range(tp):
                    shards[r][k] = chunks[r].reshape(-1, cfg.hidden_size)
            elif "linear_fc2.weight" in k:
                chunks = v.chunk(tp, dim=1)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            elif "router.weight" in k:
                for r in range(tp):
                    shards[r][k] = v
            elif "eh_proj.weight" in k:
                chunks = v.chunk(tp, dim=0)
                for r in range(tp):
                    shards[r][k] = chunks[r]
            else:
                for r in range(tp):
                    shards[r][k] = v

        return shards

    def _merge_tp(self, shards):
        """Merge TP shards into a single state dict for one PP/EP rank."""
        cfg = self.cfg
        tp = cfg.tp
        merged = {}
        all_keys = set()
        for s in shards:
            all_keys.update(s.keys())

        vis_h = cfg.vision_hidden_size
        vis_heads = cfg.vision_num_attention_heads
        vis_head_dim = vis_h // vis_heads
        vis_qg = vis_heads
        vis_gps = vis_qg // tp

        for k in sorted(all_keys):
            if "extra_state" in k:
                continue
            vals = [s[k] for s in shards if k in s]
            if not vals or not isinstance(vals[0], torch.Tensor):
                continue

            if "vision_model" in k:
                if "patch_embed" in k or "pos_embed" in k:
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
                    merged[k] = torch.cat(vals, dim=0)
                elif "linear_fc1.bias" in k and "projection" not in k:
                    merged[k] = torch.cat(vals, dim=0)
                elif "linear_proj.bias" in k or "linear_fc2.bias" in k:
                    merged[k] = vals[0]
                elif "layer_norm" in k or "final_layernorm" in k:
                    merged[k] = vals[0]
                elif "projection.encoder" in k:
                    if "linear_fc1" in k:
                        merged[k] = torch.cat(vals, dim=0)
                    elif "linear_fc2.weight" in k:
                        merged[k] = torch.cat(vals, dim=1)
                    else:
                        merged[k] = vals[0]
                else:
                    merged[k] = vals[0]
                continue

            if "embedding" in k:
                merged[k] = torch.cat(vals, dim=0)
            elif k == "language_model.output_layer.weight":
                merged[k] = torch.cat(vals, dim=0)
            elif "final_layernorm" in k:
                merged[k] = vals[0]
            elif "layer_norm_weight" in k or "layer_norm_bias" in k:
                merged[k] = vals[0]
            elif "pre_mlp_layernorm" in k:
                merged[k] = vals[0]
            elif "in_proj.weight" in k:
                merged[k] = torch.cat(vals, dim=0)
            elif "conv1d.weight" in k or "conv1d.bias" in k:
                merged[k] = _merge_gdn_conv1d(vals, cfg)
            elif "A_log" in k or "dt_bias" in k:
                merged[k] = torch.cat(vals, dim=0)
            elif "out_norm" in k:
                merged[k] = vals[0]
            elif "out_proj.weight" in k:
                merged[k] = torch.cat(vals, dim=1)
            elif "linear_qkv.weight" in k:
                # TP split is a simple contiguous dim-0 chunk, so merge is
                # just concatenation along dim 0.
                merged[k] = torch.cat(vals, dim=0)
            elif "linear_proj.weight" in k:
                merged[k] = torch.cat(vals, dim=1)
            elif "q_layernorm" in k or "k_layernorm" in k:
                merged[k] = vals[0]
            elif "mlp.experts.linear_fc1.weight" in k:
                viewed = [x.view(2, -1, cfg.hidden_size) for x in vals]
                merged[k] = torch.cat(viewed, dim=1).view(-1, cfg.hidden_size)
            elif "shared_experts.linear_fc1.weight" in k:
                viewed = [x.view(2, -1, cfg.hidden_size) for x in vals]
                merged[k] = torch.cat(viewed, dim=1).view(-1, cfg.hidden_size)
            elif "linear_fc1.weight" in k:
                viewed = [x.view(2, -1, cfg.hidden_size) for x in vals]
                merged[k] = torch.cat(viewed, dim=1).view(-1, cfg.hidden_size)
            elif "linear_fc2.weight" in k:
                merged[k] = torch.cat(vals, dim=1)
            elif "router.weight" in k:
                merged[k] = vals[0]
            elif "eh_proj.weight" in k:
                merged[k] = torch.cat(vals, dim=0)
            else:
                merged[k] = vals[0]

        return merged

    @staticmethod
    def _add_extra_states(sd):
        """Add _extra_state tensors for TE layers."""
        result = dict(sd)
        extra = torch.empty(0, dtype=torch.uint8)
        prefixes = set()
        for k in list(result.keys()):
            for pattern in constants.EXTRA_STATE_KEYS:
                if pattern in k:
                    idx = k.find(pattern)
                    if idx >= 0:
                        base = k[: idx + len(pattern)]
                        prefixes.add(base)
        for base in prefixes:
            es_key = f"{base}._extra_state"
            if es_key not in result:
                result[es_key] = extra.clone()
        return result

    def _adjust_embedding(self, meg_sd, ref_dir):
        """Adjust embedding vocab size to match a reference checkpoint."""
        if not self.adjust_embedding or ref_dir is None:
            return meg_sd

        ref_path = find_megatron_shard(ref_dir, 0, 0)
        if ref_path is None:
            return meg_sd
        ref_sd = torch.load(ref_path, map_location="cpu", weights_only=False)["model"]
        ref_emb = ref_sd.get("language_model.embedding.word_embeddings.weight")
        if ref_emb is None:
            return meg_sd

        emb_key = "language_model.embedding.word_embeddings.weight"
        current = meg_sd[emb_key]
        target_vocab_per_rank = ref_emb.shape[0]
        target_vocab = target_vocab_per_rank * self.cfg.tp

        if current.shape[0] == target_vocab:
            return meg_sd

        print(f"  Adjusting embedding vocab: {current.shape[0]} -> {target_vocab}")
        if current.shape[0] < target_vocab:
            pad_size = target_vocab - current.shape[0]
            padding = torch.zeros(pad_size, current.shape[1], dtype=current.dtype)
            meg_sd[emb_key] = torch.cat([current, padding], dim=0)
        else:
            meg_sd[emb_key] = current[:target_vocab]
        return meg_sd

    # -------------------------------------------------------------------------
    # Public run methods
    # -------------------------------------------------------------------------
    def run_hf2meg(self, hf_dir, save_dir, ref_dir=None, skip_value=False):
        """Run HF -> Megatron conversion and save release checkpoint."""
        cfg = self.cfg
        hf_dir = ensure_hf_path(hf_dir)
        print(f"Loading HF weights from {hf_dir}...")
        hf_sd = load_hf_weights(hf_dir)
        print(f"Loaded {len(hf_sd)} HF parameters")

        print("Converting to Megatron format...")
        meg_sd = self.convert_to_megatron(hf_sd)
        del hf_sd
        print(f"Converted to {len(meg_sd)} Megatron parameters")

        meg_sd = self._adjust_embedding(meg_sd, ref_dir)

        print(f"Splitting by PP (stages={cfg.pp})...")
        pp_stages = split_pp_layers(meg_sd, cfg)
        del meg_sd

        shards_dict = {}
        for pp_rank in range(cfg.pp):
            tp_shards = self._split_tp(pp_stages[pp_rank])
            for tp_rank in range(cfg.tp):
                shard = self._add_extra_states(tp_shards[tp_rank])
                shards_dict[(pp_rank, tp_rank)] = shard

        save_megatron_release_checkpoint(shards_dict, save_dir, cfg)
        print(f"Saved release checkpoint to {os.path.join(save_dir, 'release')}")

        if ref_dir:
            success = validate_hf2meg_against_ref(shards_dict, cfg, ref_dir, skip_value=skip_value)
            return success
        return True

    def run_meg2hf(self, meg_dir, save_dir, ref_dir=None, skip_value=False):
        """Run Megatron -> HF conversion and save safetensors."""
        cfg = self.cfg
        pp_merged = {}
        for pp_rank in range(cfg.pp):
            print(f"Loading PP stage {pp_rank}...")
            tp_merged = {}
            for tp_rank in range(cfg.tp):
                shard_path = find_megatron_shard(meg_dir, tp_rank, pp_rank)
                if shard_path is None:
                    raise FileNotFoundError(
                        f"Cannot find checkpoint for TP={tp_rank}, PP={pp_rank} in {meg_dir}"
                    )
                tp_merged[tp_rank] = load_megatron_shard(shard_path)
            pp_merged[pp_rank] = self._merge_tp([tp_merged[r] for r in range(cfg.tp)])
            del tp_merged

        print("Merging PP stages...")
        full_sd = merge_pp_layers(pp_merged, cfg)
        del pp_merged
        print(f"Merged Megatron keys: {len(full_sd)}")

        print("Converting to HF format...")
        hf_sd = self.convert_to_hf(full_sd)
        del full_sd
        print(f"Converted HF keys: {len(hf_sd)}")

        save_hf_checkpoint(hf_sd, save_dir)
        print(f"Saved HF checkpoint to {save_dir}")

        if ref_dir:
            success = validate_meg2hf_against_ref(hf_sd, cfg, ref_dir, skip_value=skip_value)
            return success
        return True


# -----------------------------------------------------------------------------
# Dense converter
# -----------------------------------------------------------------------------
class DenseConverter(BaseConverter):
    """Dense Qwen3.5 converter (no MoE-specific handling)."""

    def __init__(self, cfg: Config, adjust_embedding=False):
        super().__init__(cfg, adjust_embedding=adjust_embedding)


# -----------------------------------------------------------------------------
# MoE converter
# -----------------------------------------------------------------------------
class MoEConverter(BaseConverter):
    """MoE Qwen3.5 converter (extends dense with MoE MLP/EP handling)."""

    def __init__(self, cfg: Config, adjust_embedding=False):
        super().__init__(cfg, adjust_embedding=adjust_embedding)

    def run_hf2meg(self, hf_dir, save_dir, ref_dir=None, skip_value=False):
        cfg = self.cfg
        print(f"Loading HF weights from {hf_dir}...")
        hf_sd = load_hf_weights(hf_dir)
        print(f"Loaded {len(hf_sd)} HF parameters")

        print("Converting to Megatron format...")
        meg_sd = self.convert_to_megatron(hf_sd)
        del hf_sd
        print(f"Converted to {len(meg_sd)} Megatron parameters")

        meg_sd = self._adjust_embedding(meg_sd, ref_dir)

        print(f"Splitting by PP (stages={cfg.pp})...")
        pp_stages = split_pp_layers(meg_sd, cfg)
        del meg_sd

        print(f"Splitting by EP (ranks={cfg.ep}) and TP (ranks={cfg.tp})...")
        shards_dict = {}
        for pp_rank in range(cfg.pp):
            ep_shards = split_ep_experts(pp_stages[pp_rank], cfg)
            for ep_rank in range(cfg.ep):
                tp_shards = self._split_tp(ep_shards[ep_rank])
                for tp_rank in range(cfg.tp):
                    shard = self._add_extra_states(tp_shards[tp_rank])
                    shards_dict[(pp_rank, tp_rank, ep_rank)] = shard

        save_megatron_release_checkpoint(shards_dict, save_dir, cfg)
        print(f"Saved release checkpoint to {os.path.join(save_dir, 'release')}")

        if ref_dir:
            success = validate_hf2meg_against_ref(
                shards_dict, cfg, ref_dir, use_ep=True, skip_value=skip_value
            )
            return success
        return True

    def run_meg2hf(self, meg_dir, save_dir, ref_dir=None, skip_value=False):
        cfg = self.cfg
        pp_merged = {}
        for pp_rank in range(cfg.pp):
            print(f"Loading PP stage {pp_rank}...")
            tp_merged = {}
            for tp_rank in range(cfg.tp):
                ep_shards = []
                for ep_rank in range(cfg.ep):
                    path = find_megatron_shard(meg_dir, tp_rank, pp_rank, ep_rank)
                    if path is None:
                        raise FileNotFoundError(
                            f"Cannot find checkpoint for TP={tp_rank}, PP={pp_rank}, EP={ep_rank} in {meg_dir}"
                        )
                    ep_shards.append(load_megatron_shard(path))
                tp_merged[tp_rank] = merge_ep_experts(ep_shards, cfg)
                del ep_shards
            pp_merged[pp_rank] = self._merge_tp([tp_merged[r] for r in range(cfg.tp)])
            del tp_merged

        print("Merging PP stages...")
        full_sd = merge_pp_layers(pp_merged, cfg)
        del pp_merged
        print(f"Merged Megatron keys: {len(full_sd)}")

        print("Converting to HF format...")
        hf_sd = self.convert_to_hf(full_sd)
        del full_sd
        print(f"Converted HF keys: {len(hf_sd)}")

        save_hf_checkpoint(hf_sd, save_dir)
        print(f"Saved HF checkpoint to {save_dir}")

        if ref_dir:
            success = validate_meg2hf_against_ref(hf_sd, cfg, ref_dir, skip_value=skip_value)
            return success
        return True
