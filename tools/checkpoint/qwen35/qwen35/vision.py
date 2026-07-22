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

"""Vision model conversion helpers (HF <-> Megatron)."""


def convert_vision_hf2meg(hf_sd, meg_sd, cfg):
    """Convert vision model parameters from HF to Megatron."""
    mg_pfx = "vision_model"
    hf_pfx = "model.visual"
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads

    for s in ["weight", "bias"]:
        hk = f"{hf_pfx}.patch_embed.proj.{s}"
        mk = f"{mg_pfx}.patch_embed.proj.{s}"
        if hk in hf_sd:
            val = hf_sd[hk]
            if s == "weight" and cfg.use_linear_proj:
                # HF uses nn.Conv3d, Megatron uses nn.Linear: flatten spatial dims
                val = val.view(cfg.vision_hidden_size, -1)
            meg_sd[mk] = val

    hk = f"{hf_pfx}.pos_embed.weight"
    mk = f"{mg_pfx}.pos_embed.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

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


def convert_vision_meg2hf(full_sd, hf_sd, cfg):
    """Convert vision model parameters from Megatron to HF."""
    mg_pfx = "vision_model"
    hf_pfx = "model.visual"
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads

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

    mk = f"{mg_pfx}.pos_embed.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.pos_embed.weight"] = full_sd[mk]

    for i in range(cfg.vision_num_layers):
        mg_blk = f"{mg_pfx}.decoder.layers.{i}"
        hf_blk = f"{hf_pfx}.blocks.{i}"

        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.self_attention.linear_qkv.layer_norm_{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.norm1.{s}"] = full_sd[mk]

        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.mlp.linear_fc1.layer_norm_{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.norm2.{s}"] = full_sd[mk]

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

        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.self_attention.linear_proj.{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.attn.proj.{s}"] = full_sd[mk]

        for s in ["weight", "bias"]:
            for layer in ["linear_fc1", "linear_fc2"]:
                mk = f"{mg_blk}.mlp.{layer}.{s}"
                if mk in full_sd:
                    hf_sd[f"{hf_blk}.mlp.{layer}.{s}"] = full_sd[mk]

    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.projection.encoder.linear_fc1.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.linear_fc1.{s}"] = full_sd[mk]
        mk = f"{mg_pfx}.projection.encoder.linear_fc2.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.linear_fc2.{s}"] = full_sd[mk]

    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.decoder.final_layernorm.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.norm.{s}"] = full_sd[mk]
