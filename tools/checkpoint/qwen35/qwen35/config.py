"""Unified configuration loader for Qwen3.5 checkpoint conversion."""

import json
import os
from pathlib import Path

import torch
import yaml


def _flatten_config(raw):
    """Merge legacy system/model sections into a flat dict.

    Top-level keys take precedence over nested section keys.
    """
    cfg = dict(raw)
    if isinstance(raw.get("system"), dict):
        for key, value in raw["system"].items():
            cfg.setdefault(key, value)
    if isinstance(raw.get("model"), dict):
        for key, value in raw["model"].items():
            cfg.setdefault(key, value)
    return cfg


class Config:
    """Flat config built from training yaml."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        cfg = _flatten_config(raw)

        self.tp = cfg.get("tensor_model_parallel_size", 1)
        self.pp = cfg.get("pipeline_model_parallel_size", 1)
        self.ep = cfg.get("expert_model_parallel_size", 1)

        self.num_layers = _require(cfg, "num_layers")
        self.hidden_size = _require(cfg, "hidden_size")
        self.num_attention_heads = _require(cfg, "num_attention_heads")
        self.num_query_groups = _require(cfg, "num_query_groups")
        self.kv_channels = _require(cfg, "kv_channels")
        self.attention_output_gate = _require(cfg, "attention_output_gate")
        self.untie = _require(cfg, "untie_embeddings_and_output_weights")

        self.linear_attention_freq = _require(cfg, "linear_attention_freq")
        self.linear_key_head_dim = _require(cfg, "linear_key_head_dim")
        self.linear_value_head_dim = _require(cfg, "linear_value_head_dim")
        self.linear_num_key_heads = _require(cfg, "linear_num_key_heads")
        self.linear_num_value_heads = _require(cfg, "linear_num_value_heads")
        self.qk_dim = self.linear_key_head_dim * self.linear_num_key_heads
        self.v_dim = self.linear_value_head_dim * self.linear_num_value_heads

        # MoE params: num_experts == 0 (or absent) means dense
        self.num_experts = cfg.get("num_experts", 0) or 0
        if self.is_moe:
            self.moe_ffn_hidden_size = _require(cfg, "moe_ffn_hidden_size")
            self.moe_shared_expert_intermediate_size = _require(
                cfg, "moe_shared_expert_intermediate_size"
            )
            # Dense FFN size may be omitted in MoE yamls; fall back to MoE size
            self.ffn_hidden_size = cfg.get("ffn_hidden_size", self.moe_ffn_hidden_size)
        else:
            self.ffn_hidden_size = _require(cfg, "ffn_hidden_size")
            self.moe_ffn_hidden_size = 0
            self.moe_shared_expert_intermediate_size = 0

        self.vision_num_layers = _require(cfg, "vision_num_layers")
        self.vision_hidden_size = _require(cfg, "vision_hidden_size")
        self.vision_num_attention_heads = _require(cfg, "vision_num_attention_heads")
        self.vision_ffn_hidden_size = _require(cfg, "vision_ffn_hidden_size")
        self.patch_size = _require(cfg, "patch_size")
        self.temporal_patch_size = 2  # hardcoded in get_vision_model_config
        self.use_linear_proj = cfg.get("vision_patch_embed_linear", True)

    @property
    def is_moe(self):
        return self.num_experts is not None and self.num_experts > 0


def _require(d, key):
    """Return d[key], raising a clear error if the key is missing."""
    if key not in d:
        raise ValueError(f"Missing required config key: {key}")
    return d[key]


def _read_hf_config(hf_dir):
    """Load HF config.json, handling nested text_config."""
    config_path = Path(hf_dir) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    # Some HF models store text config under "text_config"
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg = cfg["text_config"]
    return cfg


def detect_model_type(hf_dir=None, meg_dir=None, yaml_path=None):
    """Detect whether the model is dense or MoE from inputs.

    Priority:
    1. YAML model.num_experts
    2. HF config.json num_experts / moe_intermediate_size
    3. Megatron checkpoint keys (router / experts)
    """
    # 1. YAML
    if yaml_path is not None and os.path.exists(yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        cfg = _flatten_config(raw)
        if cfg.get("num_experts") is not None:
            return "moe" if int(cfg["num_experts"]) > 0 else "dense"

    # 2. HF config
    if hf_dir is not None:
        cfg = _read_hf_config(hf_dir)
        if cfg is not None:
            if cfg.get("num_experts") is not None and int(cfg.get("num_experts", 0)) > 0:
                return "moe"
            if cfg.get("moe_intermediate_size") is not None:
                return "moe"

    # 3. Megatron checkpoint keys
    if meg_dir is not None:
        release_dir = os.path.join(meg_dir, "release")
        if os.path.isdir(release_dir):
            search_dir = release_dir
        else:
            search_dir = meg_dir
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if not fname.endswith(".pt"):
                    continue
                try:
                    sd = torch.load(
                        os.path.join(root, fname), map_location="cpu", weights_only=False
                    )
                    model_keys = set(sd.get("model", {}).keys())
                    if any("mlp.router.weight" in k for k in model_keys):
                        return "moe"
                    if any("mlp.experts.linear_fc1.weight0" in k for k in model_keys):
                        return "moe"
                except Exception:
                    continue
            # Only check first shard found to avoid walking huge dirs
            break

    return "dense"
