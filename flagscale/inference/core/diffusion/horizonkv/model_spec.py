# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit distilled model identities; geometry alone cannot identify AR weights."""

from pathlib import Path

MODEL_CONFIGS = {
    "self-forcing-1.3b": dict(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30),
    "krea-14b": dict(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40),
}

__all__ = [
    "MODEL_CONFIGS",
    "model_config",
    "denoising_steps",
    "validate_krea_checkpoint",
    "kv_memory_estimate",
]


def model_config(variant):
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_variant {variant!r}; choose {tuple(MODEL_CONFIGS)}")
    return dict(
        model_type="t2v",
        freq_dim=256,
        in_dim=16,
        out_dim=16,
        text_len=512,
        eps=1e-6,
        **MODEL_CONFIGS[variant],
    )


def denoising_steps(variant):
    model_config(variant)
    # Krea's v2v.get_denoising_schedule uses linspace(1000, 0, steps).long().
    # Both lists are indices BEFORE the native FlowMatchScheduler's shift=5 warp.
    return [1000, 750, 500, 250, 0] if variant == "krea-14b" else [1000, 750, 500, 250]


def validate_krea_checkpoint(path):
    """Check the original complete wrapper checkpoint using only its tensor header."""
    from safetensors import safe_open

    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("krea-14b requires krea-realtime-video-14b.safetensors")
    expected = {
        "model.patch_embedding.weight": (5120, 16, 1, 2, 2),
        "model.blocks.0.self_attn.k.weight": (5120, 5120),
        "model.blocks.39.self_attn.k.weight": (5120, 5120),
        "model.blocks.39.ffn.0.weight": (13824, 5120),
    }
    with safe_open(path, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        for key, shape in expected.items():
            if key not in keys or tuple(tensors.get_slice(key).get_shape()) != shape:
                raise ValueError(f"Not a Krea 14B wrapper checkpoint: expected {key} {shape}")
        if any(not key.startswith("model.") for key in keys):
            raise ValueError("Expected a generator-only Krea safetensors checkpoint")
    return {"format": "safetensors", "tensor_count": len(keys), "bytes": path.stat().st_size}


def kv_memory_estimate(variant, *, latent_frames, cache_frames, height=480, width=832):
    """Raw BF16 K+V storage only; excludes allocator, summaries, staging, and weights."""
    config = model_config(variant)
    tokens = (height // 16) * (width // 16)
    per_frame = 2 * 2 * config["num_layers"] * config["dim"] * tokens
    return dict(
        bytes_per_latent_frame=per_frame,
        host_history_bytes=latent_frames * per_frame,
        gpu_kv_bytes=cache_frames * per_frame,
    )
