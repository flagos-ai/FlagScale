# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the local session selection/recompute code without CUDA."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def session_methods(monkeypatch):
    from flagscale.inference.core.diffusion.horizonkv import baseline_session as native

    encode = Mock()
    monkeypatch.setattr(native, "encode_video_latent", encode)
    ns = dict(get_clean_context_frames=native.GenerationSession.get_clean_context_frames,
              recompute_kv_cache=native.GenerationSession.recompute_kv_cache,
              encode_video_latent=encode)
    session = SimpleNamespace(
        params=SimpleNamespace(keep_first_frame=False, kv_cache_num_frames=3),
        kv_cache_num_frames=3,
        all_latents=torch.arange(12, dtype=torch.float32).reshape(1, 12, 1, 1, 1),
        block_idx=1,
        current_start_frame=3,
        gpu="cpu",
        resume_latents=None,
        conditional_dict={"prompt_embeds": "text"},
        frame_context_cache=deque([torch.full((1, 1, 3, 1, 1), i) for i in range(9)]),
    )
    session.get_clean_context_frames = lambda models: ns["get_clean_context_frames"](
        session, models
    )
    model = SimpleNamespace(
        blocks=[SimpleNamespace(self_attn=SimpleNamespace())],
        _prepare_blockwise_causal_attn_mask=Mock(return_value="mask"),
    )
    pipeline = SimpleNamespace(
        num_frame_per_block=3,
        frame_seq_length=1560,
        generator=SimpleNamespace(model=model),
        kv_cache1="kv",
        crossattn_cache="cross",
        _initialize_kv_cache=Mock(),
    )
    return session, SimpleNamespace(pipeline=pipeline, transformer=Mock(), vae_encoder=Mock()), ns


def test_second_block_uses_first_three_original_latents(session_methods):
    session, models, ns = session_methods
    assert session.get_clean_context_frames(models).flatten().tolist() == [0, 1, 2]
    ns["encode_video_latent"].assert_not_called()


def test_later_block_reencodes_boundary_and_keeps_recent_two(session_methods):
    session, models, ns = session_methods
    session.block_idx, session.current_start_frame = 3, 9
    ns["encode_video_latent"].return_value = (torch.full((1, 1, 1, 1), 99.0), None)
    # It must not silently degrade to a raw recent-three window [6, 7, 8].
    assert session.get_clean_context_frames(models).flatten().tolist() == [99, 7, 8]
    call = ns["encode_video_latent"].call_args
    assert call.args[1] == [None] * 55
    assert call.kwargs["stream"] is False
    assert torch.equal(call.kwargs["frames"], session.frame_context_cache[0][0].half())


def test_sink_option_differs_from_default(session_methods):
    session, models, ns = session_methods
    session.block_idx, session.current_start_frame = 3, 9
    session.params.keep_first_frame = True
    assert session.get_clean_context_frames(models).flatten().tolist() == [0, 7, 8]
    ns["encode_video_latent"].assert_not_called()


def test_recompute_resets_cache_once_and_forwards_three_clean_frames(session_methods):
    session, models, ns = session_methods
    seen = []

    def transformer(**kwargs):
        seen.append((models.pipeline.generator.model.block_mask, kwargs))

    models.transformer.side_effect = transformer
    assert ns["recompute_kv_cache"](session, models) == 3
    models.pipeline._initialize_kv_cache.assert_called_once()
    mask, kwargs = seen[0]
    assert mask == "mask"
    assert kwargs["noisy_image_or_video"].flatten().tolist() == [0, 1, 2]
    assert not kwargs["timestep"].any()
    assert kwargs["current_start"] == 3 * 1560
    assert models.pipeline.generator.model.block_mask is None


def test_first_block_has_no_history_recompute(session_methods):
    session, models, ns = session_methods
    session.block_idx = session.current_start_frame = 0
    assert ns["recompute_kv_cache"](session, models) == 0
    models.transformer.assert_not_called()


@pytest.mark.parametrize(
    "block,expected",
    [(1, [0, 1, 2]), (2, [0, 1, 2, 3, 4, 5]), (3, [99, 4, 5, 6, 7, 8])],
)
def test_six_context_frames_fill_then_reencode_boundary(session_methods, block, expected):
    session, models, ns = session_methods
    session.params.kv_cache_num_frames = session.kv_cache_num_frames = 6
    session.block_idx, session.current_start_frame = block, 3 * block
    session.frame_context_cache = deque(
        [torch.full((1, 1, 3, 1, 1), i) for i in range(33)], maxlen=21
    )
    ns["encode_video_latent"].return_value = (torch.full((1, 1, 1, 1), 99.0), None)
    assert session.get_clean_context_frames(models).flatten().tolist() == expected
    if block < 3:
        ns["encode_video_latent"].assert_not_called()
    else:
        # The deque holds pixels 12..32. Only its boundary pixel 12 is encoded;
        # the remaining five historical latents come directly from all_latents.
        encoded = ns["encode_video_latent"].call_args.kwargs["frames"]
        assert torch.equal(encoded, torch.full_like(encoded, 12))
        assert ns["recompute_kv_cache"](session, models) == 6
        call = models.transformer.call_args.kwargs
        assert call["noisy_image_or_video"].flatten().tolist() == expected
        assert call["current_start"] == 6 * 1560
        assert models.pipeline.generator.model.block_mask is None


def test_comparison_configs_keep_independent_attention_algorithms():
    root = Path(__file__).resolve().parents[2]
    configs = root / "examples/horizonkv/conf/inference"
    horizon = OmegaConf.load(configs / "horizonkv.yaml").engine
    baseline = OmegaConf.load(configs / "ar_diffusion_baseline.yaml").engine
    options = horizon.pipeline.from_pretrained.horizonkv
    assert (options.cache, options.topk, options.sparsity, options.sink, options.fixed) == (
        15,
        7,
        0.5,
        1,
        False,
    )
    assert "HorizonKVTransformation" in horizon.transformations
    assert baseline.pipeline["class"].endswith("ARDiffusionPipeline")
    assert baseline.pipeline.from_pretrained.context_latent_frames == 6
    assert "source_root" not in baseline.pipeline.from_pretrained
    assert "transformations" not in baseline
    assert (
        horizon.pipeline.from_pretrained.checkpoint == baseline.pipeline.from_pretrained.checkpoint
    )


@pytest.mark.parametrize("context", [0, -3, 1, 4, 6.0, True])
def test_invalid_context_rejected_before_runtime_side_effects(context):
    from flagscale.inference.core.diffusion.horizonkv.baseline import ARDiffusionRuntime

    with pytest.raises(ValueError, match="positive multiple of 3"):
        ARDiffusionRuntime("/missing", "/missing", context)


def test_ar_diffusion_five_step_shift_matches_original_schedule():
    from flagscale.inference.core.diffusion.horizonkv.model_spec import denoising_steps

    from flagscale.inference.core.diffusion.horizonkv.wan import scheduler as scheduler_module

    scheduler = scheduler_module.FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    timetable = torch.cat((scheduler.timesteps, torch.tensor([0])))
    times = timetable[1000 - torch.tensor(denoising_steps("krea-14b"))]
    assert times.tolist() == pytest.approx([1000, 937.5, 833.333333, 625, 0])
