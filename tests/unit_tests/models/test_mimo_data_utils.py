# Copyright (c) 2026, BAAI. All rights reserved.

"""Unit tests for flagscale.models.mimo.bridge.data.

Pure-CPU tests: the module under test only needs torch CPU tensors and the
stdlib-only sibling ``parallelism``; no distributed runtime, no
GPU, and no Megatron runtime is required.
"""

from dataclasses import FrozenInstanceError

import pytest
import torch

try:
    # Preferred path: the package import chain (used in the full FlagScale env).
    from flagscale.models.mimo.bridge.data import (
        ModuleDataRole,
        SamplingInfo,
        _batch_dim_for_tensor,
        drop_modality_inputs,
        get_sampling_info,
        is_patch_packed_visual_dict,
        needs_data_for_role,
        prepare_batch_for_module,
        should_drop_modality_inputs,
        slice_batch_for_module_dp,
    )
    from flagscale.models.mimo.bridge.parallelism import (
        MIMOLayout,
        MIMOParallelismConfig,
        ModuleParallelismConfig,
    )
except ImportError:  # pragma: no cover - exercised only without the full stack
    # Fallback for environments where the flagscale.models.mimo package init
    # chain cannot run (e.g. plain CI): load the module under test and its
    # stdlib-only sibling by path, inside a synthetic package so the
    # relative import ``from .parallelism import ...`` resolves.
    import importlib.util
    import sys
    import types
    from pathlib import Path

    _mimo_dir = Path(__file__).resolve().parents[3] / "flagscale" / "models" / "mimo" / "bridge"

    _pkg = types.ModuleType("_mimo_data_utils_tests")
    _pkg.__path__ = [str(_mimo_dir)]
    _pkg.__package__ = _pkg.__name__
    sys.modules[_pkg.__name__] = _pkg

    def _load_by_path(name):
        spec = importlib.util.spec_from_file_location(
            f"{_pkg.__name__}.{name}", _mimo_dir / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _parallelism_config = _load_by_path("parallelism")
    _module = _load_by_path("data")

    (
        ModuleDataRole,
        SamplingInfo,
        _batch_dim_for_tensor,
        drop_modality_inputs,
        get_sampling_info,
        is_patch_packed_visual_dict,
        needs_data_for_role,
        prepare_batch_for_module,
        should_drop_modality_inputs,
        slice_batch_for_module_dp,
    ) = (
        _module.ModuleDataRole,
        _module.SamplingInfo,
        _module._batch_dim_for_tensor,
        _module.drop_modality_inputs,
        _module.get_sampling_info,
        _module.is_patch_packed_visual_dict,
        _module.needs_data_for_role,
        _module.prepare_batch_for_module,
        _module.should_drop_modality_inputs,
        _module.slice_batch_for_module_dp,
    )
    (
        MIMOLayout,
        MIMOParallelismConfig,
        ModuleParallelismConfig,
    ) = (
        _parallelism_config.MIMOLayout,
        _parallelism_config.MIMOParallelismConfig,
        _parallelism_config.ModuleParallelismConfig,
    )


def _colocated_config() -> MIMOParallelismConfig:
    """Language and vision both span the full world (colocated)."""
    return MIMOParallelismConfig(
        module_parallelisms={
            "vision": ModuleParallelismConfig(data_parallel_size=4),  # full world 4
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=4, data_parallel_size=1
            ),  # full world 4
        },
        layout=MIMOLayout.COLOCATED,
    )


def _non_colocated_config() -> MIMOParallelismConfig:
    """Vision owns ranks [0, 4); language owns ranks [4, 12)."""
    return MIMOParallelismConfig(
        module_parallelisms={
            "vision": ModuleParallelismConfig(data_parallel_size=4),  # ranks [0, 4)
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
            ),  # ranks [4, 12)
        },
        layout=MIMOLayout.NON_COLOCATED,
    )


# ---------------------------------------------------------------------------
# ModuleDataRole: construction, validation, stage properties
# ---------------------------------------------------------------------------


def test_role_defaults_and_stage_properties():
    role = ModuleDataRole("vision")
    assert role.module_name == "vision"
    assert role.pp_rank == 0
    assert role.pp_size == 1
    assert not role.is_language
    assert role.is_first_stage
    assert role.is_last_stage


def test_role_language_and_stage_properties():
    role = ModuleDataRole("language", pp_rank=1, pp_size=3)
    assert role.is_language
    assert not role.is_first_stage
    assert not role.is_last_stage
    assert ModuleDataRole("language", pp_rank=2, pp_size=3).is_last_stage
    assert ModuleDataRole("language", pp_rank=0, pp_size=3).is_first_stage


def test_role_rejects_invalid_module_name():
    with pytest.raises(AssertionError, match="non-empty string"):
        ModuleDataRole("")
    with pytest.raises(AssertionError, match="non-empty string"):
        ModuleDataRole(123)


def test_role_rejects_invalid_pipeline_stage():
    with pytest.raises(AssertionError, match="pp_size must be >= 1"):
        ModuleDataRole("vision", pp_size=0)
    with pytest.raises(AssertionError, match="out of range"):
        ModuleDataRole("vision", pp_rank=2, pp_size=2)
    with pytest.raises(AssertionError, match="must be an integer"):
        ModuleDataRole("vision", pp_rank=0.5)


def test_role_is_immutable():
    role = ModuleDataRole("language")
    with pytest.raises(FrozenInstanceError):
        role.module_name = "vision"  # frozen dataclass


# ---------------------------------------------------------------------------
# Data-needed-by-role decision
# ---------------------------------------------------------------------------


def test_needs_data_language_all_pipeline_stages():
    for pp_rank in range(3):
        role = ModuleDataRole("language", pp_rank=pp_rank, pp_size=3)
        assert needs_data_for_role(role)


def test_needs_data_modality_only_first_stage():
    assert needs_data_for_role(ModuleDataRole("vision"))
    assert needs_data_for_role(ModuleDataRole("vision", pp_rank=0, pp_size=2))
    assert not needs_data_for_role(ModuleDataRole("vision", pp_rank=1, pp_size=2))
    # Unknown modality modules behave like vision.
    assert needs_data_for_role(ModuleDataRole("audio"))
    assert not needs_data_for_role(ModuleDataRole("audio", pp_rank=2, pp_size=4))


# ---------------------------------------------------------------------------
# SamplingInfo and get_sampling_info
# ---------------------------------------------------------------------------


def test_get_sampling_info_all_data_loading_ranks_share_sampler():
    # Language: any PP stage loads, sampler is unsharded.
    for pp_rank in range(3):
        info = get_sampling_info(ModuleDataRole("language", pp_rank=pp_rank, pp_size=3))
        assert info == SamplingInfo(sampler_dp_rank=0, sampler_dp_size=1, needs_data=True)
    # Vision first stage loads; deeper stages do not.
    assert get_sampling_info(ModuleDataRole("vision")).needs_data
    assert not get_sampling_info(ModuleDataRole("vision", pp_rank=1, pp_size=2)).needs_data


def test_sampling_info_defaults_and_validation():
    assert SamplingInfo().sampler_dp_rank == 0
    assert SamplingInfo().sampler_dp_size == 1
    assert not SamplingInfo().needs_data
    with pytest.raises(AssertionError, match="sampler_dp_size must be >= 1"):
        SamplingInfo(sampler_dp_size=0)
    with pytest.raises(AssertionError, match="out of range"):
        SamplingInfo(sampler_dp_rank=2, sampler_dp_size=2)


# ---------------------------------------------------------------------------
# DP slicing of tensors (batch dim, [3, B, S] position_ids, divisibility)
# ---------------------------------------------------------------------------


def test_slice_regular_tensor_along_dim_zero():
    tokens = torch.arange(12 * 8).reshape(12, 8)
    sliced = slice_batch_for_module_dp({"tokens": tokens}, dp_rank=1, dp_size=3)
    assert sliced["tokens"].shape == (4, 8)
    torch.testing.assert_close(sliced["tokens"], tokens[4:8])


def test_slice_regular_tensor_first_and_last_shards():
    tokens = torch.arange(12 * 8).reshape(12, 8)
    first = slice_batch_for_module_dp({"tokens": tokens}, dp_rank=0, dp_size=3)["tokens"]
    last = slice_batch_for_module_dp({"tokens": tokens}, dp_rank=2, dp_size=3)["tokens"]
    torch.testing.assert_close(first, tokens[0:4])
    torch.testing.assert_close(last, tokens[8:12])


def test_slice_position_ids_three_b_s_layout_uses_batch_dim_one():
    # Qwen-VL MRoPE position_ids are [3, batch, seq].
    position_ids = torch.arange(3 * 12 * 8).reshape(3, 12, 8)
    sliced = slice_batch_for_module_dp({"position_ids": position_ids}, dp_rank=1, dp_size=3)
    assert sliced["position_ids"].shape == (3, 4, 8)
    torch.testing.assert_close(sliced["position_ids"], position_ids[:, 4:8, :])


def test_slice_2d_position_ids_uses_dim_zero():
    # A 2-D position_ids (dim() < 3) falls back to batch dim 0.
    position_ids = torch.arange(12 * 8).reshape(12, 8)
    sliced = slice_batch_for_module_dp({"position_ids": position_ids}, dp_rank=2, dp_size=3)
    assert sliced["position_ids"].shape == (4, 8)
    torch.testing.assert_close(sliced["position_ids"], position_ids[8:12])


def test_batch_dim_for_tensor():
    assert _batch_dim_for_tensor("tokens", torch.zeros(8, 16)) == 0
    assert _batch_dim_for_tensor("labels", torch.zeros(8, 16)) == 0
    assert _batch_dim_for_tensor("position_ids", torch.zeros(8, 16)) == 0
    assert _batch_dim_for_tensor("position_ids", torch.zeros(3, 8, 16)) == 1
    # A [3, 8] position_ids is not the [3, B, S] layout (dim() < 3).
    assert _batch_dim_for_tensor("position_ids", torch.zeros(3, 8)) == 0


def test_slice_rejects_non_divisible_tensor():
    batch = {"tokens": torch.zeros(10, 8)}
    with pytest.raises(AssertionError, match="not divisible by DP size 3"):
        slice_batch_for_module_dp(batch, dp_rank=0, dp_size=3)


def test_slice_rejects_invalid_dp_args():
    batch = {"tokens": torch.zeros(8, 8)}
    with pytest.raises(AssertionError, match="positive integer"):
        slice_batch_for_module_dp(batch, dp_rank=0, dp_size=0)
    with pytest.raises(AssertionError, match="out of range"):
        slice_batch_for_module_dp(batch, dp_rank=3, dp_size=3)


def test_slice_dp_size_one_returns_copy_with_same_values():
    batch = {"tokens": torch.arange(8).reshape(4, 2), "step": 3}
    sliced = slice_batch_for_module_dp(batch, dp_rank=0, dp_size=1)
    assert sliced is not batch
    assert sliced.keys() == batch.keys()
    torch.testing.assert_close(sliced["tokens"], batch["tokens"])
    assert sliced["step"] == 3


# ---------------------------------------------------------------------------
# DP slicing of nested dicts, lists, and pass-through values
# ---------------------------------------------------------------------------


def test_slice_nested_dicts_recursively():
    batch = {
        "tokens": torch.arange(8 * 4).reshape(8, 4),
        "modality_inputs": {
            "vision": {
                "pixel_values": torch.arange(8 * 3 * 4).reshape(8, 3, 4),
                "encoder_kwargs": {"dtype": torch.float32},
            }
        },
    }
    sliced = slice_batch_for_module_dp(batch, dp_rank=1, dp_size=2)
    assert sliced["tokens"].shape == (4, 4)
    assert sliced["modality_inputs"]["vision"]["pixel_values"].shape == (4, 3, 4)
    torch.testing.assert_close(
        sliced["modality_inputs"]["vision"]["pixel_values"],
        batch["modality_inputs"]["vision"]["pixel_values"][4:8],
    )
    assert sliced["modality_inputs"]["vision"]["encoder_kwargs"]["dtype"] == torch.float32


def test_slice_divisible_list_is_sliced():
    batch = {"segments": [0, 1, 2, 3, 4, 5]}
    sliced = slice_batch_for_module_dp(batch, dp_rank=1, dp_size=3)
    assert sliced["segments"] == [2, 3]


def test_slice_non_divisible_list_passes_through_as_metadata():
    batch = {"video_meta": ["a", "b", "c"]}
    sliced = slice_batch_for_module_dp(batch, dp_rank=0, dp_size=2)
    assert sliced["video_meta"] == ["a", "b", "c"]


def test_slice_passes_through_scalars_and_none():
    batch = {"tokens": torch.zeros(4, 4), "step": 7, "label": None, "mode": "train"}
    sliced = slice_batch_for_module_dp(batch, dp_rank=0, dp_size=2)
    assert sliced["step"] == 7
    assert sliced["label"] is None
    assert sliced["mode"] == "train"
    assert sliced["tokens"].shape == (2, 4)


# ---------------------------------------------------------------------------
# Patch-packed visual inputs: {hidden_states, grid_thw} joint slicing
# ---------------------------------------------------------------------------


def _patch_packed_batch(grid_rows, feature_dim=6, merge_unit=1):
    """Build a patch-packed visual dict; grid_rows are (t, h, w) triples."""
    grid = torch.tensor(grid_rows, dtype=torch.long)
    patches_per_image = [int(t * h * w // merge_unit) for t, h, w in grid_rows]
    hs = torch.arange(sum(patches_per_image) * feature_dim).reshape(
        sum(patches_per_image), feature_dim
    )
    return {
        "hidden_states": hs,
        "grid_thw": grid,
        "image_sizes": [(224, 224), (336, 336), (224, 224), (336, 336)],
    }


def test_is_patch_packed_visual_dict_detection():
    packed = _patch_packed_batch([(1, 2, 2), (1, 2, 2)])
    assert is_patch_packed_visual_dict(packed)
    assert not is_patch_packed_visual_dict({"hidden_states": torch.zeros(4, 6)})
    assert not is_patch_packed_visual_dict({"grid_thw": torch.zeros(4, 3)})
    assert not is_patch_packed_visual_dict({"hidden_states": 5, "grid_thw": torch.zeros(4, 3)})
    assert not is_patch_packed_visual_dict(
        {"hidden_states": torch.zeros(4, 6), "grid_thw": torch.zeros(4, 2)}
    )
    assert not is_patch_packed_visual_dict("not a dict")


def test_slice_patch_packed_visual_dict_jointly():
    # grid_thw rows: 4 images with 4, 6, 8, 2 patches -> 20 patch rows total.
    packed = _patch_packed_batch([(1, 2, 2), (1, 2, 3), (2, 2, 2), (1, 1, 2)])
    sliced = slice_batch_for_module_dp({"vision_input": packed}, dp_rank=1, dp_size=2)[
        "vision_input"
    ]
    assert sliced["grid_thw"].shape == (2, 3)
    torch.testing.assert_close(sliced["grid_thw"], packed["grid_thw"][2:4])
    # Shard 1 covers images 2,3 -> patches [10, 20).
    torch.testing.assert_close(sliced["hidden_states"], packed["hidden_states"][10:20])
    # Non-patch metadata passes through unchanged.
    assert sliced["image_sizes"] == packed["image_sizes"]


def test_slice_patch_packed_visual_dict_first_shard():
    packed = _patch_packed_batch([(1, 2, 2), (1, 2, 3), (2, 2, 2), (1, 1, 2)])
    sliced = slice_batch_for_module_dp({"vision_input": packed}, dp_rank=0, dp_size=2)[
        "vision_input"
    ]
    torch.testing.assert_close(sliced["grid_thw"], packed["grid_thw"][0:2])
    torch.testing.assert_close(sliced["hidden_states"], packed["hidden_states"][0:10])


def test_slice_patch_packed_visual_dict_uneven_patch_counts_per_shard():
    # Images 0,1 have 2 patches each; images 2,3 have 6 patches each.  Shards
    # must follow the per-image boundary even though patch counts differ.
    packed = _patch_packed_batch([(1, 1, 2), (1, 2, 1), (2, 3, 1), (3, 2, 1)])
    sliced = slice_batch_for_module_dp({"vision_input": packed}, dp_rank=1, dp_size=2)[
        "vision_input"
    ]
    assert sliced["grid_thw"].shape == (2, 3)
    torch.testing.assert_close(sliced["grid_thw"], packed["grid_thw"][2:4])
    torch.testing.assert_close(sliced["hidden_states"], packed["hidden_states"][4:16])


def test_slice_patch_packed_rejects_non_divisible_image_count():
    packed = _patch_packed_batch([(1, 2, 2), (1, 2, 2), (1, 2, 2)])
    with pytest.raises(AssertionError, match="not divisible by encoder DP"):
        slice_batch_for_module_dp({"vision_input": packed}, dp_rank=0, dp_size=2)


def test_slice_patch_packed_rejects_hidden_states_mismatch():
    packed = _patch_packed_batch([(1, 2, 2), (1, 2, 2)])
    packed["hidden_states"] = torch.zeros(packed["hidden_states"].size(0) + 1, 6)
    with pytest.raises(AssertionError, match="expected hidden_states dim 0"):
        slice_batch_for_module_dp({"vision_input": packed}, dp_rank=0, dp_size=2)


# ---------------------------------------------------------------------------
# Dropping modality inputs on language-only ranks
# ---------------------------------------------------------------------------


def test_should_drop_modality_inputs_only_non_colocated_language():
    role = ModuleDataRole("language")
    assert should_drop_modality_inputs(role, _non_colocated_config(), world_size=12)
    assert not should_drop_modality_inputs(role, _colocated_config(), world_size=4)


def test_should_drop_modality_inputs_resolves_auto_layout():
    config = MIMOParallelismConfig(
        module_parallelisms=_non_colocated_config().module_parallelisms,
        layout=MIMOLayout.AUTO,
    )
    assert should_drop_modality_inputs(ModuleDataRole("language"), config, world_size=12)


def test_should_drop_modality_inputs_rejects_non_language_role():
    with pytest.raises(AssertionError, match="only meaningful for language ranks"):
        should_drop_modality_inputs(ModuleDataRole("vision"), _non_colocated_config(), 12)


def test_drop_modality_inputs_sets_none_and_does_not_mutate_input():
    batch = {
        "tokens": torch.zeros(4, 8),
        "modality_inputs": {"vision": {"x": 1}},
        "imgs": torch.zeros(3080, 16),
        "videos": torch.zeros(4, 3, 16, 16),
        "image_thw_grids": torch.ones(4, 3, dtype=torch.long),
        "video_thw_grids": torch.ones(4, 3, dtype=torch.long),
    }
    dropped = drop_modality_inputs(batch)
    assert dropped["modality_inputs"] is None
    # Raw patch-packed Qwen-VL modality keys are dropped too: their leading
    # dim is the total patch count, not the sample batch, so the sample-DP
    # slicer must never see them.
    assert dropped["imgs"] is None
    assert dropped["videos"] is None
    assert dropped["image_thw_grids"] is None
    assert dropped["video_thw_grids"] is None
    assert batch["modality_inputs"] == {"vision": {"x": 1}}  # input untouched
    assert batch["imgs"] is not None
    assert dropped is not batch
    assert "tokens" in dropped


def test_prepare_batch_for_module_non_colocated_language_rank():
    # Language-only rank: raw modality inputs dropped, then DP-sliced.
    batch = {
        "tokens": torch.arange(8 * 4).reshape(8, 4),
        "position_ids": torch.arange(3 * 8 * 4).reshape(3, 8, 4),
        "modality_inputs": {
            "vision": {
                "pixel_values": torch.zeros(8, 3, 4),
            }
        },
    }
    role = ModuleDataRole("language", pp_rank=0, pp_size=2)
    prepared = prepare_batch_for_module(
        batch,
        dp_rank=1,
        dp_size=2,
        role=role,
        config=_non_colocated_config(),
        world_size=12,
    )
    assert prepared["modality_inputs"] is None
    assert prepared["tokens"].shape == (4, 4)
    torch.testing.assert_close(prepared["tokens"], batch["tokens"][4:8])
    assert prepared["position_ids"].shape == (3, 4, 4)
    torch.testing.assert_close(prepared["position_ids"], batch["position_ids"][:, 4:8, :])


def test_prepare_batch_for_module_colocated_vision_rank_keeps_modality_inputs():
    # Colocated vision first stage: raw modality inputs kept and sliced.
    batch = {
        "tokens": torch.zeros(8, 4),
        "modality_inputs": {
            "vision": {
                "pixel_values": torch.arange(8 * 3).reshape(8, 3),
            }
        },
    }
    role = ModuleDataRole("vision", pp_rank=0, pp_size=1)
    prepared = prepare_batch_for_module(
        batch, dp_rank=1, dp_size=2, role=role, config=_colocated_config(), world_size=4
    )
    assert prepared["modality_inputs"] is not None
    torch.testing.assert_close(
        prepared["modality_inputs"]["vision"]["pixel_values"],
        batch["modality_inputs"]["vision"]["pixel_values"][4:8],
    )


def test_prepare_batch_for_module_colocated_language_rank_keeps_modality_inputs():
    # Colocated language ranks also host the vision module, so the raw
    # modality inputs are kept (their vision DP shard consumes them).
    batch = {
        "tokens": torch.zeros(8, 4),
        "modality_inputs": {"vision": {"pixel_values": torch.zeros(8, 3)}},
    }
    role = ModuleDataRole("language", pp_rank=0, pp_size=1)
    prepared = prepare_batch_for_module(
        batch, dp_rank=1, dp_size=2, role=role, config=_colocated_config(), world_size=4
    )
    assert prepared["modality_inputs"] is not None
    assert prepared["modality_inputs"]["vision"]["pixel_values"].shape == (4, 3)
