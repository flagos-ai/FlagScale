# Copyright (c) 2026, BAAI. All rights reserved.

"""Unit tests for flagscale.models.mimo.bridge.recipe.qwen35.

Pure-CPU tests for the Qwen3.5 non-colocated grid contract: the supported
2+6 families, the fail-fast validators, and the batch contract.  The module
under test is dependency-free (stdlib + mimo_parallelism_config only), so no
torch/Megatron runtime is required.
"""

import pytest

try:
    from flagscale.models.mimo.bridge.parallelism import (
        MIMOParallelismConfig,
        ModuleParallelismConfig,
    )
    from flagscale.models.mimo.bridge.recipe.qwen35 import (
        IMAGES_MODULE_NAME,
        QWEN35_GRID_SUPPORTED_FAMILIES,
        QWEN35_GRID_WORLD_SIZE,
        build_qwen35_grid_config_from_args,
        compute_qwen35_grid_sequence_parallel,
        compute_qwen35_pipeline_layer_split,
        qwen35_grid_data_contract,
        validate_qwen35_grid_config,
    )
except ImportError:  # pragma: no cover - exercised only without the full stack
    import importlib.util
    import sys
    import types
    from pathlib import Path

    _root = Path(__file__).resolve().parents[3] / "flagscale" / "models" / "mimo" / "bridge"

    # Register the flagscale.models.mimo.bridge package path WITHOUT executing
    # its __init__.py (which imports torch): relative imports inside the loaded
    # modules then resolve against the real module files.
    _flagscale_pkg = types.ModuleType("flagscale")
    _flagscale_pkg.__path__ = []
    sys.modules.setdefault("flagscale", _flagscale_pkg)
    _models_pkg = types.ModuleType("flagscale.models")
    _models_pkg.__path__ = []
    sys.modules.setdefault("flagscale.models", _models_pkg)
    _mimo_pkg = types.ModuleType("flagscale.models.mimo")
    _mimo_pkg.__path__ = []
    sys.modules.setdefault("flagscale.models.mimo", _mimo_pkg)
    _bridge_pkg = types.ModuleType("flagscale.models.mimo.bridge")
    _bridge_pkg.__path__ = [str(_root)]
    sys.modules.setdefault("flagscale.models.mimo.bridge", _bridge_pkg)
    _recipe_pkg = types.ModuleType("flagscale.models.mimo.bridge.recipe")
    _recipe_pkg.__path__ = [str(_root / "recipe")]
    sys.modules.setdefault("flagscale.models.mimo.bridge.recipe", _recipe_pkg)

    def _load(name, rel_path):
        # Load under the full package name so relative imports resolve to the
        # SAME module instances (a second top-level copy would create distinct
        # enum classes and break identity checks).
        _full_name = f"flagscale.models.mimo.bridge.{name}"
        if _full_name in sys.modules:
            return sys.modules[_full_name]
        _path = _root / rel_path
        _spec = importlib.util.spec_from_file_location(_full_name, _path)
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_full_name] = _module
        _spec.loader.exec_module(_module)
        return _module

    _cfg_mod = _load("parallelism", "parallelism.py")
    _grid_mod = _load("recipe.qwen35", "recipe/qwen35.py")
    MIMOParallelismConfig = _cfg_mod.MIMOParallelismConfig
    ModuleParallelismConfig = _cfg_mod.ModuleParallelismConfig
    (
        IMAGES_MODULE_NAME,
        QWEN35_GRID_SUPPORTED_FAMILIES,
        QWEN35_GRID_WORLD_SIZE,
        build_qwen35_grid_config_from_args,
        compute_qwen35_grid_sequence_parallel,
        compute_qwen35_pipeline_layer_split,
        qwen35_grid_data_contract,
        validate_qwen35_grid_config,
    ) = (
        _grid_mod.IMAGES_MODULE_NAME,
        _grid_mod.QWEN35_GRID_SUPPORTED_FAMILIES,
        _grid_mod.QWEN35_GRID_WORLD_SIZE,
        _grid_mod.build_qwen35_grid_config_from_args,
        _grid_mod.compute_qwen35_grid_sequence_parallel,
        _grid_mod.compute_qwen35_pipeline_layer_split,
        _grid_mod.qwen35_grid_data_contract,
        _grid_mod.validate_qwen35_grid_config,
    )


def _grid_config(images, language, layout="non_colocated"):
    return MIMOParallelismConfig(
        module_parallelisms={
            IMAGES_MODULE_NAME: images,
            "language": language,
        },
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Supported families: the canonical 8 layouts validate and return 1..8.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family_index",
    list(range(1, len(QWEN35_GRID_SUPPORTED_FAMILIES) + 1)),
)
def test_all_supported_families_validate(family_index):
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[family_index - 1]
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE) == family_index


def test_family_signatures_match_mission_matrix():
    """The canonical family list matches the mission's 8 layouts exactly."""
    signatures = []
    for images, language in QWEN35_GRID_SUPPORTED_FAMILIES:
        signatures.append(
            (
                images.tensor_model_parallel_size,
                images.data_parallel_size,
                language.tensor_model_parallel_size,
                language.pipeline_model_parallel_size,
                language.data_parallel_size,
            )
        )
    assert signatures == [
        (2, 1, 1, 1, 6),
        (2, 1, 1, 2, 3),
        (2, 1, 1, 3, 2),
        (2, 1, 1, 6, 1),
        (2, 1, 2, 1, 3),
        (2, 1, 2, 3, 1),
        (1, 2, 1, 1, 6),
        (1, 2, 1, 3, 2),
    ]


# ---------------------------------------------------------------------------
# World size and rank-span constraints.
# ---------------------------------------------------------------------------


def test_rejects_world_size_not_8():
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6),
    )
    with pytest.raises(ValueError, match="world size 8"):
        validate_qwen35_grid_config(config, 16)


def test_rejects_images_not_on_ranks_0_2():
    # Non-zero images rank_offset breaks the tiling invariant first; the
    # images rank-span check is defense-in-depth (see language variant).
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1, rank_offset=2),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=4),
    )
    with pytest.raises(ValueError, match="tile"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_language_not_on_ranks_2_8():
    # Overlapping the images span triggers the tiling overlap error before the
    # language rank-span check; the span check itself is defense-in-depth
    # (with images on [0, 2) and exact tiling, language always covers [2, 8)).
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6),
    )
    with pytest.raises(ValueError, match="tile"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_missing_images_module():
    config = MIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2
            )
        }
    )
    with pytest.raises(ValueError, match="'images'"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_extra_modules():
    config = MIMOParallelismConfig(
        module_parallelisms={
            IMAGES_MODULE_NAME: ModuleParallelismConfig(
                tensor_model_parallel_size=2, data_parallel_size=1
            ),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2
            ),
            "audio": ModuleParallelismConfig(data_parallel_size=8),
        }
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


# ---------------------------------------------------------------------------
# Fail-fast: CP / EP / ETP / vision PP / vision DP > language DP / unknown layout.
# ---------------------------------------------------------------------------


def test_rejects_cp_gt_1_on_language():
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            context_parallel_size=2,
            data_parallel_size=3,
            rank_offset=2,
        ),
    )
    with pytest.raises(ValueError, match="CP > 1"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_cp_gt_1_on_images():
    config = _grid_config(
        ModuleParallelismConfig(
            tensor_model_parallel_size=1, data_parallel_size=1, context_parallel_size=2
        ),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2),
    )
    with pytest.raises(ValueError, match="CP > 1"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_ep_gt_1():
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            data_parallel_size=6,
            expert_model_parallel_size=2,
            rank_offset=2,
        ),
    )
    with pytest.raises(ValueError, match="EP > 1"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_etp_gt_1():
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            data_parallel_size=6,
            expert_tensor_parallel_size=2,
            rank_offset=2,
        ),
    )
    with pytest.raises(ValueError, match="ETP > 1"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_vision_pp_gt_1():
    config = _grid_config(
        ModuleParallelismConfig(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2, data_parallel_size=1
        ),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2),
    )
    with pytest.raises(ValueError, match="vision.*PP"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_vision_dp_gt_language_dp():
    # images [0,2) tp1 dp2; language [2,8) tp1 pp6 dp1 -> vision DP 2 > lang DP 1.
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=6,
            data_parallel_size=1,
            rank_offset=2,
        ),
    )
    with pytest.raises(ValueError, match="vision DP <= language DP"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_unsupported_family_guard():
    """The family check is a defense-in-depth guard.

    With the current constraint set (exact tiling, dense modules, TP powers of
    two, pairwise-divisible DPs, vision DP <= language DP) every valid layout
    is one of the 8 families, so the family check is unreachable through the
    public API - but it must reject a hypothetical non-canonical family if the
    earlier constraints ever relax.  Exercise the guard directly.
    """
    from flagscale.models.mimo.bridge.recipe.qwen35 import _family_index

    # Canonical pair -> family 1.
    assert (
        _family_index(
            ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
            ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                data_parallel_size=6,
                rank_offset=2,
            ),
        )
        == 1
    )
    # Tiling-valid but non-canonical pair (language TP1/PP2/DP3 with images
    # TP1/DP2) -> not a family.
    assert (
        _family_index(
            ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
            ModuleParallelismConfig(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                data_parallel_size=3,
                rank_offset=2,
            ),
        )
        is None
    )


def test_rejects_non_exact_tiling_gap():
    # language spans [2, 4) only: gap on ranks [4, 8).
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=2),
    )
    with pytest.raises(ValueError, match="tile"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


def test_rejects_explicit_colocated_layout():
    config = MIMOParallelismConfig(
        module_parallelisms={
            IMAGES_MODULE_NAME: ModuleParallelismConfig(
                tensor_model_parallel_size=2, data_parallel_size=4
            ),
            "language": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=4),
        },
        layout="colocated",
    )
    with pytest.raises(ValueError, match="non-colocated"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE)


# ---------------------------------------------------------------------------
# Builder + batch contract.
# ---------------------------------------------------------------------------


def test_build_from_specs_family_1():
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    assert family_index == 1
    images = config.get_parallelism(IMAGES_MODULE_NAME)
    language = config.get_parallelism("language")
    assert (images.rank_offset, images.total_ranks) == (0, 2)
    assert (language.rank_offset, language.total_ranks) == (2, 6)


def test_build_from_specs_family_8():
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=1,dp=2; language=tp=1,pp=3,dp=2,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    assert family_index == 8


def test_build_from_specs_rejects_invalid_spec():
    with pytest.raises(ValueError):
        build_qwen35_grid_config_from_args(
            "vision=tp=2,dp=1; language=tp=1,dp=6,rank_offset=2",
            QWEN35_GRID_WORLD_SIZE,
        )


def test_data_contract_ok():
    config, _ = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    per_module_dp = qwen35_grid_data_contract(
        config, micro_batch_size=6, global_batch_size=24, num_microbatches=4
    )
    assert per_module_dp == {IMAGES_MODULE_NAME: 1, "language": 6}


def test_data_contract_rejects_mbs_not_divisible_by_language_dp():
    config, _ = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    with pytest.raises(ValueError, match="not divisible by module 'language' DP"):
        qwen35_grid_data_contract(
            config, micro_batch_size=5, global_batch_size=20, num_microbatches=4
        )


def test_data_contract_rejects_microbatch_product_mismatch():
    config, _ = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    with pytest.raises(ValueError, match="batch contract"):
        qwen35_grid_data_contract(
            config, micro_batch_size=6, global_batch_size=48, num_microbatches=4
        )


# ---------------------------------------------------------------------------
# Language layer-count bound by PP (num_layers): pp <= num_layers, uneven OK.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family_index", list(range(1, 9)))
def test_num_layers_32_accepts_all_families(family_index):
    """The 32-layer 4B config validates against every supported family.

    Divisibility is no longer required: PP3/PP6 use MCore's uneven pipeline
    allocation (first stage base+remainder, last stage base).
    """
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[family_index - 1]
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_layers=32) == (
        family_index
    )


def test_num_layers_36_supports_pp3_family():
    # The PP3 layout family stays supported for divisible counts too.
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[2]  # family 3: L PP3/DP2
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_layers=36) == 3


@pytest.mark.parametrize(
    ("family_index", "num_layers"),
    ((3, 2), (4, 5), (6, 2), (8, 2)),
)
def test_num_layers_less_than_pp_rejected(family_index, num_layers):
    """pp > num_layers leaves at least one PP stage with zero layers."""
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[family_index - 1]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="at least"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_layers=num_layers)


def test_num_layers_equal_to_pp_accepted():
    # Exactly one layer per stage is the smallest valid split (PP3 -> 1/1/1).
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[2]  # family 3: L PP3
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_layers=3) == 3


def test_num_layers_validation_rejects_non_positive():
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[0]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="num_layers"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_layers=0)


def test_num_layers_optional_keeps_legacy_behavior():
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[2]  # family 3: L PP3
    config = _grid_config(images, language)
    # Without num_layers the PP3 family validates as before (structural check
    # only).
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE) == 3


def test_build_from_specs_threads_num_layers():
    # 32 layers with PP3 now validates (uneven 12/10/10 split).
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=3,dp=2,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert family_index == 3
    with pytest.raises(ValueError, match="at least"):
        build_qwen35_grid_config_from_args(
            "images=tp=2,dp=1; language=tp=1,pp=3,dp=2,rank_offset=2",
            QWEN35_GRID_WORLD_SIZE,
            num_layers=2,
        )
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert family_index == 1


# ---------------------------------------------------------------------------
# Uneven pipeline layer split (compute_qwen35_pipeline_layer_split).
# ---------------------------------------------------------------------------


def test_split_32_layers_pp3_12_10_10():
    assert compute_qwen35_pipeline_layer_split(32, 3) == [12, 10, 10]


def test_split_32_layers_pp6_7_5_5_5_5_5():
    assert compute_qwen35_pipeline_layer_split(32, 6) == [7, 5, 5, 5, 5, 5]


def test_split_even_and_pp1():
    assert compute_qwen35_pipeline_layer_split(32, 2) == [16, 16]
    assert compute_qwen35_pipeline_layer_split(32, 1) == [32]
    assert compute_qwen35_pipeline_layer_split(36, 3) == [12, 12, 12]


def test_split_pp2_with_remainder():
    assert compute_qwen35_pipeline_layer_split(33, 2) == [17, 16]


def test_split_matches_family_pp_of_every_supported_family():
    for images, language in QWEN35_GRID_SUPPORTED_FAMILIES:
        pp = language.pipeline_model_parallel_size
        split = compute_qwen35_pipeline_layer_split(32, pp)
        assert len(split) == pp
        assert sum(split) == 32
        assert min(split) >= 1
        # First stage holds base+remainder; every other stage holds base.
        assert split[0] == 32 // pp + 32 % pp
        assert all(layer_count == 32 // pp for layer_count in split[1:])


def test_split_rejects_invalid_args():
    with pytest.raises(ValueError, match="num_layers"):
        compute_qwen35_pipeline_layer_split(0, 3)
    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        compute_qwen35_pipeline_layer_split(32, 0)
    with pytest.raises(ValueError, match="at least"):
        compute_qwen35_pipeline_layer_split(2, 3)


# ---------------------------------------------------------------------------
# Per-module sequence parallelism (compute_qwen35_grid_sequence_parallel).
# The grid path conservatively marks BOTH modules SP-incapable, so requested
# SP resolves to per-module False for every family; TP is unaffected.
# ---------------------------------------------------------------------------


def test_sp_family1_baseline_vtp2_ltp1_requested_true():
    """Baseline 2+6 (V TP2/DP1 + L TP1/PP1/DP6): the vision module has TP2 but
    is NOT SP-capable (packed-seq attention/rotary on the full token
    dimension), so SP stays off; the language module is TP1, also off."""
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert family_index == 1
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


def test_sp_family5_vtp2_ltp2_requested_true():
    """Family 5 (V TP2/DP1 + L TP2/PP1/DP3): requested SP resolves to False
    for BOTH modules - the grid language forward does not shard embeddings
    and the mRoPE freqs stay full-length, so an SP-enabled TP2 qkv would
    all-gather dim 0 to 2x the sequence against full-length freqs (4096 vs
    2048).  Language TP2 itself is preserved."""
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert family_index == 5
    assert config.get_parallelism("language").tensor_model_parallel_size == 2
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


@pytest.mark.parametrize(
    ("family_index", "module_specs"),
    (
        (5, "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2"),
        (6, "images=tp=2,dp=1; language=tp=2,pp=3,dp=1,rank_offset=2"),
    ),
)
def test_sp_requested_true_family5_6_language_tp2_resolves_false(family_index, module_specs):
    """Regression: the language-TP2 grid families (5 and 6) resolve a
    requested global SP to per-module False for both modules while TP2
    itself stays structurally supported (family index and language module
    TP unchanged)."""
    config, index = build_qwen35_grid_config_from_args(
        module_specs, QWEN35_GRID_WORLD_SIZE, num_layers=32
    )
    assert index == family_index
    assert config.get_parallelism("language").tensor_model_parallel_size == 2
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


def test_sp_user_false_disables_everywhere():
    config, _ = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    assert compute_qwen35_grid_sequence_parallel(config, False) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }
    config1, _ = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    assert compute_qwen35_grid_sequence_parallel(config1, False) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


def test_sp_tp1_modules_stay_off_even_when_requested():
    """Family 7 (V TP1/DP2 + L TP1/PP1/DP6): both modules are TP1, so SP must
    stay off even when the user requested it."""
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=1,dp=2; language=tp=1,pp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert family_index == 7
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


@pytest.mark.parametrize("family_index", list(range(1, 9)))
def test_sp_requested_true_resolves_false_for_all_families(family_index):
    """Requested global SP resolves to per-module False for ALL eight
    supported families.  Both modules are conservatively SP-incapable in the
    grid path: the vision encoder's packed-seq attention/rotary operate on
    the full token dimension, and the grid language forward does not shard
    embeddings with full-length mRoPE freqs (qkv all-gather 2S vs S).  The
    TP2 language families (5/6) keep tensor parallelism structurally
    supported."""
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[family_index - 1]
    config = _grid_config(images, language)
    sp = compute_qwen35_grid_sequence_parallel(config, True)
    assert sp == {IMAGES_MODULE_NAME: False, "language": False}
    # TP2 layouts (language families 5/6, vision families 1-6) remain valid.
    if language.tensor_model_parallel_size > 1:
        assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE) == family_index


def test_sp_capable_modules_param_overrides_default():
    config, _ = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
    )
    # Default: no module is SP-capable in the grid path - both modules are
    # conservatively SP-incapable (language: full-sequence embeddings +
    # full-length mRoPE freqs; vision: packed-seq full-dim).
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }
    # Explicit capability grants follow requested && tp > 1 (escape hatch
    # for a future capable implementation).
    assert compute_qwen35_grid_sequence_parallel(
        config, True, sp_capable_modules=(IMAGES_MODULE_NAME, "language")
    ) == {
        IMAGES_MODULE_NAME: True,
        "language": True,
    }


# ---------------------------------------------------------------------------
# MTP fail-fast (num_mtp_layers).
# ---------------------------------------------------------------------------


def test_num_mtp_layers_gt_0_rejected():
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[0]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="MTP"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_mtp_layers=1)


def test_num_mtp_layers_0_or_none_accepted():
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[0]
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_mtp_layers=0) == 1
    assert validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_mtp_layers=None) == 1


def test_num_mtp_layers_negative_rejected():
    images, language = QWEN35_GRID_SUPPORTED_FAMILIES[0]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="num_mtp_layers"):
        validate_qwen35_grid_config(config, QWEN35_GRID_WORLD_SIZE, num_mtp_layers=-1)


def test_build_from_specs_threads_num_mtp_layers():
    with pytest.raises(ValueError, match="MTP"):
        build_qwen35_grid_config_from_args(
            "images=tp=2,dp=1; language=tp=1,dp=6,rank_offset=2",
            QWEN35_GRID_WORLD_SIZE,
            num_mtp_layers=1,
        )
    config, family_index = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,dp=6,rank_offset=2",
        QWEN35_GRID_WORLD_SIZE,
        num_mtp_layers=0,
    )
    assert family_index == 1
