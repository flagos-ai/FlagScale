# Copyright (c) 2026, BAAI. All rights reserved.

"""Unit tests for flagscale.models.mimo.bridge.recipe.qwen35.

Pure-CPU tests for the Qwen3.5 non-colocated grid contract: the predicate
validator (this stage's capability boundary), the builder, and the batch
contract.  The module under test is dependency-free (stdlib +
``bridge.parallelism`` only), so no torch/Megatron runtime is required.

The 2+6 layouts in ``_FAMILY_PRESETS`` are *test presets* — the layouts the
GPU smoke harness exercises — NOT a validation whitelist: the validator
accepts any layout inside the stage's capability boundary (see the module
docstring).
"""

import pytest

try:
    from flagscale.models.mimo.bridge.parallelism import (
        MIMOParallelismConfig,
        ModuleParallelismConfig,
    )
    from flagscale.models.mimo.bridge.recipe.qwen35 import (
        IMAGES_MODULE_NAME,
        build_qwen35_grid_config_from_args,
        compute_qwen35_grid_sequence_parallel,
        compute_qwen35_pipeline_layer_split,
        describe_qwen35_grid_modules,
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
        build_qwen35_grid_config_from_args,
        compute_qwen35_grid_sequence_parallel,
        compute_qwen35_pipeline_layer_split,
        describe_qwen35_grid_modules,
        qwen35_grid_data_contract,
        validate_qwen35_grid_config,
    ) = (
        _grid_mod.IMAGES_MODULE_NAME,
        _grid_mod.build_qwen35_grid_config_from_args,
        _grid_mod.compute_qwen35_grid_sequence_parallel,
        _grid_mod.compute_qwen35_pipeline_layer_split,
        _grid_mod.describe_qwen35_grid_modules,
        _grid_mod.qwen35_grid_data_contract,
        _grid_mod.validate_qwen35_grid_config,
    )


#: World size of the 2+6 test presets (the single-node smoke configuration).
GRID_WORLD_SIZE = 8

#: Test presets: the 2+6 (world 8, images ranks [0, 2)) layouts the GPU smoke
#: harness exercises, as ``(images_config, language_config)`` pairs.  These
#: are inputs to the tests, not library validation data.
_FAMILY_PRESETS: list[tuple[ModuleParallelismConfig, ModuleParallelismConfig]] = [
    # 1. V TP2 DP1 + L TP1 PP1 DP6
    (
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=6,
            rank_offset=2,
        ),
    ),
    # 2. V TP2 DP1 + L TP1 PP2 DP3
    (
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=3,
            rank_offset=2,
        ),
    ),
    # 3. V TP2 DP1 + L TP1 PP3 DP2
    (
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=3,
            data_parallel_size=2,
            rank_offset=2,
        ),
    ),
    # 4. V TP2 DP1 + L TP1 PP6 DP1
    (
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=6,
            data_parallel_size=1,
            rank_offset=2,
        ),
    ),
    # 5. V TP2 DP1 + L TP2 PP1 DP3
    (
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            data_parallel_size=3,
            rank_offset=2,
        ),
    ),
    # 6. V TP2 DP1 + L TP2 PP3 DP1
    (
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=3,
            data_parallel_size=1,
            rank_offset=2,
        ),
    ),
    # 7. V TP1 DP2 + L TP1 PP1 DP6
    (
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=6,
            rank_offset=2,
        ),
    ),
    # 8. V TP1 DP2 + L TP1 PP3 DP2
    (
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
        ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=3,
            data_parallel_size=2,
            rank_offset=2,
        ),
    ),
]

#: Spec string of a legal layout that the previous whitelist rejected: images
#: spans [0, 4) (TP2/DP2), language spans [4, 8) (TP1/PP1/DP4).
_SHIFTED_LAYOUT_SPEC = "images=tp=2,dp=2; language=tp=1,pp=1,dp=4,rank_offset=4"


def _grid_config(images, language, layout="non_colocated"):
    return MIMOParallelismConfig(
        module_parallelisms={
            IMAGES_MODULE_NAME: images,
            "language": language,
        },
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Test presets: the 2+6 layouts validate (regression coverage of the GPU
# smoke matrix) and match the mission matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_index", list(range(len(_FAMILY_PRESETS))))
def test_all_presets_validate(preset_index):
    images, language = _FAMILY_PRESETS[preset_index]
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE) is None


def test_presets_match_mission_matrix():
    """The preset list matches the mission's 8 smoke layouts exactly."""
    signatures = []
    for images, language in _FAMILY_PRESETS:
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
# Predicate validation: layouts outside the old whitelist but inside the
# capability boundary are accepted.
# ---------------------------------------------------------------------------


def test_accepts_shifted_rank_spans():
    """images [0, 4) + language [4, 8) satisfies every predicate; the old
    fixed-span/whitelist checks rejected it."""
    config = build_qwen35_grid_config_from_args(_SHIFTED_LAYOUT_SPEC, GRID_WORLD_SIZE)
    images = config.get_parallelism(IMAGES_MODULE_NAME)
    language = config.get_parallelism("language")
    assert (images.rank_offset, images.total_ranks) == (0, 4)
    assert (language.rank_offset, language.total_ranks) == (4, 4)


def test_accepts_world_size_other_than_8():
    """World size is a finalize input, not a pinned constant: a 16-rank
    layout that tiles exactly validates."""
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=14,rank_offset=2",
        16,
    )
    assert config.total_world_size == 16


def test_rejects_world_size_mismatch():
    """A spec tiling [0, 8) cannot run on a 16-rank world (gap [8, 16))."""
    with pytest.raises(ValueError, match="tile"):
        build_qwen35_grid_config_from_args(
            "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
            16,
        )


def test_rejects_non_power_of_two_tp():
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=3, data_parallel_size=1),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=5, rank_offset=3),
    )
    with pytest.raises(ValueError, match="power of 2"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


def test_rejects_non_pairwise_divisible_dp():
    # images [0, 2) dp2; language [2, 5) dp3: neither DP divides the other.
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=3, rank_offset=2),
    )
    with pytest.raises(ValueError, match="pairwise divisible"):
        validate_qwen35_grid_config(config, 5)


def test_rejects_tiling_overlap():
    # language starts at rank 1 while images already covers [0, 2).
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=7, rank_offset=1),
    )
    with pytest.raises(ValueError, match="tile"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


def test_rejects_non_exact_tiling_gap():
    # language spans [2, 4) only: gap on ranks [4, 8).
    config = _grid_config(
        ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=1),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=2),
    )
    with pytest.raises(ValueError, match="tile"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


def test_rejects_missing_images_module():
    config = MIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2
            )
        }
    )
    with pytest.raises(ValueError, match="'images'"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


def test_rejects_missing_language_module():
    config = MIMOParallelismConfig(
        module_parallelisms={
            IMAGES_MODULE_NAME: ModuleParallelismConfig(
                tensor_model_parallel_size=2, data_parallel_size=1
            ),
            "audio": ModuleParallelismConfig(data_parallel_size=6, rank_offset=2),
        }
    )
    with pytest.raises(ValueError, match="'language'"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


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
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


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
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


# ---------------------------------------------------------------------------
# Fail-fast: CP / EP / ETP / vision PP / vision DP > language DP.
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
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


def test_rejects_cp_gt_1_on_images():
    config = _grid_config(
        ModuleParallelismConfig(
            tensor_model_parallel_size=1, data_parallel_size=1, context_parallel_size=2
        ),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2),
    )
    with pytest.raises(ValueError, match="CP > 1"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


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
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


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
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


def test_rejects_vision_pp_gt_1():
    config = _grid_config(
        ModuleParallelismConfig(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2, data_parallel_size=1
        ),
        ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=6, rank_offset=2),
    )
    with pytest.raises(ValueError, match="vision.*PP"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


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
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE)


# ---------------------------------------------------------------------------
# Builder + module summary + batch contract.
# ---------------------------------------------------------------------------


def test_build_from_specs_returns_validated_config():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    images = config.get_parallelism(IMAGES_MODULE_NAME)
    language = config.get_parallelism("language")
    assert (images.rank_offset, images.total_ranks) == (0, 2)
    assert (language.rank_offset, language.total_ranks) == (2, 6)


def test_build_from_specs_rejects_invalid_spec():
    with pytest.raises(ValueError):
        build_qwen35_grid_config_from_args(
            "vision=tp=2,dp=1; language=tp=1,dp=6,rank_offset=2",
            GRID_WORLD_SIZE,
        )


def test_describe_qwen35_grid_modules():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    assert describe_qwen35_grid_modules(config) == (
        "images tp=2 pp=1 dp=1 ranks [0, 2); language tp=1 pp=1 dp=6 ranks [2, 8)"
    )


def test_data_contract_ok():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    per_module_dp = qwen35_grid_data_contract(
        config, micro_batch_size=6, global_batch_size=24, num_microbatches=4
    )
    assert per_module_dp == {IMAGES_MODULE_NAME: 1, "language": 6}


def test_data_contract_rejects_mbs_not_divisible_by_language_dp():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    with pytest.raises(ValueError, match="not divisible by module 'language' DP"):
        qwen35_grid_data_contract(
            config, micro_batch_size=5, global_batch_size=20, num_microbatches=4
        )


def test_data_contract_rejects_microbatch_product_mismatch():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    with pytest.raises(ValueError, match="batch contract"):
        qwen35_grid_data_contract(
            config, micro_batch_size=6, global_batch_size=48, num_microbatches=4
        )


# ---------------------------------------------------------------------------
# Language layer-count bound by PP (num_layers): pp <= num_layers, uneven OK.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_index", list(range(len(_FAMILY_PRESETS))))
def test_num_layers_32_accepts_all_presets(preset_index):
    """The 32-layer 4B config validates against every preset layout.

    Divisibility is not required: PP3/PP6 use MCore's uneven pipeline
    allocation (first stage base+remainder, last stage base).
    """
    images, language = _FAMILY_PRESETS[preset_index]
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_layers=32) is None


def test_num_layers_36_supports_pp3_layout():
    # The PP3 preset layout stays supported for divisible counts too.
    images, language = _FAMILY_PRESETS[2]  # preset 3: L PP3/DP2
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_layers=36) is None


@pytest.mark.parametrize(
    ("preset_index", "num_layers"),
    ((2, 2), (3, 5), (5, 2), (7, 2)),
)
def test_num_layers_less_than_pp_rejected(preset_index, num_layers):
    """pp > num_layers leaves at least one PP stage with zero layers."""
    images, language = _FAMILY_PRESETS[preset_index]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="at least"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_layers=num_layers)


def test_num_layers_equal_to_pp_accepted():
    # Exactly one layer per stage is the smallest valid split (PP3 -> 1/1/1).
    images, language = _FAMILY_PRESETS[2]  # preset 3: L PP3
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_layers=3) is None


def test_num_layers_validation_rejects_non_positive():
    images, language = _FAMILY_PRESETS[0]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="num_layers"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_layers=0)


def test_num_layers_optional_keeps_structural_only_behavior():
    images, language = _FAMILY_PRESETS[2]  # preset 3: L PP3
    config = _grid_config(images, language)
    # Without num_layers the PP3 layout validates as before (structural check
    # only).
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE) is None


def test_build_from_specs_threads_num_layers():
    # 32 layers with PP3 validates (uneven 12/10/10 split).
    build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=3,dp=2,rank_offset=2",
        GRID_WORLD_SIZE,
        num_layers=32,
    )
    with pytest.raises(ValueError, match="at least"):
        build_qwen35_grid_config_from_args(
            "images=tp=2,dp=1; language=tp=1,pp=3,dp=2,rank_offset=2",
            GRID_WORLD_SIZE,
            num_layers=2,
        )
    build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
        num_layers=32,
    )


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


def test_split_matches_language_pp_of_every_preset():
    for images, language in _FAMILY_PRESETS:
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
# SP resolves to per-module False for every layout; TP is unaffected.
# ---------------------------------------------------------------------------


def test_sp_preset1_baseline_vtp2_ltp1_requested_true():
    """Baseline 2+6 (V TP2/DP1 + L TP1/PP1/DP6): the vision module has TP2 but
    is NOT SP-capable (packed-seq attention/rotary on the full token
    dimension), so SP stays off; the language module is TP1, also off."""
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


def test_sp_preset5_vtp2_ltp2_requested_true():
    """Preset 5 (V TP2/DP1 + L TP2/PP1/DP3): requested SP resolves to False
    for BOTH modules - the grid language forward does not shard embeddings
    and the mRoPE freqs stay full-length, so an SP-enabled TP2 qkv would
    all-gather dim 0 to 2x the sequence against full-length freqs (4096 vs
    2048).  Language TP2 itself is preserved."""
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert config.get_parallelism("language").tensor_model_parallel_size == 2
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


@pytest.mark.parametrize(
    "module_specs",
    (
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        "images=tp=2,dp=1; language=tp=2,pp=3,dp=1,rank_offset=2",
    ),
)
def test_sp_requested_true_language_tp2_resolves_false(module_specs):
    """Regression: language-TP2 grid layouts resolve a requested global SP to
    per-module False for both modules while TP2 itself stays structurally
    supported (language module TP unchanged)."""
    config = build_qwen35_grid_config_from_args(module_specs, GRID_WORLD_SIZE, num_layers=32)
    assert config.get_parallelism("language").tensor_model_parallel_size == 2
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


def test_sp_user_false_disables_everywhere():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    assert compute_qwen35_grid_sequence_parallel(config, False) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }
    config1 = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
    )
    assert compute_qwen35_grid_sequence_parallel(config1, False) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


def test_sp_tp1_modules_stay_off_even_when_requested():
    """Preset 7 (V TP1/DP2 + L TP1/PP1/DP6): both modules are TP1, so SP must
    stay off even when the user requested it."""
    config = build_qwen35_grid_config_from_args(
        "images=tp=1,dp=2; language=tp=1,pp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
        num_layers=32,
    )
    assert compute_qwen35_grid_sequence_parallel(config, True) == {
        IMAGES_MODULE_NAME: False,
        "language": False,
    }


@pytest.mark.parametrize("preset_index", list(range(len(_FAMILY_PRESETS))))
def test_sp_requested_true_resolves_false_for_all_presets(preset_index):
    """Requested global SP resolves to per-module False for all preset
    layouts.  Both modules are conservatively SP-incapable in the grid path:
    the vision encoder's packed-seq attention/rotary operate on the full
    token dimension, and the grid language forward does not shard embeddings
    with full-length mRoPE freqs (qkv all-gather 2S vs S).  TP2 layouts keep
    tensor parallelism structurally supported."""
    images, language = _FAMILY_PRESETS[preset_index]
    config = _grid_config(images, language)
    sp = compute_qwen35_grid_sequence_parallel(config, True)
    assert sp == {IMAGES_MODULE_NAME: False, "language": False}
    if language.tensor_model_parallel_size > 1:
        assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE) is None


def test_sp_capable_modules_param_overrides_default():
    config = build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=2,pp=1,dp=3,rank_offset=2",
        GRID_WORLD_SIZE,
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
    images, language = _FAMILY_PRESETS[0]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="MTP"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_mtp_layers=1)


def test_num_mtp_layers_0_or_none_accepted():
    images, language = _FAMILY_PRESETS[0]
    config = _grid_config(images, language)
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_mtp_layers=0) is None
    assert validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_mtp_layers=None) is None


def test_num_mtp_layers_negative_rejected():
    images, language = _FAMILY_PRESETS[0]
    config = _grid_config(images, language)
    with pytest.raises(ValueError, match="num_mtp_layers"):
        validate_qwen35_grid_config(config, GRID_WORLD_SIZE, num_mtp_layers=-1)


def test_build_from_specs_threads_num_mtp_layers():
    with pytest.raises(ValueError, match="MTP"):
        build_qwen35_grid_config_from_args(
            "images=tp=2,dp=1; language=tp=1,dp=6,rank_offset=2",
            GRID_WORLD_SIZE,
            num_mtp_layers=1,
        )
    build_qwen35_grid_config_from_args(
        "images=tp=2,dp=1; language=tp=1,dp=6,rank_offset=2",
        GRID_WORLD_SIZE,
        num_mtp_layers=0,
    )
