# Copyright (c) 2026, BAAI. All rights reserved.

"""Unit tests for flagscale.models.mimo.bridge.parallelism.

Pure-CPU tests: the module under test is dependency-free (stdlib only), so no
torch/Megatron runtime is required.
"""

import dataclasses
from types import MappingProxyType

import pytest

try:
    # Preferred path: the package import chain (used in the full FlagScale env).
    from flagscale.models.mimo.bridge.parallelism import (
        LANGUAGE_MODULE_NAME,
        MIMOLayout,
        MIMOParallelismConfig,
        ModuleParallelismConfig,
        classify_layout,
        parse_module_parallelism,
        parse_module_parallelisms,
    )
except ImportError:  # pragma: no cover - exercised only without the full stack
    # Fallback for environments without torch/Megatron (e.g. plain CI): load
    # the pure-stdlib module directly by path.
    import importlib.util
    import sys
    from pathlib import Path

    _module_path = (
        Path(__file__).resolve().parents[3]
        / "flagscale"
        / "models"
        / "mimo"
        / "bridge"
        / "parallelism.py"
    )
    _spec = importlib.util.spec_from_file_location("mimo_parallelism_config", _module_path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module  # dataclass processing requires sys.modules registration
    _spec.loader.exec_module(_module)
    (
        LANGUAGE_MODULE_NAME,
        MIMOLayout,
        MIMOParallelismConfig,
        ModuleParallelismConfig,
        classify_layout,
        parse_module_parallelism,
        parse_module_parallelisms,
    ) = (
        _module.LANGUAGE_MODULE_NAME,
        _module.MIMOLayout,
        _module.MIMOParallelismConfig,
        _module.ModuleParallelismConfig,
        _module.classify_layout,
        _module.parse_module_parallelism,
        _module.parse_module_parallelisms,
    )


# ---------------------------------------------------------------------------
# ModuleParallelismConfig: construction, sizing, immutability
# ---------------------------------------------------------------------------


def test_module_config_defaults():
    config = ModuleParallelismConfig()
    assert config.tensor_model_parallel_size == 1
    assert config.pipeline_model_parallel_size == 1
    assert config.data_parallel_size == 1
    assert config.context_parallel_size == 1
    assert config.expert_model_parallel_size == 1
    assert config.expert_tensor_parallel_size == 1
    assert config.rank_offset == 0
    assert config.dense_model_parallel_size == 1
    assert config.total_model_parallel_size == 1
    assert config.total_ranks == 1
    assert config.rank_span == 1
    assert config.rank_end == 1
    assert config.rank_range == (0, 1)


def test_module_config_rank_and_world_sizing():
    config = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        pipeline_model_parallel_size=2,
        data_parallel_size=4,
        rank_offset=16,
    )
    assert config.dense_model_parallel_size == 8
    assert config.total_model_parallel_size == 8
    assert config.total_ranks == 32
    assert config.rank_span == 32
    assert config.rank_end == 48
    assert config.rank_range == (16, 48)


def test_module_config_expert_rank_algebra_excludes_ep_from_span():
    config = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        data_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=2,
    )
    # Total ranks ignore ep/etp (they subdivide the dense token domain).
    assert config.total_ranks == 8
    assert config.expert_model_parallel_span == 4
    assert config.expert_data_parallel_size == 2


def test_module_config_is_immutable():
    config = ModuleParallelismConfig(tensor_model_parallel_size=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.tensor_model_parallel_size = 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.rank_offset = 8


@pytest.mark.parametrize(
    "field_name",
    [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "data_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    ],
)
def test_module_config_rejects_non_positive_sizes(field_name):
    with pytest.raises(ValueError, match=f"{field_name} must be a positive integer"):
        ModuleParallelismConfig(**{field_name: 0})
    with pytest.raises(ValueError, match=f"{field_name} must be a positive integer"):
        ModuleParallelismConfig(**{field_name: -2})


def test_module_config_rejects_non_integer_and_bool_sizes():
    with pytest.raises(ValueError, match="must be a positive integer"):
        ModuleParallelismConfig(tensor_model_parallel_size=1.5)
    with pytest.raises(ValueError, match="must be a positive integer"):
        ModuleParallelismConfig(data_parallel_size=True)


def test_module_config_rejects_negative_rank_offset():
    with pytest.raises(ValueError, match="rank_offset must be a non-negative integer"):
        ModuleParallelismConfig(rank_offset=-1)
    with pytest.raises(ValueError, match="rank_offset must be a non-negative integer"):
        ModuleParallelismConfig(rank_offset=1.5)


def test_module_config_rejects_bad_expert_factorization():
    with pytest.raises(ValueError, match="TP \\* CP \\* DP must be divisible"):
        ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=1,
            expert_tensor_parallel_size=3,
        )


# ---------------------------------------------------------------------------
# MIMOLayout and classify_layout
# ---------------------------------------------------------------------------


def test_layout_enum_values():
    assert MIMOLayout.COLOCATED.value == "colocated"
    assert MIMOLayout.NON_COLOCATED.value == "non_colocated"
    assert MIMOLayout.AUTO.value == "auto"


def test_classify_layout_colocated_when_all_modules_span_full_world():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=4, data_parallel_size=1),
    }
    assert classify_layout(module_parallelisms, world_size=4) is MIMOLayout.COLOCATED


def test_classify_layout_non_colocated_for_disjoint_tiling():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
        ),
    }
    assert classify_layout(module_parallelisms, world_size=12) is MIMOLayout.NON_COLOCATED


def test_classify_layout_rejects_empty_module_set():
    with pytest.raises(ValueError, match="at least one module"):
        classify_layout({}, world_size=8)


# ---------------------------------------------------------------------------
# MIMOParallelismConfig: construction and immutability
# ---------------------------------------------------------------------------


def test_container_rejects_empty_module_set():
    with pytest.raises(ValueError, match="at least one module"):
        MIMOParallelismConfig(module_parallelisms={})


def test_container_rejects_non_config_values():
    with pytest.raises(TypeError, match="must be a ModuleParallelismConfig"):
        MIMOParallelismConfig(module_parallelisms={"language": {"tp": 4}})


def test_container_is_immutable():
    config = MIMOParallelismConfig(module_parallelisms={"language": ModuleParallelismConfig()})
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.layout = MIMOLayout.COLOCATED
    with pytest.raises(TypeError):
        config.module_parallelisms["language"] = ModuleParallelismConfig(
            tensor_model_parallel_size=2
        )
    assert isinstance(config.module_parallelisms, MappingProxyType)


def test_container_accepts_str_layout_and_coerces():
    config = MIMOParallelismConfig(
        module_parallelisms={"language": ModuleParallelismConfig()},
        layout="non-colocated",
    )
    assert config.layout is MIMOLayout.NON_COLOCATED
    assert MIMOParallelismConfig._coerce_layout("colocated") is MIMOLayout.COLOCATED
    assert MIMOParallelismConfig._coerce_layout("AUTO") is MIMOLayout.AUTO
    assert MIMOParallelismConfig._coerce_layout("non_colocated") is MIMOLayout.NON_COLOCATED
    with pytest.raises(ValueError, match="unknown MIMO layout"):
        MIMOParallelismConfig._coerce_layout("bogus")
    with pytest.raises(TypeError, match="layout must be a MIMOLayout or str"):
        MIMOParallelismConfig._coerce_layout(123)


# ---------------------------------------------------------------------------
# Non-colocated layout: exact tiling (no gaps, no overlaps)
# ---------------------------------------------------------------------------


def _non_colocated_modules():
    return {
        "vision": ModuleParallelismConfig(data_parallel_size=4),  # ranks [0, 4)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
        ),  # ranks [4, 12)
    }


def test_non_colocated_valid_tiling_finalizes():
    config = MIMOParallelismConfig(module_parallelisms=_non_colocated_modules())
    config.finalize(world_size=12)
    assert config.total_world_size == 12
    assert config.module_names == ["vision", "language"]
    assert config.get_parallelism("language").rank_range == (4, 12)
    assert config.rank_ranges == [(0, 4, "vision"), (4, 12, "language")]


def test_non_colocated_rejects_overlapping_rank_ranges():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),  # [0, 4)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=1, rank_offset=2
        ),  # [2, 6) overlaps
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="no overlaps"):
        config.finalize(world_size=6)


def test_non_colocated_rejects_gaps_between_rank_ranges():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=2),  # [0, 2)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=1, rank_offset=3
        ),  # [3, 7): gap at rank 2
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="no gaps"):
        config.finalize(world_size=7)


def test_non_colocated_rejects_partial_world_coverage():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),  # [0, 4)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=1, rank_offset=4
        ),  # [4, 8)
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="covered \\[0, 8\\), but world_size is 12"):
        config.finalize(world_size=12)


def test_non_colocated_tiling_ignores_input_dict_order():
    # language listed first: rank_ranges must still be sorted by offset.
    module_parallelisms = {
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
        ),
        "vision": ModuleParallelismConfig(data_parallel_size=4),
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    config.finalize(world_size=12)
    assert config.rank_ranges == [(0, 4, "vision"), (4, 12, "language")]


def test_non_colocated_requires_language_module():
    config = MIMOParallelismConfig(
        module_parallelisms={"vision": ModuleParallelismConfig(data_parallel_size=8)}
    )
    with pytest.raises(ValueError, match="must be in module_parallelisms"):
        config.finalize(world_size=8)


# ---------------------------------------------------------------------------
# Colocated layout: full-world validation
# ---------------------------------------------------------------------------


def _colocated_modules():
    return {
        "vision": ModuleParallelismConfig(data_parallel_size=4),  # full world 4
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=1
        ),  # full world 4
    }


def test_colocated_valid_full_world_finalizes():
    config = MIMOParallelismConfig(module_parallelisms=_colocated_modules())
    config.finalize(world_size=4)
    assert config.total_world_size == 4


def test_colocated_rejects_module_not_spanning_full_world():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),  # full world 4
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2, data_parallel_size=1
        ),  # only 2 ranks
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms, layout="colocated")
    with pytest.raises(ValueError, match="spans 2 ranks, world_size is 4"):
        config.finalize(world_size=4)


def test_colocated_rejects_nonzero_rank_offset():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=1, rank_offset=1
        ),
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms, layout="colocated")
    with pytest.raises(ValueError, match="rank_offset=1, must be 0"):
        config.finalize(world_size=4)


# ---------------------------------------------------------------------------
# AUTO layout resolution
# ---------------------------------------------------------------------------


def test_auto_layout_resolves_colocated():
    config = MIMOParallelismConfig(module_parallelisms=_colocated_modules(), layout=MIMOLayout.AUTO)
    config.finalize(world_size=4)  # classifies as COLOCATED and validates


def test_auto_layout_resolves_non_colocated():
    config = MIMOParallelismConfig(module_parallelisms=_non_colocated_modules())
    config.finalize(world_size=12)  # classifies as NON_COLOCATED and validates


def test_explicit_layout_contradicting_auto_is_rejected():
    # Same configs classify as COLOCATED, but an explicit NON_COLOCATED request
    # must fail tiling validation (the two full-world ranges overlap).
    config = MIMOParallelismConfig(module_parallelisms=_colocated_modules(), layout="non_colocated")
    with pytest.raises(ValueError, match="no overlaps"):
        config.finalize(world_size=4)

    config = MIMOParallelismConfig(module_parallelisms=_non_colocated_modules(), layout="colocated")
    with pytest.raises(ValueError, match="spans 4 ranks"):
        config.finalize(world_size=12)


def test_finalize_rejects_invalid_world_size():
    config = MIMOParallelismConfig(module_parallelisms=_colocated_modules())
    for bad_world in (0, -1, True, 4.0):
        with pytest.raises(ValueError, match="world_size must be a positive integer"):
            config.finalize(world_size=bad_world)


# ---------------------------------------------------------------------------
# Cross-module constraints: TP power of two, DP pairwise divisibility
# ---------------------------------------------------------------------------


def test_rejects_non_power_of_two_tp():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=3),  # [0, 3)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=3, data_parallel_size=1, rank_offset=3
        ),  # [3, 6)
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="TP=3, but TP size must be a power of 2"):
        config.finalize(world_size=6)


def test_rejects_non_pairwise_divisible_dp():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=3),  # DP=3
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=3
        ),  # DP=2
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="DP sizes must be pairwise divisible"):
        config.finalize(world_size=11)


def test_dp_pairwise_divisible_variants_are_accepted():
    # dp1 == dp2
    config = MIMOParallelismConfig(
        module_parallelisms={
            "vision": ModuleParallelismConfig(data_parallel_size=2),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=2
            ),
        }
    )
    config.finalize(world_size=10)
    # dp1 multiple of dp2
    config = MIMOParallelismConfig(
        module_parallelisms={
            "vision": ModuleParallelismConfig(data_parallel_size=4),
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
            ),
        }
    )
    config.finalize(world_size=12)


# ---------------------------------------------------------------------------
# Encoder modules must remain dense (EP == ETP == 1) - MegatronMIMO MoE
# ---------------------------------------------------------------------------


def test_rejects_expert_parallelism_on_encoder_module():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            data_parallel_size=4, expert_model_parallel_size=2, expert_tensor_parallel_size=2
        ),  # [0, 4)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
        ),  # [4, 12)
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="must remain dense"):
        config.finalize(world_size=12)


def test_rejects_expert_tensor_parallelism_alone_on_encoder_module():
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            data_parallel_size=4, expert_tensor_parallel_size=2
        ),  # EP=1 but ETP=2
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=4
        ),
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="must remain dense"):
        config.finalize(world_size=12)


def test_language_module_may_use_expert_parallelism():
    # The dense check is language-exempt: the mcore MIMO MoE machinery only
    # supports expert parallelism on the language module.
    module_parallelisms = {
        "vision": ModuleParallelismConfig(data_parallel_size=4),  # [0, 4)
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=4,
            data_parallel_size=2,
            rank_offset=4,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=2,
        ),  # [4, 12)
    }
    config = MIMOParallelismConfig(module_parallelisms=module_parallelisms)
    config.finalize(world_size=12)


# ---------------------------------------------------------------------------
# Parser: repeatable spec strings
# ---------------------------------------------------------------------------


def test_parse_single_module_spec():
    name, config = parse_module_parallelism("language=tp=4,pp=2,dp=2,rank_offset=0")
    assert name == "language"
    assert config.tensor_model_parallel_size == 4
    assert config.pipeline_model_parallel_size == 2
    assert config.data_parallel_size == 2
    assert config.rank_offset == 0
    assert config.context_parallel_size == 1
    assert config.expert_model_parallel_size == 1
    assert config.expert_tensor_parallel_size == 1
    # dense = tp*cp*pp = 8; total = dense*dp = 16.
    assert config.dense_model_parallel_size == 8
    assert config.total_ranks == 16


def test_parse_spec_accepts_full_field_names():
    name, config = parse_module_parallelism(
        "vision=tensor_model_parallel_size=2,context_parallel_size=2,data_parallel_size=4"
    )
    assert name == "vision"
    assert config.tensor_model_parallel_size == 2
    assert config.context_parallel_size == 2
    assert config.data_parallel_size == 4


def test_parse_spec_accepts_whitespace():
    name, config = parse_module_parallelism("  language = tp=4, pp=2 , dp=2 , rank_offset=0  ")
    assert name == "language"
    assert config.tensor_model_parallel_size == 4
    assert config.pipeline_model_parallel_size == 2
    assert config.data_parallel_size == 2


def test_parse_repeatable_semicolon_separated_specs():
    specs = "vision=tp=1,dp=4; language=tp=4,dp=2,rank_offset=4"
    module_parallelisms = parse_module_parallelisms(specs)
    assert set(module_parallelisms) == {"vision", "language"}
    assert module_parallelisms["vision"].rank_range == (0, 4)
    assert module_parallelisms["language"].rank_range == (4, 12)


def test_parse_accepts_iterable_of_specs():
    module_parallelisms = parse_module_parallelisms(
        ["vision=tp=1,dp=4", "language=tp=4,dp=2,rank_offset=4"]
    )
    assert set(module_parallelisms) == {"vision", "language"}
    assert module_parallelisms["language"].rank_offset == 4


def test_parse_roundtrip_into_container_and_finalize():
    config = MIMOParallelismConfig.from_specs(
        "vision=tp=1,dp=4;language=tp=4,dp=2,rank_offset=4",
        layout="non_colocated",
    )
    config.finalize(world_size=12)
    assert config.layout is MIMOLayout.NON_COLOCATED
    assert config.total_world_size == 12


def test_parse_rejects_missing_module_name():
    with pytest.raises(ValueError, match="invalid module spec"):
        parse_module_parallelism("language=")
    with pytest.raises(ValueError, match="invalid module name"):
        parse_module_parallelism("=tp=4,pp=2")
    with pytest.raises(ValueError, match="invalid module name"):
        parse_module_parallelism("language 1=tp=4,pp=2")


def test_parse_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown parallelism key 'foo'"):
        parse_module_parallelism("language=foo=4")


def test_parse_rejects_malformed_pair():
    with pytest.raises(ValueError, match="invalid parallelism pair"):
        parse_module_parallelism("language=tp,pp=2")


def test_parse_rejects_non_integer_value():
    with pytest.raises(ValueError, match="must be an integer"):
        parse_module_parallelism("language=tp=four")


def test_parse_rejects_duplicate_key():
    with pytest.raises(ValueError, match="duplicate parallelism key 'dp'"):
        parse_module_parallelism("language=dp=2,dp=4")


def test_parse_rejects_duplicate_module_name():
    with pytest.raises(ValueError, match="duplicate module name 'language'"):
        parse_module_parallelisms("language=tp=4,dp=1;language=tp=2,dp=2")


def test_parse_negative_size_delegates_to_config_validation():
    with pytest.raises(ValueError, match="tensor_model_parallel_size must be a positive integer"):
        parse_module_parallelism("language=tp=-2")
