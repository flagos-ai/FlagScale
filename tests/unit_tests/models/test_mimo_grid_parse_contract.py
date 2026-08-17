# Copyright (c) 2026, BAAI. All rights reserved.

"""CPU unit tests for apply_grid_parse_time_contract (bridge.training).

The parse-time grid contract fail-fasts on non-default legacy global
parallel sizes (grid module layouts come from ``--mimo-module-specs``
exclusively) and preserves the user's sequence-parallel intent in
``args.mimo_sequence_parallel`` before Megatron validation rewrites both
(PP clamp in validate_yaml; SP force-off under TP == 1 in both validators).

Run (inside the container, from the FlagScale repo root):

    source /root/miniconda3/bin/activate flagscale
    PYTHONPATH=/workspace/multimodal/Megatron-LM-FL python -m pytest \
        tests/unit_tests/models/test_mimo_grid_parse_contract.py
"""

import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_MEGATRON_REPO = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "Megatron-LM-FL"))
if os.path.isdir(_MEGATRON_REPO) and _MEGATRON_REPO not in sys.path:
    sys.path.insert(0, _MEGATRON_REPO)

import pytest

from flagscale.models.mimo.bridge.training import apply_grid_parse_time_contract


def _args(**overrides):
    values = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
        virtual_pipeline_model_parallel_size=None,
        dualpipev_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_defaults_pass_and_preserve_sp_false():
    args = _args()
    apply_grid_parse_time_contract(args)
    assert args.mimo_sequence_parallel is False


def test_sp_intent_preserved_when_requested():
    args = _args(sequence_parallel=True)
    apply_grid_parse_time_contract(args)
    # TP is 1, so Megatron validation would force sequence_parallel off; the
    # user's intent survives in the grid-specific arg.
    assert args.mimo_sequence_parallel is True
    assert args.sequence_parallel is True  # untouched here; validation drops it later


@pytest.mark.parametrize(
    "name",
    (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
    ),
)
def test_rejects_non_default_parallel_size(name):
    args = _args(**{name: 2})
    with pytest.raises(ValueError, match="mimo-module-specs"):
        apply_grid_parse_time_contract(args)
    with pytest.raises(ValueError, match=name):
        apply_grid_parse_time_contract(args)


@pytest.mark.parametrize(
    "name",
    ("virtual_pipeline_model_parallel_size",),
)
def test_rejects_pipeline_variants(name):
    args = _args(**{name: 2})
    with pytest.raises(ValueError, match=name):
        apply_grid_parse_time_contract(args)


def test_rejects_use_dualpipev():
    # dualpipev_pipeline_model_parallel_size is not an argparse attribute (it
    # is derived from the switch in post_validate_args); the contract probes
    # the real user-facing --use-dualpipev switch instead.
    args = _args(use_dualpipev=True)
    with pytest.raises(ValueError, match="use_dualpipev"):
        apply_grid_parse_time_contract(args)


def test_rejection_lists_all_offenders():
    args = _args(tensor_model_parallel_size=2, context_parallel_size=2)
    with pytest.raises(ValueError) as exc_info:
        apply_grid_parse_time_contract(args)
    message = str(exc_info.value)
    assert "tensor_model_parallel_size" in message
    assert "context_parallel_size" in message


def test_data_parallel_size_exempt():
    # DP is derived from the world size during validation, never asserted.
    args = _args(data_parallel_size=8)
    apply_grid_parse_time_contract(args)


def test_nested_model_parallel_namespace_read():
    # Nested-namespace tests encode the megatron-native --yaml-cfg layout
    # (parallel sizes nested under args.model_parallel); this is a hypothetical
    # shape for FlagScale runner configs, which flatten these to top-level CLI
    # flags (load_yaml preserves whatever structure the yaml declares).  The
    # contract stays correct if that layout ever reaches it.
    args = SimpleNamespace(
        model_parallel=SimpleNamespace(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
    )
    with pytest.raises(ValueError, match="tensor_model_parallel_size"):
        apply_grid_parse_time_contract(args)


def test_nested_sp_intent_preserved():
    # Same hypothetical megatron-native --yaml-cfg nesting as above; SP intent
    # must survive whether the flag arrives top-level or nested.
    args = SimpleNamespace(
        model_parallel=SimpleNamespace(
            tensor_model_parallel_size=1,
            sequence_parallel=True,
        )
    )
    apply_grid_parse_time_contract(args)
    assert args.mimo_sequence_parallel is True


def test_missing_args_pass_with_sp_false():
    # No parallel-size attributes at all (all defaults) and no SP flag.
    args = SimpleNamespace()
    apply_grid_parse_time_contract(args)
    assert args.mimo_sequence_parallel is False
