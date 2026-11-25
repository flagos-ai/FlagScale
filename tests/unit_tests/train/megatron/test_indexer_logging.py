# Copyright 2026 FlagOS Contributors

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.plugin.platform import get_platform
from megatron.training.distributed_backends import resolve_distributed_backend
from megatron.training.indexer_logging import _get_indexer_logging_layer_counts, log_indexer_metrics


def args(**kwargs):
    settings = dict(
        num_layers=4,
        mtp_num_layers=0,
        csa_compress_ratios=[0, 0, 4, 128],
        calculate_per_token_loss=False,
        cuda_graph_impl="none",
    )
    return SimpleNamespace(**(settings | kwargs))


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, (4, 1)),
        ({"csa_dense_mode": True}, (4, 0)),
        (
            {
                "experimental_attention_variant": "dsa",
                "csa_compress_ratios": None,
                "indexer_types": ["full", "share", "shared", "full"],
            },
            (4, 2),
        ),
        (
            {
                "experimental_attention_variant": "dsa",
                "csa_compress_ratios": None,
                "dsa_indexer_topk_freq": 2,
            },
            (4, 2),
        ),
    ],
)
def test_indexer_count(overrides, expected):
    assert _get_indexer_logging_layer_counts(args(**overrides)) == expected


@pytest.mark.parametrize("repeated,expected", [(False, 3), (True, 2)])
def test_hybrid_mtp_counts_modules_separately_from_slots(repeated, expected):
    pattern = "DDDD/DD/DD"
    parsed = parse_hybrid_pattern(pattern)
    config = MLATransformerConfig(
        num_layers=len(parsed.main_pattern),
        mtp_num_layers=parsed.mtp_num_depths,
        hidden_size=128,
        num_attention_heads=4,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        csa_compress_ratios=[0, 0, 4, 128, 4, 128],
        mtp_use_repeated_layer=repeated,
        use_cpu_initialization=True,
    )
    # FlagScale receives the pattern as a training argument alongside model fields.
    training_args = SimpleNamespace(**vars(config), hybrid_layer_pattern=pattern)
    assert _get_indexer_logging_layer_counts(training_args) == (6, expected)


def test_repeated_mtp_logs_loss_per_module(monkeypatch):
    # Two prediction depths contribute to one MTP slot. Use the real tracker and
    # logger with identity reductions, as on one rank.
    helper = DSAIndexerLossLoggingHelper
    tracker = {"values": torch.zeros(3)}
    monkeypatch.setattr(helper, "tracker", tracker)
    monkeypatch.setattr(helper, "reduce_loss_in_tracker", lambda **kwargs: None)
    config = args(
        num_layers=1,
        mtp_num_layers=2,
        csa_compress_ratios=None,
        experimental_attention_variant="dsa",
        mtp_use_repeated_layer=True,
    )
    for layer, value in [(1, 2.0), (2, 4.0), (2, 6.0)]:
        helper.save_loss_to_tracker(torch.tensor(value), layer, 3)
    totals = {}
    log_indexer_metrics(config, 1.0, 1, None, None, totals, ProcessGroupCollection())
    assert totals["indexer loss"].item() == 6.0


@pytest.fixture(scope="module")
def four_rank_groups():
    platform = get_platform()
    if platform.device_name() != "cpu":
        platform.set_device(int(os.environ["LOCAL_RANK"]))
    owns_world = not dist.is_initialized()
    if owns_world:
        backend = os.environ.get("DISTRIBUTED_BACKEND")
        if backend is None:
            backend = dist.Backend.default_device_backend_map[platform.device_name()]
        if backend == "gloo":
            pytest.skip("This regression requires native all_reduce AVG support")
        dist.init_process_group(
            backend=resolve_distributed_backend(backend), timeout=timedelta(seconds=90)
        )
    rank = dist.get_rank()
    local_groups = []
    topology = {}
    try:
        for name, memberships in [
            ("pp", [[0, 2], [1, 3]]),
            ("replica", [[0, 1], [2, 3]]),
            ("singleton", [[0], [1], [2], [3]]),
        ]:
            for members in memberships:
                group = dist.new_group(members, timeout=timedelta(seconds=90))
                if rank in members:
                    topology[name] = group
                    local_groups.append(group)
        yield topology
    finally:
        for group in reversed(local_groups):
            dist.destroy_process_group(group)
        if owns_world:
            dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("WORLD_SIZE") != "4", reason="requires torchrun with four ranks")
@pytest.mark.parametrize("replica_axis", ["cp", "dp"])
def test_empty_pipeline_stage_and_replica_reduction(monkeypatch, replica_axis, four_rank_groups):
    platform = get_platform()
    rank = dist.get_rank()
    groups = ProcessGroupCollection(
        pp=four_rank_groups["pp"],
        cp=four_rank_groups["replica" if replica_axis == "cp" else "singleton"],
        dp=four_rank_groups["replica" if replica_axis == "dp" else "singleton"],
    )
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
    for per_token, expected in [(False, 3.0), (True, 6.0 if replica_axis == "cp" else 3.0)]:
        if rank >= 2:  # PP stage 0 never owns an indexer. Stage 1 has losses 2 and 4.
            value = torch.tensor(
                2.0 * (rank - 1), device=platform.device(platform.current_device())
            )
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(value, 3, 4)
        totals = {}
        log_indexer_metrics(
            args(calculate_per_token_loss=per_token), 1.0, 1, None, None, totals, groups
        )
        assert totals["indexer loss"].item() == expected
