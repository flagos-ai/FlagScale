# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from flagscale.runner.tracing.inspector import InspectorReader
from flagscale.runner.tracing.profiler_bridge import bridge_evidence


def event(**changes):
    value = dict(
        hostname="node",
        pid=10,
        profiler_process_token=999,
        call_seq=3,
        comm_inspector_id="1234",
        comm_rank=0,
        comm_nranks=2,
        api="ncclAllReduce",
        count=16,
        datatype=7,
        peer=-1,
        group_id=7,
        comm_api_seq=-1,
        profiler_bridge=True,
    )
    return dict(value, **changes)


def record(**changes):
    value = dict(
        record_type="flagscale_profiler_v5_v1",
        hostname="node",
        pid=10,
        process_token=999,
        call_seq=3,
        task_id=20,
        revision=1,
        comm_id="1234",
        rank=0,
        nranks=2,
        api="AllReduce",
        count=16,
        datatype="ncclFloat32",
        peer=-1,
        expected_channels=2,
        graph_captured=False,
        invalid=False,
        channels=[dict(channel_id=1, begin=10, end=20), dict(channel_id=3, begin=12, end=22)],
    )
    return dict(value, **changes)


def test_exact_mapping_works_without_inferred_group_sequence():
    result = bridge_evidence(event(), (record(),))
    assert result["kernel_execution_status"] == "completed"
    assert result["inspector_call_sequence"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"hostname": "other"},
        {"pid": 11},
        {"process_token": 1000},
        {"call_seq": 4},
        {"comm_id": "5678"},
        {"rank": 1},
        {"nranks": 4},
        {"api": "Broadcast"},
        {"count": 32},
        {"datatype": "ncclFloat16"},
    ],
)
def test_other_call_or_signature_never_supplies_completion(changes):
    assert bridge_evidence(event(), (record(**changes),))["kernel_execution_status"] == "unknown"


def test_p2p_peer_is_checked():
    call = event(api="ncclSend", peer=1)
    assert (
        bridge_evidence(call, (record(api="Send", peer=1),))["kernel_execution_status"]
        == "completed"
    )
    assert (
        bridge_evidence(call, (record(api="Send", peer=2),))["kernel_execution_status"] == "unknown"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_channels": 3},
        {"channels": []},
        {"expected_channels": 0},
        {"graph_captured": True},
        {"invalid": True},
        {"channels": [dict(channel_id=0, begin=10, end=0)]},
        {"channels": [dict(channel_id=0, begin=10, end=20)] * 2},
    ],
)
def test_incomplete_or_unsupported_evidence_stays_unknown(changes):
    assert bridge_evidence(event(), (record(**changes),))["kernel_execution_status"] == "unknown"


def test_multiple_tasks_do_not_confirm_the_entire_call():
    evidence = bridge_evidence(event(), (record(), record(task_id=21)))
    assert evidence["kernel_execution_status"] == "unknown"
    assert evidence["inspector_correlation_reason"] == "multiple_device_tasks"


def test_channel_updates_are_monotonic():
    initial = record(channels=[])
    final = record(revision=2)
    assert bridge_evidence(event(), (initial, final))["kernel_execution_status"] == "completed"
    changed = copy.deepcopy(final)
    changed["revision"] = 3
    changed["channels"][0]["end"] = 30
    assert (
        bridge_evidence(event(), (initial, final, changed))["kernel_execution_status"] == "unknown"
    )


def test_conflicting_same_revision_is_rejected():
    assert (
        bridge_evidence(event(), (record(), record(channels=[])))["kernel_execution_status"]
        == "unknown"
    )


def test_collector_error_invalidates_prior_positive_evidence():
    error = dict(record_type="flagscale_bridge_error_v1", hostname="node", pid=10)
    assert bridge_evidence(event(), (record(), error))["kernel_execution_status"] == "unknown"


def test_reader_updates_live_mapping_without_duplicates(tmp_path):
    path = tmp_path / "bridge.jsonl"
    initial = record(channels=[])
    path.write_text(json.dumps(initial) + "\n")
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(event())["kernel_execution_status"] == "unknown"
    path.write_text(json.dumps(initial) + "\n" + json.dumps(record(revision=2)) + "\n")
    result = reader.poll().for_call(event())
    assert result["kernel_execution_status"] == "completed"
    assert result == reader.poll().for_call(event())


def test_enabled_bridge_does_not_fall_back_when_mapping_is_absent(tmp_path):
    result = InspectorReader(tmp_path, enabled=True).poll().for_call(event())
    assert result["inspector_correlation_reason"] == "api_task_mapping_missing"
    assert result["kernel_execution_status"] == "unknown"


@pytest.mark.parametrize("api", ["AllGather", "Broadcast"])
@pytest.mark.parametrize("datatype,width", [(1, 1), (4, 8), (7, 4), (9, 2)])
def test_copy_collectives_validate_official_byte_task_encoding(api, datatype, width):
    call = event(api="nccl" + api, datatype=datatype)
    task = record(api=api, count=16 * width, datatype="ncclInt8")
    assert bridge_evidence(call, (task,))["kernel_execution_status"] == "completed"
    task["count"] += 1
    assert bridge_evidence(call, (task,))["kernel_execution_status"] == "unknown"
