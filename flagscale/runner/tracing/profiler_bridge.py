# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate explicitly associated v5 device evidence; never infer call order."""

from __future__ import annotations

from typing import Any

_DATATYPES = {
    0: "ncclInt8",
    1: "ncclUint8",
    2: "ncclInt32",
    3: "ncclUint32",
    4: "ncclInt64",
    5: "ncclUint64",
    6: "ncclFloat16",
    7: "ncclFloat32",
    8: "ncclFloat64",
    9: "ncclBfloat16",
    10: "ncclFloat8e4m3",
    11: "ncclFloat8e5m2",
}


def _integer(value: Any) -> int:
    return value if type(value) is int else -1


def _hex(value: Any) -> int:
    try:
        return int(str(value), 16)
    except (ValueError, TypeError):
        return -1


def bridge_evidence(event: dict, records: tuple[dict, ...]) -> dict:
    result = {
        "inspector_correlation": "none",
        "inspector_candidate_count": 0,
        "inspector_kernel_status": "unknown",
        "kernel_execution_status": "unknown",
        "inspector_match_method": "profiler_api_parent",
        "inspector_correlation_reason": "api_task_mapping_missing",
    }
    host, pid = str(event.get("hostname", "")), _integer(event.get("pid"))
    token = _integer(event.get("profiler_process_token"))
    call = _integer(event.get("call_seq"))
    if pid <= 0 or token <= 0 or call <= 0:
        return result
    scoped = [r for r in records if r.get("hostname") == host and r.get("pid") == pid]
    if any(r.get("record_type") == "flagscale_bridge_error_v1" for r in scoped):
        return dict(result, inspector_correlation_reason="bridge_collection_error")
    candidates = [
        r
        for r in scoped
        if r.get("process_token") == token
        and r.get("call_seq") == call
        and r.get("record_type") == "flagscale_profiler_v5_v1"
    ]
    if not candidates:
        return result
    tasks = {_integer(r.get("task_id")) for r in candidates}
    result["inspector_candidate_count"] = len(tasks)
    # Multiple tasks per API need an explicit coverage contract, not an any()
    # completion rule. Until then the whole API remains unknown.
    if len(tasks) != 1 or min(tasks) <= 0:
        return dict(
            result,
            inspector_correlation="ambiguous",
            inspector_correlation_reason="multiple_device_tasks",
        )
    api = str(event.get("api", "")).removeprefix("nccl")
    comm = _hex(event.get("comm_inspector_id"))
    expected_count = event.get("count")
    expected_datatype = _DATATYPES.get(event.get("datatype"))
    if api in {"AllGather", "Broadcast"}:
        sizes = {0: 1, 1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2, 10: 1, 11: 1}
        size = sizes.get(event.get("datatype"))
        if size is None or type(expected_count) is not int or expected_count < 0:
            return dict(result, inspector_correlation_reason="invalid_api_parameters")
        expected_count *= size
        expected_datatype = "ncclInt8"
    for row in candidates:
        if (
            comm < 0
            or _hex(row.get("comm_id")) != comm
            or row.get("rank") != event.get("comm_rank")
            or row.get("nranks") != event.get("comm_nranks")
            or row.get("api") != api
            or row.get("count") != expected_count
            or row.get("datatype") != expected_datatype
            or (api in {"Send", "Recv"} and row.get("peer") != event.get("peer"))
        ):
            return dict(
                result,
                inspector_correlation="ambiguous",
                inspector_correlation_reason="mapping_identity_conflict",
            )
        if row.get("graph_captured") is not False:
            return dict(result, inspector_correlation_reason="graph_replay_identity_unavailable")
        if row.get("invalid") is not False:
            return dict(result, inspector_correlation_reason="invalid_channel_evidence")
    revisions: dict[int, dict] = {}
    previous_channels: dict[int, tuple[int, int]] = {}
    expected = None
    for row in sorted(candidates, key=lambda r: _integer(r.get("revision"))):
        rev = _integer(row.get("revision"))
        if rev < 1 or (rev in revisions and revisions[rev] != row):
            return dict(result, inspector_correlation_reason="conflicting_evidence_revisions")
        revisions[rev] = row
        n = _integer(row.get("expected_channels"))
        if n <= 0 or (expected is not None and expected != n):
            return dict(result, inspector_correlation_reason="channel_coverage_unavailable")
        expected = n
        channels = row.get("channels")
        if not isinstance(channels, list):
            return dict(result, inspector_correlation_reason="invalid_channel_evidence")
        current = {}
        for channel in channels:
            if not isinstance(channel, dict):
                return dict(result, inspector_correlation_reason="invalid_channel_evidence")
            key = _integer(channel.get("channel_id"))
            begin, end = _integer(channel.get("begin")), _integer(channel.get("end"))
            if key < 0 or key in current or begin <= 0 or end < 0:
                return dict(result, inspector_correlation_reason="invalid_channel_evidence")
            current[key] = (begin, end)
        for key, (begin, end) in previous_channels.items():
            new = current.get(key)
            if new is None or new[0] != begin or (end > 0 and new[1] != end):
                return dict(result, inspector_correlation_reason="conflicting_channel_evidence")
        previous_channels = current
    result.update(
        inspector_correlation="unique",
        inspector_call_sequence=call,
        inspector_completion=revisions[max(revisions)],
    )
    complete = len(previous_channels) == expected and all(
        end >= begin > 0 for begin, end in previous_channels.values()
    )
    result["inspector_correlation_reason"] = (
        "device_channels_complete" if complete else "device_channels_incomplete"
    )
    if complete:
        result.update(inspector_kernel_status="completed", kernel_execution_status="completed")
    return result
