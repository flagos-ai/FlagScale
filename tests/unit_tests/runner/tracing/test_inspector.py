# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from flagscale.runner.tracing.inspector import InspectorIndex, InspectorReader


def _payload(start_us=1_000_100):
    return {
        "header": {
            "id": "0x1234",
            "rank": 0,
            "n_ranks": 2,
            "nnodes": 1,
        },
        "metadata": {
            "inspector_output_format_version": "v4.0",
            "dump_timestamp_us": start_us + 100,
            "hostname": "host-a",
            "pid": 10,
        },
        "coll_perf": {
            "coll": "AllReduce",
            "coll_sn": 7,
            "coll_msg_size_bytes": 2048,
            "coll_exec_time_us": 100,
            "coll_timing_source": "kernel_gpu",
            "event_trace_ts": {
                "coll_start_ts": start_us,
                "coll_stop_ts": start_us + 100,
                "kernel_events": [
                    {
                        "channel_id": 0,
                        "kernel_start_ts": start_us + 5,
                        "kernel_stop_ts": start_us + 95,
                    }
                ],
            },
        },
    }


def _event():
    return {
        "hostname": "host-a",
        "pid": 10,
        "comm_rank": 0,
        "api": "ncclAllReduce",
        "count": 1024,
        "datatype": 6,
        "timestamp_unix_ns": 1_000_000_000,
    }


def test_unique_completed_record_is_positive_kernel_evidence(tmp_path):
    output = tmp_path / "inspector.json"
    output.write_text(json.dumps(_payload()) + "\n", encoding="utf-8")

    evidence = InspectorReader(tmp_path, enabled=True).poll().for_call(_event())

    assert evidence["inspector_correlation"] == "unique"
    assert evidence["inspector_kernel_status"] == "completed"
    assert evidence["kernel_execution_status"] == "completed"
    assert evidence["inspector_completion"]["timing_source"] == "kernel_gpu"
    assert len(evidence["inspector_completion"]["kernel_events"]) == 1


def test_absent_or_ambiguous_inspector_record_stays_unknown(tmp_path):
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(_event())["inspector_kernel_status"] == "unknown"

    output = tmp_path / "inspector.jsonl"
    output.write_text(
        json.dumps(_payload()) + "\n" + json.dumps(_payload(start_us=1_000_200)) + "\n",
        encoding="utf-8",
    )
    evidence = reader.poll().for_call(_event())

    assert evidence["inspector_correlation"] == "ambiguous"
    assert evidence["inspector_candidate_count"] == 2
    assert evidence["inspector_kernel_status"] == "unknown"
    assert evidence["kernel_execution_status"] == "unknown"


def test_disabled_inspector_is_reported_as_not_collected():
    evidence = InspectorIndex(False).for_call(_event())

    assert evidence["inspector_correlation"] == "not_collected"
    assert evidence["inspector_kernel_status"] == "not_collected"
    assert evidence["kernel_execution_status"] == "unknown"


def _identified_event(**changes):
    event = dict(_event(), comm_inspector_id="1234", comm_nranks=2, comm_seq=7, comm_api_seq=7)
    event.update(changes)
    return event


def _identified_payload(comm_id="0x1234", sequence=7):
    payload = _payload()
    payload["header"]["id"] = comm_id
    payload["coll_perf"]["coll_sn"] = sequence
    return payload


def _write_payloads(path, payloads):
    path.write_text("\n".join(json.dumps(payload) for payload in payloads) + "\n")


def test_repeated_calls_use_communicator_and_sequence(tmp_path):
    _write_payloads(
        tmp_path / "inspector.log", [_identified_payload(sequence=s) for s in [6, 7, 8]]
    )
    index = InspectorReader(tmp_path, enabled=True).poll()
    assert len(index.completions) == 3
    evidence = index.for_call(_identified_event())
    assert evidence["inspector_correlation"] == "unique"
    assert evidence["inspector_completion"]["sequence_number"] == 7
    assert evidence["kernel_execution_status"] == "completed"
    assert evidence["inspector_match_method"] == "communicator_sequence"


def test_identical_calls_on_other_communicators_are_excluded(tmp_path):
    _write_payloads(
        tmp_path / "inspector.log", [_identified_payload(), _identified_payload("0x5678")]
    )
    index = InspectorReader(tmp_path, enabled=True).poll()
    assert len(index.completions) == 2
    evidence = index.for_call(_identified_event())
    assert evidence["inspector_correlation"] == "unique"
    assert evidence["inspector_completion"]["communicator_id"] == "0x1234"


def test_missing_sequence_does_not_match_another_completed_call(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload(sequence=6)])
    evidence = InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())
    assert evidence["inspector_correlation"] == "none"
    assert evidence["kernel_execution_status"] == "unknown"


def test_single_rank_call_does_not_borrow_two_rank_evidence(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload()])
    evidence = (
        InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event(comm_nranks=1))
    )
    assert evidence["inspector_correlation"] == "none"
    assert evidence["kernel_execution_status"] == "unknown"


def test_unknown_native_communicator_does_not_fall_back_to_time_matching(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload()])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(comm_inspector_id=""))
    )
    assert evidence["inspector_correlation"] == "none"
    assert evidence["kernel_execution_status"] == "unknown"


def test_exact_sequence_can_match_after_a_long_device_queue(tmp_path):
    payload = _identified_payload()
    payload["coll_perf"]["event_trace_ts"]["coll_start_ts"] += 20_000_000
    _write_payloads(tmp_path / "inspector.log", [payload])
    evidence = InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())
    assert evidence["inspector_correlation"] == "unique"


def test_duplicate_log_and_json_export_is_counted_once(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload()])
    _write_payloads(tmp_path / "inspector.jsonl", [_identified_payload()])
    reader = InspectorReader(tmp_path, enabled=True)
    assert len(reader.poll().completions) == 1
    assert len(reader.poll().completions) == 1


def test_cpu_timing_is_not_device_completion_evidence(tmp_path):
    payload = _identified_payload()
    payload["coll_perf"]["coll_timing_source"] = "collective_cpu"
    _write_payloads(tmp_path / "inspector.log", [payload])
    evidence = InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())
    assert evidence["inspector_correlation"] == "unique"
    assert evidence["kernel_execution_status"] == "unknown"


def test_incomplete_channel_timestamps_remain_unknown(tmp_path):
    payload = _identified_payload()
    payload["coll_perf"]["event_trace_ts"]["kernel_events"][0]["kernel_stop_ts"] = 0
    _write_payloads(tmp_path / "inspector.log", [payload])
    evidence = InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())
    assert evidence["kernel_execution_status"] == "unknown"


def test_v4_serialized_communicator_hash_is_supported(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload("0xc1e71ef3608318")])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(comm_inspector_id="c1e71ef3608318d2"))
    )
    assert evidence["inspector_correlation"] == "unique"


def test_other_versions_do_not_assume_v4_hash_truncation(tmp_path):
    payload = _identified_payload("0xc1e71ef3608318")
    payload["metadata"]["inspector_output_format_version"] = "v5.0"
    _write_payloads(tmp_path / "inspector.log", [payload])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(comm_inspector_id="c1e71ef3608318d2"))
    )
    assert evidence["inspector_correlation"] == "none"


def test_partial_log_line_is_retried_when_complete(tmp_path):
    path = tmp_path / "inspector.log"
    encoded = json.dumps(_identified_payload())
    path.write_text(encoded[: len(encoded) // 2])
    reader = InspectorReader(tmp_path, enabled=True)
    assert not reader.poll().completions
    path.write_text(encoded + "\n")
    assert len(reader.poll().completions) == 1


def test_v41_rccl_communicator_hash_is_supported(tmp_path):
    payload = _identified_payload("0x45c7f7370d5396")
    payload["metadata"]["inspector_output_format_version"] = "v4.1"
    _write_payloads(tmp_path / "inspector.log", [payload])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(comm_inspector_id="45c7f7370d539655"))
    )
    assert evidence["inspector_correlation"] == "unique"
    assert evidence["kernel_execution_status"] == "completed"


def test_mixed_apis_use_type_local_sequence_not_global_sequence(tmp_path):
    payload = _identified_payload(sequence=0)
    payload["coll_perf"]["coll"] = "AllGather"
    _write_payloads(tmp_path / "inspector.log", [payload])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(api="ncclAllGather", comm_seq=6, comm_api_seq=0))
    )
    assert evidence["inspector_correlation"] == "unique"
    assert evidence["inspector_completion"]["sequence_number"] == 0
    assert evidence["inspector_call_sequence"] == 0


def test_missing_or_invalid_type_sequence_does_not_fall_back_to_global(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload()])
    index = InspectorReader(tmp_path, enabled=True).poll()
    for value in [None, -1]:
        evidence = index.for_call(_identified_event(comm_seq=7, comm_api_seq=value))
        assert evidence["inspector_correlation"] == "none"
        assert evidence["kernel_execution_status"] == "unknown"


def test_different_apis_can_share_type_local_sequence(tmp_path):
    first = _identified_payload(sequence=0)
    second = _identified_payload(sequence=0)
    second["coll_perf"]["coll"] = "Broadcast"
    _write_payloads(tmp_path / "inspector.log", [first, second])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(api="ncclBroadcast", comm_seq=9, comm_api_seq=0))
    )
    assert evidence["inspector_correlation"] == "unique"
    assert evidence["inspector_completion"]["api"] == "Broadcast"


def test_group_sequence_never_confirms_a_possibly_reordered_call(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload()])
    index = InspectorReader(tmp_path, enabled=True).poll()
    for event in [_identified_event(group_id=1), dict(_event(), group_id=1)]:
        evidence = index.for_call(event)
        assert evidence["kernel_execution_status"] == "unknown"
        assert evidence["inspector_correlation_reason"] == "group_api_task_mapping_unavailable"


def test_p2p_reports_unsupported_mapping_instead_of_missing_data(tmp_path):
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(api="ncclSend", comm_api_seq=-1))
    )
    assert evidence["kernel_execution_status"] == "unknown"
    assert evidence["inspector_correlation_reason"] == "p2p_api_task_mapping_unavailable"


def test_invalidated_sequence_reports_reason_without_time_fallback(tmp_path):
    _write_payloads(tmp_path / "inspector.log", [_identified_payload()])
    evidence = (
        InspectorReader(tmp_path, enabled=True)
        .poll()
        .for_call(_identified_event(comm_api_seq=-1, group_id=0))
    )
    assert evidence["kernel_execution_status"] == "unknown"
    assert evidence["inspector_correlation_reason"] == "profiler_sequence_unavailable"


def test_long_lived_reader_accepts_cpu_to_gpu_refinement(tmp_path):
    partial = _identified_payload()
    partial["coll_perf"]["coll_timing_source"] = "collective_cpu"
    partial["coll_perf"]["event_trace_ts"]["kernel_events"] = []
    path = tmp_path / "inspector.log"
    _write_payloads(path, [partial])
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(_identified_event())["kernel_execution_status"] == "unknown"
    _write_payloads(path, [partial, _identified_payload()])
    evidence = reader.poll().for_call(_identified_event())
    assert evidence["kernel_execution_status"] == "completed"
    assert len(reader.poll().completions) == 1
    assert evidence == InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())


def test_complete_then_stale_partial_does_not_regress(tmp_path):
    partial = _identified_payload()
    partial["coll_perf"]["event_trace_ts"]["kernel_events"][0]["kernel_stop_ts"] = 0
    path = tmp_path / "inspector.log"
    _write_payloads(path, [_identified_payload()])
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(_identified_event())["kernel_execution_status"] == "completed"
    _write_payloads(path, [_identified_payload(), partial])
    assert reader.poll().for_call(_identified_event())["kernel_execution_status"] == "completed"
    assert len(reader.poll().completions) == 1


def test_incomplete_channel_can_be_completed(tmp_path):
    partial = _identified_payload()
    partial["coll_perf"]["event_trace_ts"]["kernel_events"][0]["kernel_stop_ts"] = 0
    path = tmp_path / "inspector.log"
    _write_payloads(path, [partial])
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(_identified_event())["kernel_execution_status"] == "unknown"
    _write_payloads(path, [partial, _identified_payload()])
    assert reader.poll().for_call(_identified_event())["kernel_execution_status"] == "completed"


def test_conflicting_gpu_revisions_remain_ambiguous(tmp_path):
    changed = _identified_payload()
    changed["coll_perf"]["event_trace_ts"]["kernel_events"][0]["kernel_stop_ts"] += 1
    path = tmp_path / "inspector.log"
    _write_payloads(path, [_identified_payload()])
    reader = InspectorReader(tmp_path, enabled=True)
    reader.poll()
    _write_payloads(path, [_identified_payload(), changed])
    result = reader.poll().for_call(_identified_event())
    assert result["inspector_correlation"] == "ambiguous"
    assert result["kernel_execution_status"] == "unknown"
    assert result == InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())


def test_refinement_cannot_borrow_evidence_from_another_sequence(tmp_path):
    partial = _identified_payload()
    partial["coll_perf"]["coll_timing_source"] = "collective_cpu"
    partial["coll_perf"]["event_trace_ts"]["kernel_events"] = []
    _write_payloads(tmp_path / "inspector.log", [partial, _identified_payload(sequence=8)])
    result = InspectorReader(tmp_path, enabled=True).poll().for_call(_identified_event())
    assert result["kernel_execution_status"] == "unknown"
