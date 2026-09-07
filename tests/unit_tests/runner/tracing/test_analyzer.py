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

from flagscale.runner.tracing.analyzer import TraceAnalyzer

RUN_ID = "run-1"
COMM = "0123456789abcdef"


def _event(event_type, **overrides):
    event = {
        "schema_version": 1,
        "event": event_type,
        "run_id": RUN_ID,
        "timestamp_unix_ns": 1_000_000_000,
        "timestamp_mono_ns": 1_000_000_000,
        "rank": 0,
        "pid": 10,
        "comm_uid_hash": COMM,
        "comm_rank": 0,
        "comm_nranks": 2,
        "comm_seq": 7,
        "api": "ncclAllReduce",
        "phase": "enter",
        "count": 1024,
        "datatype": 6,
        "op": 0,
        "root": -1,
        "result": 0,
        "call_seq": 1,
    }
    event.update(overrides)
    return event


def _ingest(analyzer, event, observed=1.0, unix_ns=1_000_000_000):
    return analyzer.ingest(
        event,
        observed_monotonic_s=observed,
        observed_unix_ns=unix_ns,
    )


def test_missing_enter_reports_live_missing_rank_without_claiming_crash():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        heartbeat_timeout_s=10,
        collective_timeout_s=5,
    )
    _ingest(analyzer, _event("comm_init", rank=0, comm_rank=0))
    _ingest(analyzer, _event("comm_init", rank=1, comm_rank=1))
    _ingest(
        analyzer,
        _event(
            "heartbeat",
            rank=1,
            last_nccl_api="ncclAllReduce",
            last_nccl_phase="exit",
            last_nccl_seq=6,
            last_progress_mono_ns=900_000_000,
        ),
        observed=2.0,
        unix_ns=1_000_000_000,
    )
    _ingest(analyzer, _event("nccl_call", rank=0, comm_rank=0), observed=2.0)

    findings = analyzer.scan(now_monotonic_s=8.0, now_unix_ns=8_000_000_000)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.hang_type == "collective_missing_enter"
    assert finding.details["missing_comm_ranks"] == [1]
    assert finding.details["reason"] == "rank_alive_but_not_entered"
    assert finding.details["missing_rank_status"][0]["heartbeat"] == "alive"


def test_missing_enter_correlates_a_stale_heartbeat_as_suspected_crash():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        heartbeat_timeout_s=3,
        collective_timeout_s=2,
    )
    _ingest(analyzer, _event("comm_init", rank=1, comm_rank=1))
    _ingest(analyzer, _event("heartbeat", rank=1), observed=1.0)
    _ingest(analyzer, _event("nccl_call", rank=0, comm_rank=0), observed=1.0)

    findings = analyzer.scan(now_monotonic_s=5.0, now_unix_ns=5_000_000_000)

    assert findings[0].details["reason"] == "rank_exit_or_crash_suspected"
    assert findings[0].details["confidence"] == "suspected"


def test_api_mismatch_is_reported_without_missing_enter_when_all_ranks_enter():
    analyzer = TraceAnalyzer(run_id=RUN_ID, collective_timeout_s=1)
    _ingest(analyzer, _event("nccl_call", comm_rank=0, rank=0), observed=1.0)
    _ingest(
        analyzer,
        _event("nccl_call", comm_rank=1, rank=1, api="ncclBroadcast", root=0),
        observed=1.0,
    )

    findings = analyzer.scan(now_monotonic_s=3.0, now_unix_ns=3_000_000_000)
    assert [finding.hang_type for finding in findings] == ["collective_signature_mismatch"]
    assert findings[0].details["mismatch_type"] == "api_mismatch"

    assert analyzer.scan(now_monotonic_s=10.0, now_unix_ns=10_000_000_000) == []


def test_api_mismatch_does_not_hide_a_rank_that_never_enters():
    analyzer = TraceAnalyzer(run_id=RUN_ID, collective_timeout_s=1)
    _ingest(
        analyzer,
        _event("nccl_call", comm_rank=0, rank=0, comm_nranks=3),
        observed=1.0,
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            comm_rank=1,
            rank=1,
            comm_nranks=3,
            api="ncclBroadcast",
            root=0,
        ),
        observed=1.0,
    )

    findings = analyzer.scan(now_monotonic_s=3.0, now_unix_ns=3_000_000_000)

    assert [finding.hang_type for finding in findings] == [
        "collective_signature_mismatch",
        "collective_missing_enter",
    ]
    assert findings[1].details["entered_comm_ranks"] == [0, 1]
    assert findings[1].details["missing_comm_ranks"] == [2]


def test_parameter_and_root_mismatch_are_distinguished():
    parameter_analyzer = TraceAnalyzer(run_id=RUN_ID)
    _ingest(parameter_analyzer, _event("nccl_call", comm_rank=0, count=1024))
    _ingest(parameter_analyzer, _event("nccl_call", comm_rank=1, count=2048))
    parameter = parameter_analyzer.scan(now_monotonic_s=2.0, now_unix_ns=2_000_000_000)
    assert parameter[0].details["mismatch_type"] == "parameter_mismatch"

    root_analyzer = TraceAnalyzer(run_id=RUN_ID)
    _ingest(
        root_analyzer,
        _event("nccl_call", comm_rank=0, api="ncclBroadcast", op=-1, root=0),
    )
    _ingest(
        root_analyzer,
        _event("nccl_call", comm_rank=1, api="ncclBroadcast", op=-1, root=1),
    )
    root = root_analyzer.scan(now_monotonic_s=2.0, now_unix_ns=2_000_000_000)
    assert root[0].details["mismatch_type"] == "root_mismatch"


def test_delayed_enter_is_reported_after_every_rank_enters():
    analyzer = TraceAnalyzer(run_id=RUN_ID, delayed_enter_threshold_s=5)
    _ingest(
        analyzer,
        _event("nccl_call", comm_rank=0, timestamp_unix_ns=1_000_000_000),
    )
    _ingest(
        analyzer,
        _event("nccl_call", comm_rank=1, timestamp_unix_ns=8_000_000_000),
    )

    findings = analyzer.scan(now_monotonic_s=9.0, now_unix_ns=9_000_000_000)

    assert findings[0].hang_type == "delayed_collective_enter"
    assert findings[0].details["slow_comm_ranks"] == [1]
    assert findings[0].details["enter_spread_s"] == 7.0


def test_checkpoint_missing_enter_uses_checkpoint_timeout_and_still_reports_hang():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        collective_timeout_s=5,
        checkpoint_timeout_s=20,
    )
    _ingest(analyzer, _event("comm_init", rank=0, comm_rank=0))
    _ingest(analyzer, _event("comm_init", rank=1, comm_rank=1))
    _ingest(
        analyzer,
        _event("heartbeat", rank=1, phase="checkpointing"),
        observed=1.0,
    )
    _ingest(analyzer, _event("nccl_call", rank=0, comm_rank=0), observed=1.0)

    assert analyzer.scan(now_monotonic_s=8.0, now_unix_ns=8_000_000_000) == []

    findings = analyzer.scan(now_monotonic_s=22.0, now_unix_ns=22_000_000_000)
    assert [finding.hang_type for finding in findings] == ["collective_missing_enter"]
    assert findings[0].details["detection_phase"] == "checkpointing"
    assert findings[0].details["detection_threshold_s"] == 20
    assert findings[0].details["threshold_reason"] == "heartbeat_phase_checkpointing"


def test_training_missing_enter_keeps_normal_timeout():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        collective_timeout_s=5,
        checkpoint_timeout_s=20,
    )
    _ingest(analyzer, _event("comm_init", rank=1, comm_rank=1))
    _ingest(analyzer, _event("heartbeat", rank=1, phase="train"), observed=1.0)
    _ingest(analyzer, _event("nccl_call", rank=0, comm_rank=0), observed=1.0)

    findings = analyzer.scan(now_monotonic_s=8.0, now_unix_ns=8_000_000_000)
    assert [finding.hang_type for finding in findings] == ["collective_missing_enter"]
    assert findings[0].details["detection_phase"] == "train"
    assert findings[0].details["detection_threshold_s"] == 5
    assert findings[0].details["threshold_reason"] == "normal_collective_phase"


def test_checkpoint_delayed_enter_keeps_phase_after_heartbeat_returns_to_train():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        delayed_enter_threshold_s=5,
        checkpoint_timeout_s=20,
    )
    _ingest(
        analyzer,
        _event("heartbeat", rank=0, phase="checkpointing"),
        observed=1.0,
    )
    _ingest(
        analyzer,
        _event("nccl_call", rank=0, comm_rank=0, timestamp_unix_ns=1_000_000_000),
        observed=1.0,
    )
    assert analyzer.scan(now_monotonic_s=2.0, now_unix_ns=2_000_000_000) == []

    _ingest(
        analyzer,
        _event(
            "heartbeat",
            rank=0,
            phase="train",
            timestamp_unix_ns=3_000_000_000,
        ),
        observed=3.0,
    )
    _ingest(
        analyzer,
        _event("nccl_call", comm_rank=1, rank=1, timestamp_unix_ns=8_000_000_000),
        observed=8.0,
    )

    assert analyzer.scan(now_monotonic_s=9.0, now_unix_ns=9_000_000_000) == []


def test_events_from_other_runs_are_ignored():
    analyzer = TraceAnalyzer(run_id=RUN_ID)
    accepted = _ingest(analyzer, _event("nccl_call", run_id="other-run"))
    assert accepted is False
    assert analyzer.scan(now_monotonic_s=100.0, now_unix_ns=100_000_000_000) == []


def test_process_heartbeat_timeout_is_reported_as_suspected_liveness_failure():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        initial_heartbeat_timeout_s=2,
        heartbeat_timeout_s=3,
        detect_heartbeat_timeouts=True,
    )
    _ingest(analyzer, _event("process_start", rank=1, pid=42), observed=1.0)
    _ingest(analyzer, _event("heartbeat", rank=1, pid=42), observed=1.5)

    findings = analyzer.scan(now_monotonic_s=5.0, now_unix_ns=5_000_000_000)

    assert findings[0].hang_type == "rank_heartbeat_timeout"
    assert findings[0].details["timeout_type"] == "subsequent_heartbeat"
    assert findings[0].details["confidence"] == "suspected"


def test_unique_p2p_send_recv_pair_is_resolved_without_a_finding():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    _ingest(
        analyzer,
        _event("nccl_call", api="ncclSend", comm_rank=0, peer=1, call_seq=1),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=1,
            pid=11,
            comm_rank=1,
            peer=0,
            call_seq=2,
        ),
    )

    assert analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000) == []


def test_p2p_calls_to_different_destination_ranks_are_matched_independently():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclSend",
            comm_nranks=3,
            comm_rank=0,
            peer=1,
            call_seq=1,
            p2p_op_index=0,
        ),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclSend",
            comm_nranks=3,
            comm_rank=0,
            peer=2,
            call_seq=2,
            p2p_op_index=1,
        ),
    )

    assert set(analyzer._p2p_calls) == {
        (RUN_ID, COMM, 0, 1),
        (RUN_ID, COMM, 0, 2),
    }

    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=1,
            pid=11,
            comm_nranks=3,
            comm_rank=1,
            peer=0,
            call_seq=1,
            p2p_op_index=0,
        ),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=2,
            pid=12,
            comm_nranks=3,
            comm_rank=2,
            peer=0,
            call_seq=1,
            p2p_op_index=0,
        ),
    )

    assert analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000) == []
    assert analyzer._p2p_calls == {}


def test_indexed_p2p_calls_do_not_accumulate_resolved_history():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    call_count = 200
    last_send = None
    for operation_index in range(call_count):
        last_send = _event(
            "nccl_call",
            api="ncclSend",
            comm_rank=0,
            peer=1,
            call_seq=operation_index + 1,
            p2p_op_index=operation_index,
            p2p_op_index_scope="peer_direction",
        )
        _ingest(
            analyzer,
            last_send,
        )

    direction = analyzer._p2p_calls[(RUN_ID, COMM, 0, 1)]
    assert len(direction.calls) == call_count
    assert len(direction.by_operation) == call_count

    for operation_index in reversed(range(call_count)):
        _ingest(
            analyzer,
            _event(
                "nccl_call",
                api="ncclRecv",
                rank=1,
                pid=11,
                comm_rank=1,
                peer=0,
                call_seq=call_count + operation_index + 1,
                p2p_op_index=operation_index,
                p2p_op_index_scope="peer_direction",
            ),
        )

    assert analyzer._p2p_calls == {}
    assert last_send is not None
    _ingest(analyzer, last_send)
    assert analyzer._p2p_calls == {}
    assert analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000) == []


def test_unique_p2p_parameter_mismatch_is_confirmed():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    _ingest(
        analyzer,
        _event("nccl_call", api="ncclSend", comm_rank=0, peer=1, call_seq=1),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=1,
            pid=11,
            comm_rank=1,
            peer=0,
            count=2048,
            call_seq=2,
        ),
    )

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)

    assert [finding.hang_type for finding in findings] == ["p2p_call_mismatch"]
    assert findings[0].details["confidence"] == "confirmed"


def test_missing_p2p_counterpart_is_only_suspected():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    _ingest(
        analyzer,
        _event("nccl_call", api="ncclSend", comm_rank=0, peer=1, call_seq=1),
    )

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)

    assert [finding.hang_type for finding in findings] == ["p2p_missing_counterpart"]
    assert findings[0].details["confidence"] == "suspected"


def test_missing_p2p_burst_on_same_peer_pair_is_coalesced():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    calls = [
        _event(
            "nccl_call",
            api="ncclSend",
            comm_seq=10,
            comm_rank=0,
            peer=1,
            call_seq=10,
            p2p_op_index=4,
            timestamp_unix_ns=1_000_000_000,
        ),
        _event(
            "nccl_call",
            api="ncclRecv",
            comm_seq=11,
            comm_rank=0,
            peer=1,
            call_seq=11,
            p2p_op_index=4,
            timestamp_unix_ns=1_100_000_000,
        ),
        _event(
            "nccl_call",
            api="ncclSend",
            comm_seq=12,
            comm_rank=0,
            peer=1,
            call_seq=12,
            p2p_op_index=5,
            timestamp_unix_ns=1_200_000_000,
        ),
    ]
    for event in calls:
        _ingest(analyzer, event)

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)

    assert [finding.hang_type for finding in findings] == ["p2p_missing_counterpart"]
    finding = findings[0]
    assert finding.comm_seq == 10
    assert finding.details["aggregated_missing_call_count"] == 3
    assert finding.details["comm_seq_range"] == [10, 12]
    assert finding.details["local_p2p_op_index_range"] == [4, 5]
    assert finding.details["affected_directions"] == [
        {"src_comm_rank": 0, "dst_comm_rank": 1, "observed_api": "ncclSend"},
        {"src_comm_rank": 1, "dst_comm_rank": 0, "observed_api": "ncclRecv"},
    ]


def test_missing_p2p_calls_on_different_communicators_remain_separate():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclSend",
            comm_rank=0,
            peer=1,
            call_seq=1,
            timestamp_unix_ns=1_000_000_000,
        ),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclSend",
            comm_uid_hash="fedcba9876543210",
            comm_rank=0,
            peer=1,
            call_seq=2,
            timestamp_unix_ns=1_100_000_000,
        ),
    )

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)

    assert [finding.comm_uid_hash for finding in findings] == [
        COMM,
        "fedcba9876543210",
    ]


def test_late_p2p_counterpart_is_absorbed_without_reverse_missing_report():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        p2p_timeout_s=5,
        p2p_match_window_s=2,
    )
    _ingest(
        analyzer,
        _event("nccl_call", api="ncclSend", comm_rank=0, peer=1, call_seq=1),
    )

    first_findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)
    assert [finding.hang_type for finding in first_findings] == ["p2p_missing_counterpart"]

    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=1,
            pid=11,
            comm_rank=1,
            peer=0,
            call_seq=2,
        ),
        observed=11,
        unix_ns=11_000_000_000,
    )

    assert analyzer.scan(now_monotonic_s=20, now_unix_ns=20_000_000_000) == []


def test_reported_missing_p2p_call_is_cleaned_after_grace_period():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    _ingest(
        analyzer,
        _event("nccl_call", api="ncclSend", comm_rank=0, peer=1, call_seq=1),
    )

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)
    assert [finding.hang_type for finding in findings] == ["p2p_missing_counterpart"]
    assert analyzer._p2p_calls

    assert analyzer.scan(now_monotonic_s=14, now_unix_ns=14_000_000_000) == []
    assert analyzer._p2p_calls

    assert analyzer.scan(now_monotonic_s=15, now_unix_ns=15_000_000_000) == []
    assert analyzer._p2p_calls == {}
    assert analyzer._reported_missing_p2p_until == {}


def test_multiple_p2p_candidates_are_reported_as_ambiguous():
    analyzer = TraceAnalyzer(run_id=RUN_ID, p2p_timeout_s=5, p2p_match_window_s=2)
    for call_seq in (1, 2):
        _ingest(
            analyzer,
            _event(
                "nccl_call",
                api="ncclSend",
                comm_rank=0,
                peer=1,
                call_seq=call_seq,
            ),
        )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=1,
            pid=11,
            comm_rank=1,
            peer=0,
            call_seq=3,
        ),
    )

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)

    assert [finding.hang_type for finding in findings] == ["ambiguous_p2p_match"]
    assert findings[0].details["confidence"] == "unknown"


def test_p2p_operation_index_resolves_completed_pair_before_missing_next_call():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        p2p_timeout_s=5,
        p2p_match_window_s=2,
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclSend",
            comm_rank=0,
            peer=1,
            call_seq=1,
            p2p_op_index=0,
        ),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclSend",
            comm_rank=0,
            peer=1,
            call_seq=2,
            p2p_op_index=1,
        ),
    )
    _ingest(
        analyzer,
        _event(
            "nccl_call",
            api="ncclRecv",
            rank=1,
            pid=11,
            comm_rank=1,
            peer=0,
            call_seq=3,
            p2p_op_index=0,
        ),
    )

    findings = analyzer.scan(now_monotonic_s=10, now_unix_ns=10_000_000_000)

    assert [finding.hang_type for finding in findings] == ["p2p_missing_counterpart"]
    assert findings[0].details["local_p2p_op_index"] == 1


def test_missing_enter_is_downgraded_when_probe_dropped_events():
    analyzer = TraceAnalyzer(
        run_id=RUN_ID,
        heartbeat_timeout_s=10,
        collective_timeout_s=2,
    )
    _ingest(analyzer, _event("comm_init", rank=1, comm_rank=1))
    _ingest(
        analyzer,
        _event("heartbeat", rank=1),
        observed=1,
    )
    _ingest(analyzer, _event("probe_status", rank=1, dropped_events=4), observed=1)
    _ingest(analyzer, _event("nccl_call", rank=0, comm_rank=0), observed=1)

    findings = analyzer.scan(now_monotonic_s=5, now_unix_ns=5_000_000_000)

    assert findings[0].details["reason"] == "probe_event_loss_possible"
    assert findings[0].details["confidence"] == "suspected"
