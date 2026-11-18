# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3

from .test_profiler_bridge import event, record
from flagscale.runner.tracing.inspector import InspectorReader


def _append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def test_unchanged_poll_does_not_parse_history_and_late_call_is_queryable(tmp_path):
    path = tmp_path / "bridge.jsonl"
    for seq in range(1, 401):
        _append(path, record(call_seq=seq, task_id=seq))
    reader = InspectorReader(tmp_path, enabled=True, cache_kib=64)
    index = reader.poll()
    assert reader.parsed_records == 400
    reader.poll()
    assert reader.parsed_records == 400
    _append(path, record(call_seq=401, task_id=401))
    index = reader.poll()
    assert reader.parsed_records == 401
    assert index.for_call(event(call_seq=1))["kernel_execution_status"] == "completed"
    assert index.for_call(event(call_seq=401))["kernel_execution_status"] == "completed"


def test_partial_tail_and_read_budget_never_claim_complete_early(tmp_path):
    path = tmp_path / "bridge.jsonl"
    _append(path, record())
    with path.open("a") as f:
        f.write('{"record_type":')
    reader = InspectorReader(tmp_path, enabled=True)
    index = reader.poll()
    assert not index.caught_up
    assert index.for_call(event())["kernel_execution_status"] == "unknown"
    with path.open("a") as f:
        f.write('"other"}\n')
    assert reader.poll().for_call(event())["kernel_execution_status"] == "completed"


def test_replaced_file_preserves_conflicting_revision(tmp_path):
    path = tmp_path / "bridge.jsonl"
    _append(path, record())
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(event())["kernel_execution_status"] == "completed"
    replacement = tmp_path / "replacement"
    changed = record()
    changed["channels"][0]["end"] += 1
    _append(replacement, changed)
    replacement.replace(path)
    result = reader.poll().for_call(event())
    assert result["kernel_execution_status"] == "unknown"
    assert result["inspector_correlation_reason"] == "conflicting_evidence_revisions"


def test_malformed_record_and_index_failure_are_not_silently_ignored(tmp_path, monkeypatch):
    path = tmp_path / "bridge.jsonl"
    _append(path, record())
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(event())["kernel_execution_status"] == "completed"
    with path.open("a") as f:
        f.write("broken\n")
    assert reader.poll().for_call(event())["inspector_correlation_reason"] == "inspector_read_error"
    _append(path, record(revision=2))

    def full(_payload):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(reader._store, "add", full)
    assert reader.poll().for_call(event())["kernel_execution_status"] == "unknown"
    assert any("disk is full" in e["error"] for e in reader.diagnostics["errors"])


def test_small_poll_budget_drains_without_losing_events(tmp_path):
    path = tmp_path / "bridge.jsonl"
    for seq in range(1, 21):
        _append(path, record(call_seq=seq, task_id=seq))
    reader = InspectorReader(tmp_path, enabled=True, poll_bytes=100)
    index = reader.poll()
    assert not index.caught_up
    for _ in range(21):
        index = reader.poll()
        if index.caught_up:
            break
    assert index.caught_up and reader.parsed_records == 20
    assert index.for_call(event(call_seq=1))["kernel_execution_status"] == "completed"


def test_snapshot_rewrite_and_restart_rebuild(tmp_path):
    path = tmp_path / "bridge.json"
    path.write_text(json.dumps(record(), indent=2))
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(event())["kernel_execution_status"] == "completed"
    previous = reader.parsed_records
    reader.poll()
    assert reader.parsed_records == previous
    path.write_text(json.dumps(record(call_seq=99), indent=2))
    assert reader.poll().for_call(event(call_seq=99))["kernel_execution_status"] == "completed"
    restarted = InspectorReader(tmp_path, enabled=True)
    assert restarted.poll().for_call(event(call_seq=99))["kernel_execution_status"] == "completed"


def test_lookup_budget_fails_closed_without_deleting_history(tmp_path):
    _append(tmp_path / "bridge.jsonl", record())
    reader = InspectorReader(tmp_path, enabled=True)
    index = reader.poll()
    reader._store.lookup_bytes = 10
    assert (
        index.for_call(event())["inspector_correlation_reason"]
        == "inspector_lookup_budget_exceeded"
    )
    reader._store.lookup_bytes = 8 * 1024 * 1024
    assert index.for_call(event())["kernel_execution_status"] == "completed"


def test_unavailable_local_index_does_not_kill_the_monitor(tmp_path, monkeypatch):
    from flagscale.runner.tracing import inspector_store

    def denied(**_kwargs):
        raise PermissionError("local cache directory is read-only")

    monkeypatch.setattr(inspector_store, "InspectorStore", denied)
    reader = InspectorReader(tmp_path, enabled=True)
    result = reader.poll().for_call(event())
    assert result["kernel_execution_status"] == "unknown"
    assert reader.diagnostics["errors"][0]["stage"] == "index_open"
