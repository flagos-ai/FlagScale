# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import errno
import json
from pathlib import Path

from flagscale.runner.heartbeat.health_reader import HardwareHealthReader
from flagscale.runner.heartbeat.monitor import _append_health_read_events


def _write(path, timestamp=1_000_000_000):
    path.write_text(
        json.dumps(
            {
                "run_id": "run",
                "node_rank": 0,
                "collected_at_unix_ns": timestamp,
                "status": "healthy",
                "hostname": "host-a",
                "gpus": [],
            }
        )
    )


def test_transient_failure_remains_in_journal_after_recovery(tmp_path):
    path = tmp_path / "gpu_health_node_0.json"
    journal = tmp_path / "health_read_events.jsonl"
    _write(path)
    reader = HardwareHealthReader(tmp_path, "run", True, 30, 1)
    assert reader.poll(1_000_000_000)[0].read_events == ()
    path.unlink()
    for now in [2_000_000_000, 3_000_000_000]:
        index, findings = reader.poll(now)
        assert not findings
        _append_health_read_events(journal, index.read_events)
    _write(path, 4_000_000_000)
    index, findings = reader.poll(5_000_000_000)
    assert not findings and not index.read_errors
    _append_health_read_events(journal, index.read_events)
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["transition"] for event in events] == ["first_failure", "read_recovered"]
    recovered = events[-1]
    assert recovered["failure_count"] == 2
    assert recovered["first_failure_unix_ns"] == 2_000_000_000
    assert recovered["last_failure_unix_ns"] == 3_000_000_000
    assert recovered["recovered_at_unix_ns"] == 5_000_000_000
    assert recovered["first_error"]["errno"] == errno.ENOENT
    assert recovered["last_error"]["path"] == str(path)
    assert recovered["observer_hostname"] and recovered["observer_pid"] > 0
    assert reader._read_episodes == {}


def test_repeated_failure_is_throttled_and_does_not_extend_sample_age(tmp_path):
    path = tmp_path / "gpu_health_node_0.json"
    _write(path)
    reader = HardwareHealthReader(tmp_path, "run", True, 30, 1)
    reader.poll(1_000_000_000)
    path.unlink()
    assert reader.poll(2_000_000_000)[0].read_events[0]["transition"] == "first_failure"
    for second in range(3, 32):
        assert reader.poll(second * 1_000_000_000)[0].read_events == ()
    index, _ = reader.poll(32_000_000_000)
    event = index.read_events[0]
    assert event["transition"] == "ongoing"
    assert event["failure_count"] == 31
    assert event["last_valid_sample_unix_ns"] == 1_000_000_000
    assert event["snapshot_status"] == "stale"


def test_error_kind_changes_and_read_recovery_does_not_mean_gpu_healthy(tmp_path):
    path = tmp_path / "gpu_health_node_0.json"
    reader = HardwareHealthReader(tmp_path, "run", True, 30, 1)
    first = reader.poll(1_000_000_000)[0].read_events[0]
    assert first["last_error"]["stage"] == "read"
    path.write_text("{")
    change = reader.poll(2_000_000_000)[0].read_events[0]
    assert change["transition"] == "error_changed"
    assert change["last_error"]["stage"] == "parse"
    _write(path)
    event = reader.poll(50_000_000_000)[0].read_events[0]
    assert event["transition"] == "read_recovered"
    assert event["snapshot_status"] == "stale"
    path.unlink()
    event = reader.poll(51_000_000_000)[0].read_events[0]
    assert event["failure_count"] == 1


def test_discovery_omission_is_not_mistaken_for_recovery(tmp_path, monkeypatch):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text("{")
    reader = HardwareHealthReader(tmp_path, "run", True, 30)
    reader.poll(1_000_000_000)
    path.unlink()
    monkeypatch.setattr(Path, "glob", lambda *_args: iter(()))
    index, _ = reader.poll(2_000_000_000)
    assert index.read_errors[0]["errno"] == errno.ENOENT
    assert all(event["transition"] != "read_recovered" for event in index.read_events)


def test_journal_failure_preserves_event_in_monitor_log(tmp_path, caplog):
    _append_health_read_events(
        tmp_path / "missing" / "events.jsonl", ({"transition": "first_failure"},)
    )
    assert "Could not persist health read events" in caplog.text
    assert "first_failure" in caplog.text
