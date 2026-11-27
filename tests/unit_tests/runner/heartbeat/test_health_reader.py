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
from pathlib import Path

import pytest

from flagscale.runner.heartbeat.health_reader import HardwareHealthReader


def _snapshot(status="warning", **overrides):
    payload = {
        "run_id": "run",
        "node_rank": 0,
        "hostname": "host-a",
        "collected_at_unix_ns": 9_000_000_000,
        "source": "nvidia-smi_nvml",
        "status": status,
        "gpus": [
            {"index": 0, "uuid": "GPU-0", "status": "healthy", "issues": []},
            {
                "index": 1,
                "uuid": "GPU-1",
                "status": status,
                "issues": [{"severity": status, "reason": "row_remap_pending"}],
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_reader_correlates_cuda_visible_devices_and_deduplicates_findings(tmp_path):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    reader = HardwareHealthReader(tmp_path, "run", True, stale_after_s=30)

    index, findings = reader.poll(now_unix_ns=10_000_000_000)
    rank = index.for_rank(
        {
            "hostname": "host-a",
            "local_rank": 1,
            "cuda_visible_devices": "0,1",
            "assigned_gpu": {"visible_ordinal": 1, "device_token": "1"},
        }
    )

    assert index.overall_status == "warning"
    assert rank["gpu_device_health"] == "warning"
    assert rank["gpu_hardware"]["uuid"] == "GPU-1"
    assert rank["assigned_gpu"] == {
        "visible_ordinal": 1,
        "device_token": "1",
        "index": 1,
        "uuid": "GPU-1",
    }
    assert [finding["finding_type"] for finding in findings] == ["gpu_hardware_health_warning"]
    assert reader.poll(now_unix_ns=11_000_000_000)[1] == []


def test_stale_snapshot_is_not_reported_as_hardware_failure(tmp_path):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text(json.dumps(_snapshot("unhealthy")), encoding="utf-8")
    reader = HardwareHealthReader(tmp_path, "run", True, stale_after_s=1)

    index, findings = reader.poll(now_unix_ns=20_000_000_000)

    assert index.overall_status == "stale"
    assert findings == []


def test_missing_node_snapshot_is_reported_after_startup_grace(tmp_path):
    node_zero = tmp_path / "gpu_health_node_0.json"
    node_zero.write_text(json.dumps(_snapshot("healthy")), encoding="utf-8")
    reader = HardwareHealthReader(
        tmp_path,
        "run",
        True,
        stale_after_s=30,
        expected_node_count=2,
        monitor_started_unix_ns=0,
    )

    initial_index, initial_findings = reader.poll(now_unix_ns=20_000_000_000)
    assert initial_index.overall_status == "healthy"
    assert initial_index.missing_node_ranks == ()
    assert initial_findings == []

    missing_index, missing_findings = reader.poll(now_unix_ns=31_000_000_000)
    assert missing_index.overall_status == "unavailable"
    assert missing_index.missing_node_ranks == (1,)
    missing_summary = missing_index.summary()["nodes"][1]
    assert missing_summary.pop("read_error")["error_type"] == "FileNotFoundError"
    assert missing_summary == {
        "node_rank": 1,
        "hostname": None,
        "status": "unavailable",
        "sample_age_s": None,
        "source": None,
        "error": "gpu_health_snapshot_not_observed",
        "gpus": [],
    }
    assert [finding["finding_type"] for finding in missing_findings] == [
        "gpu_health_snapshot_missing"
    ]
    assert missing_findings[0]["node_rank"] == 1
    assert missing_findings[0]["expected_file"] == "gpu_health_node_1.json"
    assert reader.poll(now_unix_ns=32_000_000_000)[1] == []

    node_one = tmp_path / "gpu_health_node_1.json"
    node_one.write_text(
        json.dumps(
            _snapshot(
                "healthy",
                node_rank=1,
                hostname="host-b",
                collected_at_unix_ns=32_000_000_000,
            )
        ),
        encoding="utf-8",
    )
    recovered_index, recovered_findings = reader.poll(now_unix_ns=33_000_000_000)
    assert recovered_index.overall_status == "healthy"
    assert recovered_index.missing_node_ranks == ()
    assert recovered_findings == []

    node_one.unlink()
    cached_index, repeated_findings = reader.poll(now_unix_ns=34_000_000_000)
    assert repeated_findings == []
    assert cached_index.missing_node_ranks == ()
    assert cached_index.summary()["nodes"][1]["using_cached_snapshot"] is True


@pytest.mark.parametrize("failure", ["missing", "invalid_json", "permission", "wrong_run"])
def test_cached_health_preserves_age_expires_and_recovers(tmp_path, monkeypatch, failure):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text(json.dumps(_snapshot("healthy")))
    reader = HardwareHealthReader(tmp_path, "run", True, 30, 1, 0)
    assert reader.poll(10_000_000_000)[0].overall_status == "healthy"
    original_read = Path.read_text
    if failure == "missing":
        path.unlink()
    elif failure == "invalid_json":
        path.write_text("{")
    elif failure == "wrong_run":
        path.write_text(json.dumps(_snapshot("healthy", run_id="other")))
    else:

        def denied(p, *args, **kwargs):
            if p == path:
                raise PermissionError("injected read failure")
            return original_read(p, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)

    index, findings = reader.poll(20_000_000_000)
    assert findings == []
    assert index.overall_status == "healthy"
    node = index.summary()["nodes"][0]
    assert node["using_cached_snapshot"] is True
    assert node["sample_age_s"] == 11
    assert index.snapshots[0]["collected_at_unix_ns"] == 9_000_000_000
    expired, findings = reader.poll(40_000_000_000)
    assert expired.overall_status == "stale"
    assert expired.for_rank({"hostname": "host-a"})["gpu_device_health"] == "stale"
    assert [f["finding_type"] for f in findings] == ["gpu_health_snapshot_read_failed"]
    assert findings[0]["read_error"] == expired.read_errors[0]
    assert reader.poll(41_000_000_000)[1] == []

    monkeypatch.setattr(Path, "read_text", original_read)
    path.write_text(json.dumps(_snapshot("healthy", collected_at_unix_ns=42_000_000_000)))
    recovered, findings = reader.poll(43_000_000_000)
    assert recovered.overall_status == "healthy"
    assert recovered.read_errors == ()
    assert "using_cached_snapshot" not in recovered.summary()["nodes"][0]
    assert findings == []
    path.unlink()
    assert len(reader.poll(73_000_000_000)[1]) == 1


def test_known_nodes_do_not_depend_on_directory_listing(tmp_path, monkeypatch):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text(json.dumps(_snapshot("healthy")))
    monkeypatch.setattr(Path, "glob", lambda *_args: iter(()))
    reader = HardwareHealthReader(tmp_path, "run", True, 30, 1, 0)
    assert reader.poll(10_000_000_000)[0].overall_status == "healthy"


def test_discovered_node_keeps_cached_snapshot_when_listing_omits_it(tmp_path, monkeypatch):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text(json.dumps(_snapshot("healthy")))
    reader = HardwareHealthReader(tmp_path, "run", True, 30)
    reader.poll(10_000_000_000)
    monkeypatch.setattr(Path, "glob", lambda *_args: iter(()))
    path.unlink()
    index, findings = reader.poll(40_000_000_000)
    assert index.overall_status == "stale"
    assert findings[0]["finding_type"] == "gpu_health_snapshot_read_failed"


def test_old_snapshot_does_not_refresh_cached_sample(tmp_path):
    path = tmp_path / "gpu_health_node_0.json"
    path.write_text(json.dumps(_snapshot("healthy", collected_at_unix_ns=20_000_000_000)))
    reader = HardwareHealthReader(tmp_path, "run", True, 30, 1, 0)
    reader.poll(21_000_000_000)
    path.write_text(json.dumps(_snapshot("healthy")))
    index, _ = reader.poll(51_000_000_000)
    assert index.snapshots[0]["collected_at_unix_ns"] == 20_000_000_000
    assert index.overall_status == "stale"
