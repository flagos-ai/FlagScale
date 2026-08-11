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

from flagscale.runner.heartbeat.health_reader import HardwareHealthIndex
from flagscale.runner.heartbeat.monitor import HeartbeatAnalyzer


def _event(event, **overrides):
    payload = {
        "event": event,
        "run_id": "run",
        "rank": 1,
        "local_rank": 1,
        "world_size": 2,
        "pid": 42,
        "hostname": "host-a",
        "cuda_visible_devices": "0,1",
        "assigned_gpu": {"visible_ordinal": 1, "device_token": "1"},
        "progress_seq": 0,
        "iteration": None,
        "phase": "setup",
    }
    payload.update(overrides)
    return payload


def _analyzer(**overrides):
    values = {
        "run_id": "run",
        "initial_process_timeout_s": 2,
        "process_timeout_s": 3,
        "initial_progress_timeout_s": 10,
        "progress_timeout_s": 3,
        "checkpoint_timeout_s": 20,
    }
    values.update(overrides)
    return HeartbeatAnalyzer(**values)


def test_advancing_training_hook_is_exposed_as_gpu_progress():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=7, phase="train"),
        observed_s=1.5,
    )

    assert analyzer.scan(now_s=2.0, now_unix_ns=2_000_000_000) == []
    snapshot = analyzer.snapshot(now_s=2.0, now_unix_ns=2_000_000_000)
    rank = snapshot["ranks"][0]
    assert snapshot["health_scope"] == "gpu_training_progress_and_optional_hardware"
    assert snapshot["gpu_device_health"] == "not_collected"
    assert rank["process_liveness"] == "alive"
    assert rank["gpu_progress"] == "progressing"
    assert rank["iteration"] == 7
    assert rank["assigned_gpu"] == {"visible_ordinal": 1, "device_token": "1"}


def test_progress_timeout_includes_correlated_hardware_evidence():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=7, phase="train"),
        observed_s=1.5,
    )
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=7, phase="train"),
        observed_s=5.0,
    )
    hardware = HardwareHealthIndex(
        enabled=True,
        now_unix_ns=5_000_000_000,
        stale_after_s=30,
        snapshots=(
            {
                "hostname": "host-a",
                "collected_at_unix_ns": 4_000_000_000,
                "status": "unhealthy",
                "source": "nvidia-smi_nvml",
                "gpus": [{"index": 1, "uuid": "GPU-1", "status": "unhealthy"}],
            },
        ),
    )

    finding = analyzer.scan(5.0, 5_000_000_000, hardware)[0]

    assert finding["gpu_device_health"] == "unhealthy"
    assert finding["gpu_hardware"]["uuid"] == "GPU-1"


def test_live_process_with_unchanged_progress_reports_gpu_stall_once():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=7, phase="train"),
        observed_s=1.5,
    )
    # The CPU publisher remains alive, but the training hook sequence does not advance.
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=7, phase="train"),
        observed_s=5.0,
    )

    findings = analyzer.scan(now_s=5.0, now_unix_ns=5_000_000_000)
    assert [finding["finding_type"] for finding in findings] == ["gpu_progress_timeout"]
    assert findings[0]["reason"] == "rank_alive_but_gpu_progress_not_advancing"
    assert findings[0]["process_liveness"] == "alive"
    assert findings[0]["gpu_device_health"] == "not_collected"
    assert analyzer.scan(now_s=6.0, now_unix_ns=6_000_000_000) == []


def test_process_heartbeat_timeout_is_reported_again_after_recovery():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=1, phase="train"),
        observed_s=1.5,
    )

    first_findings = analyzer.scan(now_s=5.0, now_unix_ns=5_000_000_000)
    assert [finding["finding_type"] for finding in first_findings] == [
        "rank_process_heartbeat_timeout"
    ]
    assert first_findings[0]["pid"] == 42
    assert first_findings[0]["timeout_type"] == "subsequent_process_heartbeat"

    analyzer.ingest(
        _event("heartbeat", progress_seq=2, iteration=2, phase="train"),
        observed_s=5.5,
    )
    assert analyzer.scan(now_s=6.0, now_unix_ns=6_000_000_000) == []

    second_findings = analyzer.scan(now_s=9.0, now_unix_ns=9_000_000_000)
    assert [finding["finding_type"] for finding in second_findings] == [
        "rank_process_heartbeat_timeout"
    ]
    assert second_findings[0]["pid"] == 42
    assert second_findings[0]["timeout_type"] == "subsequent_process_heartbeat"


def test_gpu_progress_timeout_is_reported_again_after_recovery():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=1, phase="train"),
        observed_s=1.5,
    )
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=1, phase="train"),
        observed_s=5.0,
    )

    first_findings = analyzer.scan(now_s=5.0, now_unix_ns=5_000_000_000)
    assert [finding["finding_type"] for finding in first_findings] == ["gpu_progress_timeout"]
    assert first_findings[0]["pid"] == 42
    assert first_findings[0]["timeout_type"] == "subsequent_gpu_progress"

    analyzer.ingest(
        _event("heartbeat", progress_seq=2, iteration=2, phase="train"),
        observed_s=5.5,
    )
    assert analyzer.scan(now_s=6.0, now_unix_ns=6_000_000_000) == []
    analyzer.ingest(
        _event("heartbeat", progress_seq=2, iteration=2, phase="train"),
        observed_s=9.0,
    )

    second_findings = analyzer.scan(now_s=9.0, now_unix_ns=9_000_000_000)
    assert [finding["finding_type"] for finding in second_findings] == ["gpu_progress_timeout"]
    assert second_findings[0]["pid"] == 42
    assert second_findings[0]["timeout_type"] == "subsequent_gpu_progress"


def test_stale_process_suppresses_secondary_gpu_progress_finding():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("heartbeat", progress_seq=1, iteration=1, phase="train"),
        observed_s=1.5,
    )

    findings = analyzer.scan(now_s=5.0, now_unix_ns=5_000_000_000)
    assert [finding["finding_type"] for finding in findings] == ["rank_process_heartbeat_timeout"]


def test_normal_process_end_suppresses_teardown_timeouts():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(
        _event("process_end", progress_seq=1, iteration=1, phase="train"),
        observed_s=1.5,
    )

    assert analyzer.scan(now_s=10.0, now_unix_ns=10_000_000_000) == []
    rank = analyzer.snapshot(10.0, 10_000_000_000)["ranks"][0]
    assert rank["process_liveness"] == "exited"
    assert rank["gpu_progress"] == "completed"


def test_checkpoint_phase_uses_longer_progress_timeout():
    analyzer = _analyzer()
    analyzer.ingest(_event("process_start"), observed_s=1.0)
    analyzer.ingest(_event("heartbeat", progress_seq=1, phase="checkpointing"), observed_s=1.5)
    analyzer.ingest(_event("heartbeat", progress_seq=1, phase="checkpointing"), observed_s=10.0)

    assert analyzer.scan(now_s=10.0, now_unix_ns=10_000_000_000) == []


def test_rank_that_never_started_is_detected_from_expected_world_size():
    analyzer = _analyzer(expected_world_size=2, monitor_started_s=1)
    analyzer.ingest(_event("heartbeat", rank=0, local_rank=0), observed_s=1.5)

    findings = analyzer.scan(now_s=4, now_unix_ns=4_000_000_000)
    assert [finding["rank"] for finding in findings] == [1]
    assert findings[0]["reason"] == "rank_process_heartbeat_never_observed"
