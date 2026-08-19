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

"""Read node-local GPU health snapshots and correlate them with worker ranks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

_STATUS_PRIORITY = {
    "not_collected": 0,
    "healthy": 1,
    "unavailable": 2,
    "stale": 3,
    "warning": 4,
    "unhealthy": 5,
}


def _worst_status(statuses: list[str], default: str) -> str:
    return max(statuses, key=lambda value: _STATUS_PRIORITY.get(value, -1), default=default)


@dataclass(frozen=True)
class HardwareHealthIndex:
    enabled: bool
    now_unix_ns: int
    stale_after_s: float
    snapshots: tuple[dict[str, Any], ...] = ()
    missing_node_ranks: tuple[int, ...] = ()

    @property
    def overall_status(self) -> str:
        if not self.enabled:
            return "not_collected"
        statuses = [self._snapshot_status(snapshot) for snapshot in self.snapshots]
        statuses.extend("unavailable" for _ in self.missing_node_ranks)
        return _worst_status(statuses, "unavailable")

    def _snapshot_age_s(self, snapshot: dict[str, Any]) -> float | None:
        try:
            collected_at = int(snapshot["collected_at_unix_ns"])
        except (KeyError, TypeError, ValueError):
            return None
        return max(0.0, (self.now_unix_ns - collected_at) / 1_000_000_000)

    def _snapshot_status(self, snapshot: dict[str, Any]) -> str:
        age_s = self._snapshot_age_s(snapshot)
        if age_s is None or age_s > self.stale_after_s:
            return "stale"
        return str(snapshot.get("status") or "unavailable")

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "not_collected", "nodes": []}
        nodes = []
        for snapshot in self.snapshots:
            nodes.append(
                {
                    "node_rank": snapshot.get("node_rank"),
                    "hostname": snapshot.get("hostname"),
                    "status": self._snapshot_status(snapshot),
                    "sample_age_s": self._snapshot_age_s(snapshot),
                    "source": snapshot.get("source"),
                    "error": snapshot.get("error"),
                    "gpus": snapshot.get("gpus", []),
                }
            )
        for node_rank in self.missing_node_ranks:
            nodes.append(
                {
                    "node_rank": node_rank,
                    "hostname": None,
                    "status": "unavailable",
                    "sample_age_s": None,
                    "source": None,
                    "error": "gpu_health_snapshot_not_generated",
                    "gpus": [],
                }
            )
        return {"enabled": True, "status": self.overall_status, "nodes": nodes}

    def for_rank(self, rank_event: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"gpu_device_health": "not_collected"}
        hostname = rank_event.get("hostname")
        snapshot = next((item for item in self.snapshots if item.get("hostname") == hostname), None)
        if snapshot is None:
            return {"gpu_device_health": "unavailable"}
        snapshot_status = self._snapshot_status(snapshot)
        if snapshot_status in {"stale", "unavailable"}:
            return {
                "gpu_device_health": snapshot_status,
                "gpu_health_sample_age_s": self._snapshot_age_s(snapshot),
                "gpu_health_source": snapshot.get("source"),
            }

        gpu = _match_rank_gpu(rank_event, snapshot.get("gpus", []))
        if gpu is None:
            return {
                "gpu_device_health": "unavailable",
                "gpu_health_sample_age_s": self._snapshot_age_s(snapshot),
                "gpu_health_source": snapshot.get("source"),
                "gpu_health_error": "could_not_map_rank_to_gpu",
            }
        return {
            "gpu_device_health": str(gpu.get("status") or "unavailable"),
            "gpu_health_sample_age_s": self._snapshot_age_s(snapshot),
            "gpu_health_source": snapshot.get("source"),
            "assigned_gpu": {
                **dict(rank_event.get("assigned_gpu") or {}),
                "index": gpu.get("index"),
                "uuid": gpu.get("uuid"),
            },
            "gpu_hardware": gpu,
        }


def _match_rank_gpu(
    rank_event: dict[str, Any], gpus: list[dict[str, Any]]
) -> dict[str, Any] | None:
    try:
        local_rank = int(rank_event.get("local_rank"))
    except (TypeError, ValueError):
        return None
    if local_rank < 0:
        return None
    assigned_gpu = rank_event.get("assigned_gpu")
    device_token = assigned_gpu.get("device_token") if isinstance(assigned_gpu, dict) else None
    if device_token is None:
        visible = str(rank_event.get("cuda_visible_devices") or "")
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        device_token = tokens[local_rank] if local_rank < len(tokens) else str(local_rank)
    device_token = str(device_token)
    if device_token.startswith("GPU-"):
        return next((gpu for gpu in gpus if gpu.get("uuid") == device_token), None)
    try:
        device_index = int(device_token)
    except ValueError:
        return None
    return next((gpu for gpu in gpus if gpu.get("index") == device_index), None)


class HardwareHealthReader:
    def __init__(
        self,
        heartbeat_dir: Path,
        run_id: str,
        enabled: bool,
        stale_after_s: float,
        expected_node_count: int = 0,
        monitor_started_unix_ns: int = 0,
    ) -> None:
        self.heartbeat_dir = heartbeat_dir
        self.run_id = run_id
        self.enabled = enabled
        self.stale_after_s = stale_after_s
        self.expected_node_count = expected_node_count
        self.monitor_started_unix_ns = monitor_started_unix_ns
        self._reported: set[tuple[str, str, tuple[str, ...]]] = set()
        self._reported_missing_nodes: set[int] = set()

    def poll(self, now_unix_ns: int) -> tuple[HardwareHealthIndex, list[dict[str, Any]]]:
        if not self.enabled:
            return HardwareHealthIndex(False, now_unix_ns, self.stale_after_s), []
        snapshots: list[dict[str, Any]] = []
        for path in sorted(self.heartbeat_dir.glob("gpu_health_node_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("run_id") == self.run_id:
                snapshots.append(payload)
        missing_node_ranks = self._missing_node_ranks(snapshots, now_unix_ns)
        index = HardwareHealthIndex(
            True,
            now_unix_ns,
            self.stale_after_s,
            tuple(snapshots),
            missing_node_ranks,
        )
        findings = self._new_findings(index, now_unix_ns)
        findings.extend(self._new_missing_findings(index, now_unix_ns))
        return index, findings

    def _missing_node_ranks(
        self, snapshots: list[dict[str, Any]], now_unix_ns: int
    ) -> tuple[int, ...]:
        if self.expected_node_count <= 0:
            return ()
        monitor_age_s = max(
            0.0,
            (now_unix_ns - self.monitor_started_unix_ns) / 1_000_000_000,
        )
        if monitor_age_s <= self.stale_after_s:
            return ()
        observed_node_ranks: set[int] = set()
        for snapshot in snapshots:
            try:
                node_rank = int(snapshot.get("node_rank"))
            except (TypeError, ValueError):
                continue
            if 0 <= node_rank < self.expected_node_count:
                observed_node_ranks.add(node_rank)
        return tuple(
            node_rank
            for node_rank in range(self.expected_node_count)
            if node_rank not in observed_node_ranks
        )

    def _new_missing_findings(
        self, index: HardwareHealthIndex, now_unix_ns: int
    ) -> list[dict[str, Any]]:
        missing_node_ranks = set(index.missing_node_ranks)
        self._reported_missing_nodes.intersection_update(missing_node_ranks)
        findings: list[dict[str, Any]] = []
        for node_rank in index.missing_node_ranks:
            if node_rank in self._reported_missing_nodes:
                continue
            self._reported_missing_nodes.add(node_rank)
            findings.append(
                {
                    "finding_type": "gpu_health_snapshot_missing",
                    "run_id": self.run_id,
                    "detected_at_unix_ns": now_unix_ns,
                    "node_rank": node_rank,
                    "hostname": None,
                    "gpu_device_health": "unavailable",
                    "expected_file": f"gpu_health_node_{node_rank}.json",
                    "reason": "gpu_health_snapshot_not_generated",
                    "confidence": "observed",
                }
            )
        return findings

    def _new_findings(self, index: HardwareHealthIndex, now_unix_ns: int) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for snapshot in index.snapshots:
            if index._snapshot_status(snapshot) in {"stale", "unavailable"}:
                continue
            for gpu in snapshot.get("gpus", []):
                status = str(gpu.get("status") or "unavailable")
                if status not in {"warning", "unhealthy"}:
                    continue
                reasons = tuple(
                    sorted(
                        str(issue.get("reason"))
                        for issue in gpu.get("issues", [])
                        if issue.get("reason")
                    )
                )
                key = (str(snapshot.get("hostname")), str(gpu.get("uuid")), reasons)
                if key in self._reported:
                    continue
                self._reported.add(key)
                findings.append(
                    {
                        "finding_type": (
                            "gpu_hardware_health_failure"
                            if status == "unhealthy"
                            else "gpu_hardware_health_warning"
                        ),
                        "run_id": self.run_id,
                        "detected_at_unix_ns": now_unix_ns,
                        "node_rank": snapshot.get("node_rank"),
                        "hostname": snapshot.get("hostname"),
                        "gpu_index": gpu.get("index"),
                        "gpu_uuid": gpu.get("uuid"),
                        "gpu_device_health": status,
                        "issues": gpu.get("issues", []),
                        "gpu_hardware": gpu,
                        "reason": "gpu_hardware_health_evidence_observed",
                        "confidence": "observed",
                    }
                )
        return findings
