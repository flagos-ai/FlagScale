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

"""Read completed-operation evidence emitted by NVIDIA NCCL Inspector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

_DATATYPE_SIZES = {
    0: 1,
    1: 1,
    2: 4,
    3: 4,
    4: 8,
    5: 8,
    6: 2,
    7: 4,
    8: 8,
    9: 2,
    10: 1,
    11: 1,
}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_api(value: Any) -> str:
    normalized = "".join(character for character in str(value or "").lower() if character.isalnum())
    return normalized[4:] if normalized.startswith("nccl") else normalized


def _message_size_bytes(event: dict[str, Any]) -> int | None:
    count = _as_int(event.get("count"))
    datatype = _as_int(event.get("datatype"))
    size = _DATATYPE_SIZES.get(datatype)
    if count < 0 or size is None:
        return None
    return count * size


@dataclass(frozen=True)
class InspectorCompletion:
    hostname: str
    pid: int
    comm_rank: int
    api: str
    message_size_bytes: int
    start_timestamp_us: int
    stop_timestamp_us: int
    execution_time_us: float | None
    timing_source: str
    output_format_version: str
    kernel_events: tuple[dict[str, Any], ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> InspectorCompletion | None:
        header = payload.get("header")
        metadata = payload.get("metadata")
        performance = payload.get("coll_perf") or payload.get("p2p_perf")
        if not all(isinstance(value, dict) for value in (header, metadata, performance)):
            return None

        is_collective = "coll_perf" in payload
        api = performance.get("coll" if is_collective else "p2p")
        if api is None and not is_collective:
            api = performance.get("operation")
        message_size = _as_int(
            performance.get("coll_msg_size_bytes" if is_collective else "p2p_msg_size_bytes")
        )
        if message_size < 0:
            message_size = _as_int(performance.get("message_size_bytes"))

        trace = performance.get("event_trace_ts")
        trace = trace if isinstance(trace, dict) else {}
        start_us = _as_int(
            trace.get("coll_start_ts", trace.get("p2p_start_ts")),
            _as_int(metadata.get("dump_timestamp_us")),
        )
        stop_us = _as_int(
            trace.get("coll_stop_ts", trace.get("p2p_stop_ts")),
            _as_int(metadata.get("dump_timestamp_us")),
        )
        kernel_events = trace.get("kernel_events")
        if not isinstance(kernel_events, list):
            kernel_events = []
        if (
            not api
            or message_size < 0
            or _as_int(header.get("rank")) < 0
            or _as_int(metadata.get("pid")) < 0
        ):
            return None
        execution_time = performance.get(
            "coll_exec_time_us" if is_collective else "p2p_exec_time_us"
        )
        try:
            parsed_execution_time = float(execution_time)
        except (TypeError, ValueError):
            parsed_execution_time = None
        return cls(
            hostname=str(metadata.get("hostname") or ""),
            pid=_as_int(metadata.get("pid")),
            comm_rank=_as_int(header.get("rank")),
            api=str(api),
            message_size_bytes=message_size,
            start_timestamp_us=start_us,
            stop_timestamp_us=stop_us,
            execution_time_us=parsed_execution_time,
            timing_source=str(
                performance.get("coll_timing_source" if is_collective else "p2p_timing_source")
                or ""
            ),
            output_format_version=str(metadata.get("inspector_output_format_version") or ""),
            kernel_events=tuple(dict(event) for event in kernel_events if isinstance(event, dict)),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "pid": self.pid,
            "comm_rank": self.comm_rank,
            "api": self.api,
            "message_size_bytes": self.message_size_bytes,
            "start_timestamp_us": self.start_timestamp_us,
            "stop_timestamp_us": self.stop_timestamp_us,
            "execution_time_us": self.execution_time_us,
            "timing_source": self.timing_source,
            "output_format_version": self.output_format_version,
            "kernel_events": list(self.kernel_events),
        }


@dataclass(frozen=True)
class InspectorIndex:
    enabled: bool
    completions: tuple[InspectorCompletion, ...] = ()
    correlation_window_s: float = 5.0

    def for_call(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {
                "inspector_correlation": "not_collected",
                "inspector_kernel_status": "not_collected",
                "kernel_execution_status": "unknown",
            }

        event_size = _message_size_bytes(event)
        event_timestamp_us = _as_int(event.get("timestamp_unix_ns")) // 1000
        candidates = [
            completion
            for completion in self.completions
            if completion.hostname == str(event.get("hostname") or "")
            and completion.pid == _as_int(event.get("pid"))
            and completion.comm_rank == _as_int(event.get("comm_rank"))
            and _normalize_api(completion.api) == _normalize_api(event.get("api"))
            and (event_size is None or completion.message_size_bytes == event_size)
            and abs(completion.start_timestamp_us - event_timestamp_us)
            <= self.correlation_window_s * 1_000_000
        ]
        if len(candidates) != 1:
            return {
                "inspector_correlation": "none" if not candidates else "ambiguous",
                "inspector_candidate_count": len(candidates),
                "inspector_kernel_status": "unknown",
                "kernel_execution_status": "unknown",
            }
        completion = candidates[0]
        return {
            "inspector_correlation": "unique",
            "inspector_candidate_count": 1,
            "inspector_kernel_status": "completed",
            "kernel_execution_status": "completed",
            "inspector_completion": completion.summary(),
        }


class InspectorReader:
    """Build a deduplicated index from Inspector JSON/JSONL output files."""

    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool,
        correlation_window_s: float = 5.0,
    ) -> None:
        self.directory = directory
        self.enabled = enabled
        self.correlation_window_s = correlation_window_s
        self._seen: set[tuple[Any, ...]] = set()
        self._completions: list[InspectorCompletion] = []

    def poll(self) -> InspectorIndex:
        if not self.enabled:
            return InspectorIndex(False)
        for path in sorted(self.directory.rglob("*.json*")):
            for payload in _read_payloads(path):
                completion = InspectorCompletion.from_payload(payload)
                if completion is None:
                    continue
                identity = (
                    completion.hostname,
                    completion.pid,
                    completion.comm_rank,
                    _normalize_api(completion.api),
                    completion.message_size_bytes,
                    completion.start_timestamp_us,
                    completion.stop_timestamp_us,
                )
                if identity in self._seen:
                    continue
                self._seen.add(identity)
                self._completions.append(completion)
        return InspectorIndex(
            True,
            tuple(self._completions),
            self.correlation_window_s,
        )


def _read_payloads(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    if payloads:
        return payloads
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [payload] if isinstance(payload, dict) else []
