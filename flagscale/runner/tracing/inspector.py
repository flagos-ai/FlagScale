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

from .profiler_bridge import bridge_evidence

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


def _communicator_id(value: Any) -> str:
    try:
        parsed = int(str(value), 16)
    except (TypeError, ValueError):
        return ""
    return hex(parsed) if 0 <= parsed < 1 << 64 else ""


def _communicator_matches(completion: InspectorCompletion, event_id: str) -> bool:
    if completion.communicator_id == event_id:
        return True
    # Inspector v4.0/v4.1 uses a 17-byte commHashStr buffer for "0x" plus a
    # 64-bit hash. Long IDs therefore retain only their first 14 hex digits.
    # Accept this documented encoding only at its exact maximum length.
    return (
        completion.output_format_version in {"v4.0", "v4.1"}
        and len(event_id) > 16
        and completion.communicator_id == event_id[:16]
    )


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
    communicator_id: str = ""
    communicator_size: int = -1
    sequence_number: int = -1

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
            communicator_id=_communicator_id(header.get("id")),
            communicator_size=_as_int(header.get("n_ranks")),
            sequence_number=_as_int(performance.get("coll_sn" if is_collective else "p2p_sn")),
        )

    @property
    def kernel_completed(self) -> bool:
        return (
            self.timing_source == "kernel_gpu"
            and bool(self.kernel_events)
            and all(
                _as_int(event.get("kernel_start_ts")) > 0
                and _as_int(event.get("kernel_stop_ts")) >= _as_int(event.get("kernel_start_ts"))
                for event in self.kernel_events
            )
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
            "communicator_id": self.communicator_id,
            "communicator_size": self.communicator_size,
            "sequence_number": self.sequence_number,
        }


@dataclass(frozen=True)
class InspectorIndex:
    enabled: bool
    completions: tuple[InspectorCompletion, ...] = ()
    correlation_window_s: float = 5.0
    bridge_records: tuple[dict, ...] = ()

    def for_call(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {
                "inspector_correlation": "not_collected",
                "inspector_kernel_status": "not_collected",
                "kernel_execution_status": "unknown",
            }

        if event.get("profiler_bridge") is True:
            return bridge_evidence(event, self.bridge_records)

        # A group can reorder tasks before the profiler assigns its sequence.
        # Do not infer an API identity from that task order, even if signatures
        # happen to match. P2P likewise needs an explicit API-to-task relation.
        reason = None
        if _as_int(event.get("group_id"), 0) > 0:
            reason = "group_api_task_mapping_unavailable"
        elif "comm_inspector_id" in event and _normalize_api(event.get("api")) in {
            "send",
            "recv",
        }:
            reason = "p2p_api_task_mapping_unavailable"
        elif "comm_inspector_id" in event and _as_int(event.get("comm_api_seq")) < 0:
            reason = "profiler_sequence_unavailable"
        if reason:
            return {
                "inspector_correlation": "none",
                "inspector_candidate_count": 0,
                "inspector_correlation_reason": reason,
                "inspector_kernel_status": "unknown",
                "kernel_execution_status": "unknown",
            }

        event_size = _message_size_bytes(event)
        event_timestamp_us = _as_int(event.get("timestamp_unix_ns")) // 1000
        strict_identity = "comm_inspector_id" in event
        event_id = _communicator_id(event.get("comm_inspector_id"))
        event_sequence = _as_int(event.get("comm_api_seq"))
        event_nranks = _as_int(event.get("comm_nranks"))
        candidates = [
            completion
            for completion in self.completions
            if completion.hostname == str(event.get("hostname") or "")
            and completion.pid == _as_int(event.get("pid"))
            and completion.comm_rank == _as_int(event.get("comm_rank"))
            and _normalize_api(completion.api) == _normalize_api(event.get("api"))
            and (event_size is None or completion.message_size_bytes == event_size)
            and (event_nranks <= 0 or completion.communicator_size == event_nranks)
            and (
                (
                    strict_identity
                    and bool(event_id)
                    and event_sequence >= 0
                    and event_nranks > 0
                    and _normalize_api(event.get("api")) not in {"send", "recv"}
                    and _communicator_matches(completion, event_id)
                    and completion.sequence_number == event_sequence
                )
                or (
                    not strict_identity
                    and abs(completion.start_timestamp_us - event_timestamp_us)
                    <= self.correlation_window_s * 1_000_000
                )
            )
        ]
        if len(candidates) != 1:
            return {
                "inspector_correlation": "none" if not candidates else "ambiguous",
                "inspector_candidate_count": len(candidates),
                "inspector_kernel_status": "unknown",
                "kernel_execution_status": "unknown",
            }
        completion = candidates[0]
        kernel_status = "completed" if completion.kernel_completed else "unknown"
        return {
            "inspector_correlation": "unique",
            "inspector_candidate_count": 1,
            "inspector_call_sequence": event_sequence if strict_identity else None,
            "inspector_kernel_status": kernel_status,
            "kernel_execution_status": kernel_status,
            "inspector_match_method": (
                "communicator_sequence" if strict_identity else "signature_time_window"
            ),
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
        self._seen: set[str] = set()
        self._bridge_records: dict[str, dict] = {}
        self._revisions: dict[tuple[Any, ...], list[InspectorCompletion]] = {}

    def poll(self) -> InspectorIndex:
        if not self.enabled:
            return InspectorIndex(False)
        paths = set(self.directory.rglob("*.json*")) | set(self.directory.rglob("*.log"))
        for path in sorted(paths):
            for payload in _read_payloads(path):
                if payload.get("record_type") in {
                    "flagscale_profiler_v5_v1",
                    "flagscale_bridge_error_v1",
                }:
                    self._bridge_records[json.dumps(payload, sort_keys=True)] = payload
                    continue
                completion = InspectorCompletion.from_payload(payload)
                if completion is None:
                    continue
                fingerprint = json.dumps(completion.summary(), sort_keys=True)
                if fingerprint in self._seen:
                    continue
                self._seen.add(fingerprint)
                # Retain conflicting revisions as separate candidates. Only
                # replace evidence when the new record is a compatible refinement.
                identity = _completion_identity(completion)
                revisions = self._revisions.setdefault(identity, [])
                if any(_refines(old, completion) for old in revisions):
                    continue
                self._revisions[identity] = [
                    old for old in revisions if not _refines(completion, old)
                ] + [completion]
        return InspectorIndex(
            True,
            tuple(record for revisions in self._revisions.values() for record in revisions),
            self.correlation_window_s,
            tuple(self._bridge_records.values()),
        )


def _completion_identity(record: InspectorCompletion) -> tuple[Any, ...]:
    fields = (
        "hostname",
        "pid",
        "comm_rank",
        "communicator_id",
        "communicator_size",
        "sequence_number",
        "api",
        "message_size_bytes",
        "start_timestamp_us",
        "stop_timestamp_us",
        "output_format_version",
    )
    return tuple(getattr(record, field) for field in fields)


def _refines(new: InspectorCompletion, old: InspectorCompletion) -> bool:
    """Accept added GPU evidence only when the operation identity is unchanged."""
    if _completion_identity(new) != _completion_identity(old):
        return False
    if new == old:
        return True
    if not new.kernel_completed:
        return False
    if not old.kernel_events:
        return old.timing_source != "kernel_gpu"
    if old.timing_source != new.timing_source:
        return False

    # Channel identities must be present and unique before comparing revisions.
    def channels(record):
        result = {}
        for event in record.kernel_events:
            channel = event.get("channel_id")
            if channel is None or channel in result:
                return None
            result[channel] = event
        return result

    old_channels, new_channels = channels(old), channels(new)
    if old_channels is None or new_channels is None:
        return False
    for channel, previous in old_channels.items():
        current = new_channels.get(channel)
        if current is None:
            return False
        for key, value in previous.items():
            if key in {"kernel_start_ts", "kernel_stop_ts"} and _as_int(value) <= 0:
                continue
            if current.get(key) != value:
                return False
    # Conflicting complete records, including differing durations, stay ambiguous.
    return not old.kernel_completed


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
