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
                "inspector_correlation_reason": "inspector_disabled",
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
                "inspector_correlation_reason": (
                    "device_record_missing" if not candidates else "multiple_device_records"
                ),
                "inspector_kernel_status": "unknown",
                "kernel_execution_status": "unknown",
            }
        completion = candidates[0]
        kernel_status = "completed" if completion.kernel_completed else "unknown"
        return {
            "inspector_correlation": "unique",
            "inspector_correlation_reason": (
                "device_channels_complete"
                if completion.kernel_completed
                else "device_completion_evidence_incomplete"
            ),
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
    """Incremental input with bounded buffers and a private disk-backed index."""

    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool,
        correlation_window_s: float = 5.0,
        poll_bytes: int = 16 * 1024 * 1024,
        record_bytes: int = 8 * 1024 * 1024,
        cache_kib: int = 2048,
    ) -> None:
        import sqlite3

        from .inspector_store import InspectorStore

        if poll_bytes <= 0 or record_bytes <= 0:
            raise ValueError("Inspector read budgets must be positive")
        self.directory, self.enabled = directory, enabled
        self.correlation_window_s = correlation_window_s
        self.poll_bytes, self.record_bytes = poll_bytes, record_bytes
        self._store = None
        self._files = {}
        self._errors = {}
        self._fatal = {}
        self._rotation = 0
        self.parsed_records = 0
        self.read_bytes = 0
        self.caught_up = False
        if enabled:
            try:
                self._store = InspectorStore(cache_kib=cache_kib)
            except (OSError, sqlite3.Error) as exc:
                self._fatal["index"] = {
                    "stage": "index_open",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }

    @property
    def diagnostics(self):
        return {
            "caught_up": self.caught_up,
            "parsed_records": self.parsed_records,
            "read_bytes": self.read_bytes,
            "files": len(self._files),
            "errors": [*self._errors.values(), *self._fatal.values()],
        }

    def poll(self):
        import sqlite3

        from .inspector_store import StoredInspectorIndex

        if not self.enabled:
            return InspectorIndex(False)
        if self._store is None:
            return StoredInspectorIndex(
                None, self.correlation_window_s, list(self._fatal.values()), False
            )
        # Include known paths even if a directory listing temporarily omits them.
        paths = set(self._files)
        try:
            for pattern in ("*.json*", "*.log"):
                for path in self.directory.rglob(pattern):
                    paths.add(path)
                    if len(paths) > 8192:
                        raise ValueError("inspector_file_budget_exceeded")
            self._errors.pop("discovery", None)
        except (OSError, ValueError) as exc:
            self._errors["discovery"] = {"stage": "discovery", "error": str(exc)}
            paths = set(self._files)
        paths = sorted(paths)
        if paths:
            rotation = self._rotation % len(paths)
            paths = paths[rotation:] + paths[:rotation]
            self._rotation += 1
        remaining = self.poll_bytes
        caught_up = True
        for path in paths:
            if remaining <= 0:
                caught_up = False
                break
            state = self._files.setdefault(path, {"offset": 0})
            try:
                with path.open("rb") as stream:
                    import os

                    stat = os.fstat(stream.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    prefix = stream.read(min(128, state.get("size", stat.st_size)))
                    rewritten = (
                        state.get("identity", identity) != identity
                        or stat.st_size < state["offset"]
                        or (
                            state.get("signature") != signature
                            and stat.st_size <= state.get("size", -1)
                        )
                        or (
                            state.get("prefix") is not None
                            and not prefix.startswith(state["prefix"])
                        )
                    )
                    if rewritten:
                        state["offset"] = 0
                    state.update(
                        identity=identity, signature=signature, size=stat.st_size, prefix=prefix
                    )
                    if path.suffix == ".json":
                        if state.get("loaded_signature") != signature:
                            if stat.st_size > self.record_bytes:
                                raise ValueError("inspector_snapshot_budget_exceeded")
                            stream.seek(0)
                            data = stream.read(self.record_bytes + 1)
                            remaining -= len(data)
                            self.read_bytes += len(data)
                            payload = json.loads(data)
                            if isinstance(payload, dict):
                                self._store.add(payload)
                                self.parsed_records += 1
                            state["loaded_signature"] = signature
                            state["offset"] = len(data)
                    else:
                        stream.seek(state["offset"])
                        while remaining > 0:
                            data = stream.readline(self.record_bytes + 1)
                            if not data:
                                break
                            remaining -= len(data)
                            self.read_bytes += len(data)
                            if len(data) > self.record_bytes:
                                raise ValueError("inspector_record_budget_exceeded")
                            if not data.endswith(b"\n"):
                                # Leave offset at the last complete line; no per-file
                                # in-memory remainder that can grow across polls.
                                caught_up = False
                                break
                            try:
                                payload = json.loads(data)
                            except (ValueError, UnicodeError) as exc:
                                self._fatal[str(path)] = {
                                    "path": str(path),
                                    "stage": "parse",
                                    "offset": state["offset"],
                                    "error": str(exc),
                                }
                            else:
                                if isinstance(payload, dict):
                                    self._store.add(payload)
                                    self.parsed_records += 1
                            state["offset"] = stream.tell()
                    caught_up &= state["offset"] >= stat.st_size
                self._errors.pop(str(path), None)
            except (OSError, ValueError, sqlite3.Error, OverflowError) as exc:
                self._errors[str(path)] = {
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "errno": getattr(exc, "errno", None),
                }
                caught_up = False
                if isinstance(exc, (sqlite3.Error, OverflowError)):
                    self._fatal["index"] = self._errors[str(path)]
        try:
            self._store.db.commit()
        except sqlite3.Error as exc:
            self._fatal["index"] = {"stage": "commit", "error": str(exc)}
        self.caught_up = caught_up and not self._errors and not self._fatal
        return StoredInspectorIndex(
            self._store,
            self.correlation_window_s,
            [*self._errors.values(), *self._fatal.values()],
            self.caught_up,
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
