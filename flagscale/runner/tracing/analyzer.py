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

"""In-memory analysis for CPU-side NCCL probe events.

The analyzer deliberately reports observable facts. It does not classify a missing
collective entry as a GPU or NCCL failure without independent evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

_COLLECTIVE_APIS = {
    "ncclAllReduce",
    "ncclAllGather",
    "ncclReduceScatter",
    "ncclBroadcast",
    "ncclReduce",
}


@dataclass(frozen=True)
class Finding:
    """A deduplicated hang or stall finding."""

    hang_type: str
    run_id: str
    comm_uid_hash: str
    comm_seq: int
    detected_at_unix_ns: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Heartbeat:
    event: dict[str, Any]
    effective_seen_monotonic_s: float


@dataclass
class _ProcessStart:
    event: dict[str, Any]
    effective_seen_monotonic_s: float


@dataclass
class _CollectiveRound:
    run_id: str
    comm_uid_hash: str
    comm_seq: int
    expected_nranks: int
    first_seen_monotonic_s: float
    enters: dict[int, dict[str, Any]] = field(default_factory=dict)
    detection_phase: str = "unknown"


class TraceAnalyzer:
    """Detect Not-Entered-Hang facts."""

    def __init__(
        self,
        *,
        run_id: str,
        initial_heartbeat_timeout_s: float | None = None,
        heartbeat_timeout_s: float = 30.0,
        collective_timeout_s: float = 60.0,
        delayed_enter_threshold_s: float = 30.0,
        checkpoint_timeout_s: float = 1800.0,
        detect_heartbeat_timeouts: bool = False,
    ) -> None:
        self.run_id = run_id
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.initial_heartbeat_timeout_s = float(
            heartbeat_timeout_s
            if initial_heartbeat_timeout_s is None
            else initial_heartbeat_timeout_s
        )
        self.collective_timeout_s = float(collective_timeout_s)
        self.delayed_enter_threshold_s = float(delayed_enter_threshold_s)
        self.checkpoint_timeout_s = float(checkpoint_timeout_s)
        self.detect_heartbeat_timeouts = detect_heartbeat_timeouts

        self._processes: dict[int, _ProcessStart] = {}
        self._heartbeats: dict[int, _Heartbeat] = {}
        self._probe_status: dict[int, dict[str, Any]] = {}
        self._comm_members: dict[tuple[str, str], dict[int, int]] = {}
        self._rounds: dict[tuple[str, str, int], _CollectiveRound] = {}
        self._reported: set[tuple[Any, ...]] = set()

    def ingest(
        self,
        event: dict[str, Any],
        *,
        observed_monotonic_s: float,
        observed_unix_ns: int,
    ) -> bool:
        """Ingest one decoded JSON event.

        Returns ``True`` when the event belongs to this run and was accepted.
        """
        if event.get("run_id") != self.run_id:
            return False

        event_type = event.get("event")
        if event_type == "process_start":
            rank = _as_int(event.get("rank"), default=-1)
            if rank < 0:
                return False
            timestamp_ns = _as_int(event.get("timestamp_unix_ns"), default=observed_unix_ns)
            # Liveness follows the monitor's local receipt clock, like NVIDIA's
            # rank-monitor design. It must not depend on wall-clock sync across hosts.
            effective_seen = observed_monotonic_s
            previous = self._processes.get(rank)
            if previous is None or timestamp_ns >= _as_int(
                previous.event.get("timestamp_unix_ns"), default=0
            ):
                self._processes[rank] = _ProcessStart(event, effective_seen)
            return True

        if event_type == "heartbeat":
            rank = _as_int(event.get("rank"), default=-1)
            if rank < 0:
                return False
            timestamp_ns = _as_int(event.get("timestamp_unix_ns"), default=observed_unix_ns)
            previous = self._heartbeats.get(rank)
            if previous is None or timestamp_ns >= _as_int(
                previous.event.get("timestamp_unix_ns"), default=0
            ):
                self._heartbeats[rank] = _Heartbeat(
                    event=event,
                    effective_seen_monotonic_s=observed_monotonic_s,
                )
            return True

        if event_type == "probe_status":
            rank = _as_int(event.get("rank"), default=-1)
            if rank < 0:
                return False
            previous = self._probe_status.get(rank)
            timestamp_ns = _as_int(event.get("timestamp_unix_ns"), default=observed_unix_ns)
            if previous is None or timestamp_ns >= _as_int(
                previous.get("timestamp_unix_ns"), default=0
            ):
                self._probe_status[rank] = event
            return True

        if event_type == "comm_init" and _as_int(event.get("result"), default=1) == 0:
            comm_hash = str(event.get("comm_uid_hash", ""))
            comm_rank = _as_int(event.get("comm_rank"), default=-1)
            global_rank = _as_int(event.get("rank"), default=-1)
            if comm_hash and comm_rank >= 0 and global_rank >= 0:
                self._comm_members.setdefault((self.run_id, comm_hash), {})[comm_rank] = global_rank
            return True

        if event_type != "nccl_call" or event.get("phase") != "enter":
            return event_type in {"process_start", "comm_destroy", "nccl_call"}

        api = str(event.get("api", ""))
        if api not in _COLLECTIVE_APIS:
            return True

        comm_hash = str(event.get("comm_uid_hash", ""))
        comm_seq = _as_int(event.get("comm_seq"), default=-1)
        comm_rank = _as_int(event.get("comm_rank"), default=-1)
        comm_nranks = _as_int(event.get("comm_nranks"), default=0)
        if not comm_hash or comm_seq < 0 or comm_rank < 0 or comm_nranks <= 0:
            return False

        key = (self.run_id, comm_hash, comm_seq)
        collective = self._rounds.get(key)
        if collective is None:
            collective = _CollectiveRound(
                run_id=self.run_id,
                comm_uid_hash=comm_hash,
                comm_seq=comm_seq,
                expected_nranks=comm_nranks,
                first_seen_monotonic_s=observed_monotonic_s,
            )
            self._rounds[key] = collective
        else:
            collective.expected_nranks = max(collective.expected_nranks, comm_nranks)

        # Keep the first entry per communicator rank. Duplicate records can occur if a
        # monitor restarts and replays a partial file.
        collective.enters.setdefault(comm_rank, event)
        global_rank = _as_int(event.get("rank"), default=-1)
        if global_rank >= 0:
            self._comm_members.setdefault((self.run_id, comm_hash), {})[comm_rank] = global_rank
        return True

    def scan(
        self,
        *,
        now_monotonic_s: float,
        now_unix_ns: int,
    ) -> list[Finding]:
        findings: list[Finding] = []
        if self.detect_heartbeat_timeouts:
            self._scan_heartbeats(
                now_monotonic_s=now_monotonic_s,
                now_unix_ns=now_unix_ns,
                output=findings,
            )
        completed_rounds: list[tuple[str, str, int]] = []
        for key, collective in list(self._rounds.items()):
            detection_phase = self._collective_detection_phase(collective)
            checkpointing = detection_phase == "checkpointing"
            if len(collective.enters) >= collective.expected_nranks:
                delayed = self._detect_delayed_enter(
                    collective,
                    now_unix_ns,
                    detection_phase=detection_phase,
                )
                self._emit_once(("delayed_collective_enter", *key), delayed, findings)
                completed_rounds.append(key)
                continue

            elapsed_s = now_monotonic_s - collective.first_seen_monotonic_s
            timeout_s = self.checkpoint_timeout_s if checkpointing else self.collective_timeout_s
            if elapsed_s < timeout_s:
                continue
            missing = self._detect_missing_enter(
                collective,
                now_monotonic_s=now_monotonic_s,
                now_unix_ns=now_unix_ns,
                elapsed_s=elapsed_s,
                detection_phase=detection_phase,
                timeout_s=timeout_s,
            )
            self._emit_once(("collective_missing_enter", *key), missing, findings)

        for key in completed_rounds:
            self._rounds.pop(key, None)

        return findings

    def _scan_heartbeats(
        self,
        *,
        now_monotonic_s: float,
        now_unix_ns: int,
        output: list[Finding],
    ) -> None:
        for rank, process in self._processes.items():
            pid = _as_int(process.event.get("pid"), default=-1)
            heartbeat = self._heartbeats.get(rank)
            if heartbeat is None or _as_int(heartbeat.event.get("pid"), default=-2) != pid:
                age_s = now_monotonic_s - process.effective_seen_monotonic_s
                if age_s <= self.initial_heartbeat_timeout_s:
                    continue
                finding = Finding(
                    hang_type="rank_heartbeat_timeout",
                    run_id=self.run_id,
                    comm_uid_hash="",
                    comm_seq=-1,
                    detected_at_unix_ns=now_unix_ns,
                    details={
                        "rank": rank,
                        "pid": pid,
                        "timeout_type": "initial_heartbeat",
                        "heartbeat_age_s": max(0.0, age_s),
                        "reason": "probe_heartbeat_not_observed",
                        "confidence": "suspected",
                    },
                )
                self._emit_once(("rank_heartbeat_timeout", rank, pid, "initial"), finding, output)
                continue

            age_s = now_monotonic_s - heartbeat.effective_seen_monotonic_s
            if age_s <= self.heartbeat_timeout_s:
                continue
            finding = Finding(
                hang_type="rank_heartbeat_timeout",
                run_id=self.run_id,
                comm_uid_hash="",
                comm_seq=-1,
                detected_at_unix_ns=now_unix_ns,
                details={
                    "rank": rank,
                    "pid": pid,
                    "timeout_type": "subsequent_heartbeat",
                    "heartbeat_age_s": max(0.0, age_s),
                    "last_nccl_api": heartbeat.event.get("last_nccl_api"),
                    "last_nccl_phase": heartbeat.event.get("last_nccl_phase"),
                    "last_nccl_seq": heartbeat.event.get("last_nccl_seq"),
                    "last_progress_mono_ns": heartbeat.event.get("last_progress_mono_ns"),
                    "dropped_events": _as_int(heartbeat.event.get("dropped_events"), default=0),
                    "reason": "rank_process_or_probe_thread_unresponsive",
                    "confidence": "suspected",
                },
            )
            self._emit_once(("rank_heartbeat_timeout", rank, pid, "subsequent"), finding, output)

    def _detect_delayed_enter(
        self,
        collective: _CollectiveRound,
        now_unix_ns: int,
        *,
        detection_phase: str,
    ) -> Finding | None:
        if not collective.enters:
            return None
        timestamps = {
            rank: _as_int(event.get("timestamp_unix_ns"), default=0)
            for rank, event in collective.enters.items()
        }
        if not timestamps or min(timestamps.values()) <= 0:
            return None
        earliest = min(timestamps.values())
        latest = max(timestamps.values())
        spread_s = (latest - earliest) / 1_000_000_000
        threshold_s = (
            self.checkpoint_timeout_s
            if detection_phase == "checkpointing"
            else self.delayed_enter_threshold_s
        )
        if spread_s <= threshold_s:
            return None
        slow_ranks = sorted(
            rank
            for rank, timestamp in timestamps.items()
            if (timestamp - earliest) / 1_000_000_000 > threshold_s
        )
        return Finding(
            hang_type="delayed_collective_enter",
            run_id=collective.run_id,
            comm_uid_hash=collective.comm_uid_hash,
            comm_seq=collective.comm_seq,
            detected_at_unix_ns=now_unix_ns,
            details={
                "slow_type": "pre_communication_slow",
                "api": next(iter(collective.enters.values())).get("api"),
                "enter_spread_s": spread_s,
                "slow_comm_ranks": slow_ranks,
                "detection_phase": detection_phase,
                "detection_threshold_s": threshold_s,
                "threshold_reason": self._threshold_reason(detection_phase),
                "reason": "collective_enter_spread_exceeded_threshold",
                "clock_assumption": "hosts have synchronized wall clocks",
                "confidence": "suspected",
            },
        )

    def _detect_missing_enter(
        self,
        collective: _CollectiveRound,
        *,
        now_monotonic_s: float,
        now_unix_ns: int,
        elapsed_s: float,
        detection_phase: str,
        timeout_s: float,
    ) -> Finding:
        entered = sorted(collective.enters)
        missing = sorted(set(range(collective.expected_nranks)) - set(entered))
        members = self._comm_members.get((collective.run_id, collective.comm_uid_hash), {})
        rank_status: list[dict[str, Any]] = []
        known_states: list[str] = []
        trace_event_loss_possible = False
        for comm_rank in missing:
            global_rank = members.get(comm_rank)
            status: dict[str, Any] = {"comm_rank": comm_rank, "rank": global_rank}
            heartbeat = self._heartbeats.get(global_rank) if global_rank is not None else None
            probe_status = self._probe_status.get(global_rank) if global_rank is not None else None
            dropped_events = _as_int(
                probe_status.get("dropped_events") if probe_status else 0, default=0
            )
            trace_event_loss_possible = trace_event_loss_possible or dropped_events > 0
            if heartbeat is None:
                status["heartbeat"] = "unknown"
            else:
                age_s = now_monotonic_s - heartbeat.effective_seen_monotonic_s
                state = "stale" if age_s > self.heartbeat_timeout_s else "alive"
                status.update(
                    {
                        "heartbeat": state,
                        "heartbeat_age_s": max(0.0, age_s),
                        "phase": str(heartbeat.event.get("phase") or "unknown"),
                    }
                )
                known_states.append(state)
            status["probe_dropped_events"] = dropped_events
            rank_status.append(status)

        if trace_event_loss_possible:
            reason = "probe_event_loss_possible"
            confidence = "suspected"
        elif "stale" in known_states:
            reason = "rank_exit_or_crash_suspected"
            confidence = "suspected"
        elif known_states and all(state == "alive" for state in known_states):
            reason = "rank_alive_but_not_entered"
            confidence = "observed"
        else:
            reason = "missing_rank_status_unknown"
            confidence = "observed"

        first_event = next(iter(collective.enters.values()))
        return Finding(
            hang_type="collective_missing_enter",
            run_id=collective.run_id,
            comm_uid_hash=collective.comm_uid_hash,
            comm_seq=collective.comm_seq,
            detected_at_unix_ns=now_unix_ns,
            details={
                "api": first_event.get("api"),
                "expected_nranks": collective.expected_nranks,
                "entered_comm_ranks": entered,
                "missing_comm_ranks": missing,
                "missing_rank_status": rank_status,
                "waited_s": elapsed_s,
                "detection_phase": detection_phase,
                "detection_threshold_s": timeout_s,
                "threshold_reason": self._threshold_reason(detection_phase),
                "reason": reason,
                "confidence": confidence,
                "trace_event_loss_possible": trace_event_loss_possible,
            },
        )

    def _collective_detection_phase(self, collective: _CollectiveRound) -> str:
        """Return a stable phase for one collective round.

        A checkpoint can block the heartbeat publisher, so a round that has once
        been correlated with ``checkpointing`` keeps that phase until it resolves.
        """
        if collective.detection_phase == "checkpointing":
            return collective.detection_phase

        global_ranks = {
            _as_int(event.get("rank"), default=-1) for event in collective.enters.values()
        }
        global_ranks.update(
            self._comm_members.get((collective.run_id, collective.comm_uid_hash), {}).values()
        )
        phases = {
            str(heartbeat.event.get("phase") or "unknown")
            for rank in global_ranks
            if rank >= 0 and (heartbeat := self._heartbeats.get(rank)) is not None
        }
        if "checkpointing" in phases:
            collective.detection_phase = "checkpointing"
        elif len(phases) == 1:
            collective.detection_phase = next(iter(phases))
        elif phases:
            collective.detection_phase = "mixed"
        return collective.detection_phase

    @staticmethod
    def _threshold_reason(detection_phase: str) -> str:
        if detection_phase == "checkpointing":
            return "heartbeat_phase_checkpointing"
        return "normal_collective_phase"

    def _emit_once(
        self,
        dedupe_key: tuple[Any, ...],
        finding: Finding | None,
        output: list[Finding],
    ) -> bool:
        if finding is None or dedupe_key in self._reported:
            return False
        self._reported.add(dedupe_key)
        output.append(finding)
        return True


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _effective_seen_monotonic(
    *, timestamp_ns: int, observed_monotonic_s: float, observed_unix_ns: int
) -> float:
    # Preserve event age across analyzer restarts. Future timestamps are clamped so
    # modest inter-node clock skew cannot make a heartbeat immortal.
    age_s = max(0.0, (observed_unix_ns - timestamp_ns) / 1_000_000_000)
    return observed_monotonic_s - age_s
