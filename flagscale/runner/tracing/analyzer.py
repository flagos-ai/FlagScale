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


@dataclass
class _P2PCall:
    event: dict[str, Any]
    effective_seen_monotonic_s: float


@dataclass
class _P2PDirection:
    calls: dict[tuple[Any, ...], _P2PCall] = field(default_factory=dict)
    by_operation: dict[tuple[str, int], set[tuple[Any, ...]]] = field(default_factory=dict)
    by_time_bucket: dict[tuple[str, int], set[tuple[Any, ...]]] = field(default_factory=dict)


class TraceAnalyzer:
    """Detect Not-Entered-Hang and collective Inconsistent-Hang facts."""

    def __init__(
        self,
        *,
        run_id: str,
        initial_heartbeat_timeout_s: float | None = None,
        heartbeat_timeout_s: float = 30.0,
        collective_timeout_s: float = 60.0,
        delayed_enter_threshold_s: float = 30.0,
        p2p_timeout_s: float = 60.0,
        p2p_match_window_s: float = 30.0,
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
        self.p2p_timeout_s = float(p2p_timeout_s)
        self.p2p_match_window_s = float(p2p_match_window_s)
        self.detect_heartbeat_timeouts = detect_heartbeat_timeouts

        self._processes: dict[int, _ProcessStart] = {}
        self._heartbeats: dict[int, _Heartbeat] = {}
        self._probe_status: dict[int, dict[str, Any]] = {}
        self._comm_members: dict[tuple[str, str], dict[int, int]] = {}
        self._rounds: dict[tuple[str, str, int], _CollectiveRound] = {}
        self._p2p_calls: dict[tuple[str, str, int, int], _P2PDirection] = {}
        self._seen_p2p_ids: set[tuple[Any, ...]] = set()
        self._resolved_p2p_ids: set[tuple[Any, ...]] = set()
        self._reported_missing_p2p_until: dict[tuple[Any, ...], float] = {}
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
        if api in {"ncclSend", "ncclRecv"}:
            return self._ingest_p2p(
                event,
                observed_monotonic_s=observed_monotonic_s,
                observed_unix_ns=observed_unix_ns,
            )
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
            mismatch = self._detect_signature_mismatch(collective, now_unix_ns)
            if mismatch is not None:
                self._emit_once(("collective_signature_mismatch", *key), mismatch, findings)
                if len(collective.enters) >= collective.expected_nranks:
                    # Every rank entered, so the signature mismatch fully explains
                    # this round and there is no missing-enter condition to report.
                    completed_rounds.append(key)
                    continue
                # A mismatch among the observed ranks does not explain ranks that
                # never entered. Keep evaluating the incomplete round for H1.

            if len(collective.enters) >= collective.expected_nranks:
                delayed = self._detect_delayed_enter(collective, now_unix_ns)
                self._emit_once(("delayed_collective_enter", *key), delayed, findings)
                completed_rounds.append(key)
                continue

            elapsed_s = now_monotonic_s - collective.first_seen_monotonic_s
            if elapsed_s < self.collective_timeout_s:
                continue
            missing = self._detect_missing_enter(
                collective,
                now_monotonic_s=now_monotonic_s,
                now_unix_ns=now_unix_ns,
                elapsed_s=elapsed_s,
            )
            self._emit_once(("collective_missing_enter", *key), missing, findings)

        for key in completed_rounds:
            self._rounds.pop(key, None)

        self._scan_p2p(
            now_monotonic_s=now_monotonic_s,
            now_unix_ns=now_unix_ns,
            output=findings,
        )
        return findings

    def _ingest_p2p(
        self,
        event: dict[str, Any],
        *,
        observed_monotonic_s: float,
        observed_unix_ns: int,
    ) -> bool:
        comm_hash = str(event.get("comm_uid_hash", ""))
        comm_rank = _as_int(event.get("comm_rank"), default=-1)
        peer = _as_int(event.get("peer"), default=-1)
        if not comm_hash or comm_rank < 0 or peer < 0:
            return False

        event_id = _p2p_event_id(event)
        if event_id in self._seen_p2p_ids:
            return True
        self._seen_p2p_ids.add(event_id)

        api = str(event.get("api", ""))
        src_rank, dst_rank = (comm_rank, peer) if api == "ncclSend" else (peer, comm_rank)
        key = (self.run_id, comm_hash, src_rank, dst_rank)
        timestamp_ns = _as_int(event.get("timestamp_unix_ns"), default=observed_unix_ns)
        call = _P2PCall(
            event=event,
            effective_seen_monotonic_s=_effective_seen_monotonic(
                timestamp_ns=timestamp_ns,
                observed_monotonic_s=observed_monotonic_s,
                observed_unix_ns=observed_unix_ns,
            ),
        )
        direction = self._p2p_calls.setdefault(key, _P2PDirection())
        self._add_p2p_call(direction, call)

        opposite_api = "ncclRecv" if api == "ncclSend" else "ncclSend"
        exact_candidates = self._p2p_candidates(
            direction,
            event,
            expected_api=opposite_api,
            require_same_signature=True,
        )
        # Resolve only a unique exact counterpart. Multiple candidates remain for the
        # timeout path, which reports ambiguity instead of inventing an ordering.
        if len(exact_candidates) == 1:
            candidate = exact_candidates[0]
            reverse_candidates = self._p2p_candidates(
                direction,
                candidate.event,
                expected_api=api,
                require_same_signature=True,
            )
            if len(reverse_candidates) == 1:
                self._resolve_p2p_calls(key, call, candidate)
        return True

    def _scan_p2p(
        self,
        *,
        now_monotonic_s: float,
        now_unix_ns: int,
        output: list[Finding],
    ) -> None:
        for key, direction in list(self._p2p_calls.items()):
            for call in list(direction.calls.values()):
                event = call.event
                event_id = _p2p_event_id(event)
                if event_id not in direction.calls:
                    continue
                age_s = now_monotonic_s - call.effective_seen_monotonic_s
                if age_s < self.p2p_timeout_s:
                    continue

                api = str(event.get("api", ""))
                opposite_api = "ncclRecv" if api == "ncclSend" else "ncclSend"
                candidates = self._p2p_candidates(
                    direction,
                    event,
                    expected_api=opposite_api,
                    require_same_signature=False,
                )
                exact_candidates = self._p2p_candidates(
                    direction,
                    event,
                    expected_api=opposite_api,
                    require_same_signature=True,
                )
                reverse_exact_candidates: list[_P2PCall] = []

                if len(exact_candidates) == 1:
                    reverse_exact_candidates = self._p2p_candidates(
                        direction,
                        exact_candidates[0].event,
                        expected_api=api,
                        require_same_signature=True,
                    )
                    if len(reverse_exact_candidates) == 1:
                        self._resolve_p2p_calls(key, call, exact_candidates[0])
                        continue

                src_rank, dst_rank = key[2], key[3]
                if not candidates:
                    retain_until = self._reported_missing_p2p_until.get(event_id)
                    if retain_until is not None:
                        if now_monotonic_s < retain_until:
                            continue
                        self._resolve_p2p_calls(key, call)
                        continue

                    finding = Finding(
                        hang_type="p2p_missing_counterpart",
                        run_id=self.run_id,
                        comm_uid_hash=key[1],
                        comm_seq=_as_int(event.get("comm_seq"), default=-1),
                        detected_at_unix_ns=now_unix_ns,
                        details={
                            "observed_api": api,
                            "expected_api": opposite_api,
                            "src_comm_rank": src_rank,
                            "dst_comm_rank": dst_rank,
                            "count": _as_int(event.get("count"), default=-1),
                            "datatype": _as_int(event.get("datatype"), default=-1),
                            "local_group_id": _as_int(event.get("group_id"), default=0),
                            "local_group_op_index": _as_int(event.get("group_op_index"), default=0),
                            "local_p2p_op_index": _as_int(event.get("p2p_op_index"), default=-1),
                            "waited_s": max(0.0, age_s),
                            "reason": "counterpart_not_observed_in_match_window",
                            "clock_assumption": "hosts have synchronized wall clocks",
                            "confidence": "suspected",
                        },
                    )
                    self._emit_once(("p2p_missing_counterpart", event_id), finding, output)
                    # Keep the reported call for one bounded grace period. A peer's
                    # trace file may be polled later even though the counterpart was
                    # issued within the matching window.
                    self._reported_missing_p2p_until[event_id] = (
                        now_monotonic_s + self.p2p_timeout_s
                    )
                    continue

                reverse_candidates: list[_P2PCall] = []
                if len(candidates) == 1:
                    reverse_candidates = self._p2p_candidates(
                        direction,
                        candidates[0].event,
                        expected_api=api,
                        require_same_signature=False,
                    )
                if len(candidates) == 1 and not exact_candidates and len(reverse_candidates) == 1:
                    candidate = candidates[0]
                    send_event, recv_event = (
                        (event, candidate.event) if api == "ncclSend" else (candidate.event, event)
                    )
                    finding = Finding(
                        hang_type="p2p_call_mismatch",
                        run_id=self.run_id,
                        comm_uid_hash=key[1],
                        comm_seq=_as_int(send_event.get("comm_seq"), default=-1),
                        detected_at_unix_ns=now_unix_ns,
                        details={
                            "mismatch_type": "count_or_datatype_mismatch",
                            "src_comm_rank": src_rank,
                            "dst_comm_rank": dst_rank,
                            "send": _p2p_summary(send_event),
                            "recv": _p2p_summary(recv_event),
                            "clock_assumption": "hosts have synchronized wall clocks",
                            "confidence": "confirmed",
                        },
                    )
                    pair_ids = tuple(
                        sorted(
                            (event_id, _p2p_event_id(candidate.event)),
                            key=repr,
                        )
                    )
                    self._emit_once(("p2p_call_mismatch", *pair_ids), finding, output)
                    self._resolve_p2p_calls(key, call, candidate)
                    continue

                ambiguous_by_id = {
                    _p2p_event_id(item.event): item
                    for item in [
                        call,
                        *candidates,
                        *reverse_candidates,
                        *reverse_exact_candidates,
                    ]
                }
                ambiguous_calls = list(ambiguous_by_id.values())
                ambiguous_ids = tuple(
                    sorted((_p2p_event_id(item.event) for item in ambiguous_calls), key=repr)
                )
                finding = Finding(
                    hang_type="ambiguous_p2p_match",
                    run_id=self.run_id,
                    comm_uid_hash=key[1],
                    comm_seq=_as_int(event.get("comm_seq"), default=-1),
                    detected_at_unix_ns=now_unix_ns,
                    details={
                        "src_comm_rank": src_rank,
                        "dst_comm_rank": dst_rank,
                        "candidate_calls": [_p2p_summary(item.event) for item in ambiguous_calls],
                        "reason": "multiple_time_window_candidates",
                        "clock_assumption": "hosts have synchronized wall clocks",
                        "confidence": "unknown",
                    },
                )
                self._emit_once(("ambiguous_p2p_match", *ambiguous_ids), finding, output)
                self._resolve_p2p_calls(key, *ambiguous_calls)

            if not direction.calls:
                self._p2p_calls.pop(key, None)

        # Keep resolved IDs until the next scan so a replayed record from the same
        # polling batch remains deduplicated, without retaining its call payload in
        # the candidate indexes.
        self._seen_p2p_ids.difference_update(self._resolved_p2p_ids)
        self._resolved_p2p_ids.clear()

    def _add_p2p_call(self, direction: _P2PDirection, call: _P2PCall) -> None:
        event = call.event
        event_id = _p2p_event_id(event)
        direction.calls[event_id] = call

        api = str(event.get("api", ""))
        operation_index = _as_int(event.get("p2p_op_index"), default=-1)
        if operation_index >= 0:
            direction.by_operation.setdefault((api, operation_index), set()).add(event_id)

        bucket = self._p2p_time_bucket(event)
        if bucket is not None:
            direction.by_time_bucket.setdefault((api, bucket), set()).add(event_id)

    def _resolve_p2p_calls(
        self,
        key: tuple[str, str, int, int],
        *calls: _P2PCall,
    ) -> None:
        direction = self._p2p_calls.get(key)
        if direction is None:
            return
        for call in calls:
            event = call.event
            event_id = _p2p_event_id(event)
            if direction.calls.pop(event_id, None) is None:
                continue
            self._reported_missing_p2p_until.pop(event_id, None)
            self._resolved_p2p_ids.add(event_id)

            api = str(event.get("api", ""))
            operation_index = _as_int(event.get("p2p_op_index"), default=-1)
            if operation_index >= 0:
                self._discard_p2p_index(
                    direction.by_operation,
                    (api, operation_index),
                    event_id,
                )

            bucket = self._p2p_time_bucket(event)
            if bucket is not None:
                self._discard_p2p_index(
                    direction.by_time_bucket,
                    (api, bucket),
                    event_id,
                )

        if not direction.calls:
            self._p2p_calls.pop(key, None)

    @staticmethod
    def _discard_p2p_index(
        index: dict[tuple[str, int], set[tuple[Any, ...]]],
        key: tuple[str, int],
        event_id: tuple[Any, ...],
    ) -> None:
        event_ids = index.get(key)
        if event_ids is None:
            return
        event_ids.discard(event_id)
        if not event_ids:
            index.pop(key, None)

    def _p2p_time_bucket(self, event: dict[str, Any]) -> int | None:
        timestamp_ns = _as_int(event.get("timestamp_unix_ns"), default=0)
        if timestamp_ns <= 0:
            return None
        bucket_width_ns = max(1, int(self.p2p_match_window_s * 1_000_000_000))
        return timestamp_ns // bucket_width_ns

    def _p2p_candidates(
        self,
        direction: _P2PDirection,
        event: dict[str, Any],
        *,
        expected_api: str,
        require_same_signature: bool,
    ) -> list[_P2PCall]:
        operation_index = _as_int(event.get("p2p_op_index"), default=-1)
        if operation_index >= 0:
            indexed_event_ids = direction.by_operation.get((expected_api, operation_index), set())
            indexed_candidates = self._p2p_calls_from_ids(
                direction,
                indexed_event_ids,
                event,
                require_same_signature=require_same_signature,
            )
            if indexed_candidates or event.get("p2p_op_index_scope") == "peer_direction":
                # New probe events use an authoritative per-peer/direction index.
                # An empty exact-signature result must reach the mismatch or missing
                # path rather than being replaced by a nearby operation.
                return indexed_candidates

        # Older traces use a communicator-wide P2P index, which can differ between
        # peers, and partial traces may not have a usable index. Search only the
        # neighboring time buckets instead of rescanning the full direction history.
        bucket = self._p2p_time_bucket(event)
        if bucket is None:
            return []
        event_ids: set[tuple[Any, ...]] = set()
        for candidate_bucket in (bucket - 1, bucket, bucket + 1):
            event_ids.update(direction.by_time_bucket.get((expected_api, candidate_bucket), set()))
        candidates = self._p2p_calls_from_ids(
            direction,
            event_ids,
            event,
            require_same_signature=require_same_signature,
        )

        # Preserve the old-trace sequence hint when it happens to align. New probe
        # events normally return from the direct operation-index lookup above.
        if operation_index >= 0:
            indexed_candidates = [
                candidate
                for candidate in candidates
                if _as_int(candidate.event.get("p2p_op_index"), default=-2) == operation_index
            ]
            if indexed_candidates:
                return indexed_candidates
        return candidates

    def _p2p_calls_from_ids(
        self,
        direction: _P2PDirection,
        event_ids: set[tuple[Any, ...]],
        event: dict[str, Any],
        *,
        require_same_signature: bool,
    ) -> list[_P2PCall]:
        candidates: list[_P2PCall] = []
        for event_id in sorted(event_ids, key=repr):
            candidate = direction.calls.get(event_id)
            if candidate is None:
                continue
            if require_same_signature and _p2p_signature(candidate.event) != _p2p_signature(event):
                continue
            if not _within_p2p_window(
                candidate.event,
                event,
                window_s=self.p2p_match_window_s,
            ):
                continue
            candidates.append(candidate)
        return candidates

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

    def _detect_signature_mismatch(
        self, collective: _CollectiveRound, now_unix_ns: int
    ) -> Finding | None:
        if len(collective.enters) < 2:
            return None

        by_api: dict[str, list[int]] = {}
        for comm_rank, event in collective.enters.items():
            by_api.setdefault(str(event.get("api", "")), []).append(comm_rank)
        if len(by_api) > 1:
            return Finding(
                hang_type="collective_signature_mismatch",
                run_id=collective.run_id,
                comm_uid_hash=collective.comm_uid_hash,
                comm_seq=collective.comm_seq,
                detected_at_unix_ns=now_unix_ns,
                details={
                    "mismatch_type": "api_mismatch",
                    "api_by_comm_rank": {
                        str(rank): str(event.get("api", ""))
                        for rank, event in sorted(collective.enters.items())
                    },
                    "entered_comm_ranks": sorted(collective.enters),
                    "expected_nranks": collective.expected_nranks,
                    "confidence": "confirmed",
                },
            )

        signatures: dict[tuple[Any, ...], list[int]] = {}
        for comm_rank, event in collective.enters.items():
            signatures.setdefault(_collective_signature(event), []).append(comm_rank)
        if len(signatures) <= 1:
            return None

        api = next(iter(by_api))
        roots = {_as_int(event.get("root"), default=-1) for event in collective.enters.values()}
        mismatch_type = (
            "root_mismatch"
            if api in {"ncclBroadcast", "ncclReduce"} and len(roots) > 1
            else "parameter_mismatch"
        )
        return Finding(
            hang_type="collective_signature_mismatch",
            run_id=collective.run_id,
            comm_uid_hash=collective.comm_uid_hash,
            comm_seq=collective.comm_seq,
            detected_at_unix_ns=now_unix_ns,
            details={
                "mismatch_type": mismatch_type,
                "signature_by_comm_rank": {
                    str(rank): _signature_dict(event)
                    for rank, event in sorted(collective.enters.items())
                },
                "entered_comm_ranks": sorted(collective.enters),
                "expected_nranks": collective.expected_nranks,
                "confidence": "confirmed",
            },
        )

    def _detect_delayed_enter(
        self, collective: _CollectiveRound, now_unix_ns: int
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
        if spread_s <= self.delayed_enter_threshold_s:
            return None
        slow_ranks = sorted(
            rank
            for rank, timestamp in timestamps.items()
            if (timestamp - earliest) / 1_000_000_000 > self.delayed_enter_threshold_s
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
                "reason": reason,
                "confidence": confidence,
                "trace_event_loss_possible": trace_event_loss_possible,
            },
        )

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


def _p2p_event_id(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _as_int(event.get("rank"), default=-1),
        _as_int(event.get("pid"), default=-1),
        _as_int(event.get("call_seq"), default=-1),
        str(event.get("api", "")),
        _as_int(event.get("timestamp_unix_ns"), default=0),
    )


def _p2p_signature(event: dict[str, Any]) -> tuple[int, int]:
    return (
        _as_int(event.get("count"), default=-1),
        _as_int(event.get("datatype"), default=-1),
    )


def _within_p2p_window(first: dict[str, Any], second: dict[str, Any], *, window_s: float) -> bool:
    first_ns = _as_int(first.get("timestamp_unix_ns"), default=0)
    second_ns = _as_int(second.get("timestamp_unix_ns"), default=0)
    if first_ns <= 0 or second_ns <= 0:
        return False
    return abs(first_ns - second_ns) <= int(window_s * 1_000_000_000)


def _p2p_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": _as_int(event.get("rank"), default=-1),
        "comm_rank": _as_int(event.get("comm_rank"), default=-1),
        "api": str(event.get("api", "")),
        "peer": _as_int(event.get("peer"), default=-1),
        "count": _as_int(event.get("count"), default=-1),
        "datatype": _as_int(event.get("datatype"), default=-1),
        "timestamp_unix_ns": _as_int(event.get("timestamp_unix_ns"), default=0),
        "local_group_id": _as_int(event.get("group_id"), default=0),
        "local_group_op_index": _as_int(event.get("group_op_index"), default=0),
        "local_p2p_op_index": _as_int(event.get("p2p_op_index"), default=-1),
    }


def _collective_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    api = str(event.get("api", ""))
    root = _as_int(event.get("root"), default=-1) if api in {"ncclBroadcast", "ncclReduce"} else -1
    op = (
        _as_int(event.get("op"), default=-1)
        if api in {"ncclAllReduce", "ncclReduceScatter", "ncclReduce"}
        else -1
    )
    return (
        api,
        _as_int(event.get("count"), default=-1),
        _as_int(event.get("datatype"), default=-1),
        op,
        root,
    )


def _signature_dict(event: dict[str, Any]) -> dict[str, Any]:
    api, count, datatype, op, root = _collective_signature(event)
    return {
        "api": api,
        "count": count,
        "datatype": datatype,
        "op": op,
        "root": root,
    }
