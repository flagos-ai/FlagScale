# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Private, rebuildable on-disk Inspector index; never prune raw evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import weakref
from pathlib import Path


class EvidenceLimitError(RuntimeError):
    pass


def _close_store(db, directory):
    db.close()
    directory.cleanup()


class InspectorStore:
    def __init__(self, *, cache_kib=2048, lookup_bytes=8 * 1024 * 1024):
        # Do not inherit a training TMPDIR pointing at the shared trace filesystem.
        directory = "/tmp" if Path("/tmp").is_dir() else None
        self._temporary = tempfile.TemporaryDirectory(prefix="flagscale-inspector-", dir=directory)
        self.path = Path(self._temporary.name) / "index.sqlite"
        self.db = sqlite3.connect(self.path)
        self._cleanup = weakref.finalize(self, _close_store, self.db, self._temporary)
        self.db.execute(f"PRAGMA cache_size=-{max(64, int(cache_kib))}")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
            CREATE TABLE records (
              digest TEXT PRIMARY KEY, kind TEXT, host TEXT, pid INTEGER,
              token INTEGER, call INTEGER, rank INTEGER, api TEXT, seq INTEGER,
              comm TEXT, stamp INTEGER, body TEXT NOT NULL
            );
            CREATE INDEX bridge_call ON records(kind,host,pid,token,call);
            CREATE INDEX legacy_seq ON records(kind,host,pid,rank,api,seq,comm);
            CREATE INDEX legacy_time ON records(kind,host,pid,rank,api,stamp);
        """)
        self.lookup_bytes = lookup_bytes

    def add(self, payload):
        from .inspector import InspectorCompletion, _as_int, _normalize_api

        kind = payload.get("record_type")
        if kind in {"flagscale_profiler_v5_v1", "flagscale_bridge_error_v1"}:
            fields = (
                kind,
                str(payload.get("hostname", "")),
                _as_int(payload.get("pid")),
                _as_int(payload.get("process_token")),
                _as_int(payload.get("call_seq")),
                -1,
                "",
                -1,
                "",
                -1,
            )
        else:
            completion = InspectorCompletion.from_payload(payload)
            if completion is None:
                return
            fields = (
                "legacy",
                completion.hostname,
                completion.pid,
                -1,
                -1,
                completion.comm_rank,
                _normalize_api(completion.api),
                completion.sequence_number,
                completion.communicator_id,
                completion.start_timestamp_us,
            )
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(body.encode()).hexdigest()
        self.db.execute(
            "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (digest, *fields, body),
        )

    def _rows(self, where, params):
        total = 0
        rows = []
        # Insertion order preserves the existing refinement semantics.
        for (body,) in self.db.execute(
            "SELECT body FROM records WHERE " + where + " ORDER BY rowid", params
        ):
            total += len(body.encode())
            if total > self.lookup_bytes:
                raise EvidenceLimitError("inspector_lookup_budget_exceeded")
            rows.append(json.loads(body))
        return rows

    def for_event(self, event, window_s):
        from .inspector import _as_int, _communicator_id, _normalize_api

        host, pid = str(event.get("hostname") or ""), _as_int(event.get("pid"))
        if event.get("profiler_bridge") is True:
            rows = self._rows(
                "kind=? AND host=? AND pid=?", ("flagscale_bridge_error_v1", host, pid)
            )
            rows.extend(
                self._rows(
                    "kind=? AND host=? AND pid=? AND token=? AND call=?",
                    (
                        "flagscale_profiler_v5_v1",
                        host,
                        pid,
                        _as_int(event.get("profiler_process_token")),
                        _as_int(event.get("call_seq")),
                    ),
                )
            )
            return (), tuple(rows)
        where = "kind='legacy' AND host=? AND pid=? AND rank=? AND api=?"
        params = [host, pid, _as_int(event.get("comm_rank")), _normalize_api(event.get("api"))]
        if "comm_inspector_id" in event:
            comm = _communicator_id(event.get("comm_inspector_id"))
            where += " AND seq=? AND comm IN (?,?)"
            params.extend([_as_int(event.get("comm_api_seq")), comm, comm[:16]])
        else:
            stamp = _as_int(event.get("timestamp_unix_ns")) // 1000
            where += " AND stamp BETWEEN ? AND ?"
            params.extend([stamp - window_s * 1e6, stamp + window_s * 1e6])
        return compact_completions(self._rows(where, params)), ()

    def all_completions(self):
        # Explicit diagnostic compatibility accessor, never used by online polling.
        return compact_completions(self._rows("kind='legacy'", ()))


def compact_completions(payloads):
    from .inspector import InspectorCompletion, _completion_identity, _refines

    revisions = {}
    for payload in payloads:
        completion = InspectorCompletion.from_payload(payload)
        if completion is None:
            continue
        identity = _completion_identity(completion)
        previous = revisions.setdefault(identity, [])
        if any(_refines(old, completion) for old in previous):
            continue
        revisions[identity] = [old for old in previous if not _refines(completion, old)] + [
            completion
        ]
    return tuple(row for group in revisions.values() for row in group)


class StoredInspectorIndex:
    """Live query view owned by one reader; history stays on disk."""

    def __init__(self, store, window_s, errors, caught_up):
        self.store = store
        self.correlation_window_s = window_s
        self.errors = tuple(errors)
        self.caught_up = caught_up
        self.enabled = True

    @property
    def completions(self):
        return self.store.all_completions()

    def for_call(self, event):
        from .inspector import InspectorIndex

        reason = "inspector_read_error" if self.errors else None
        if not self.caught_up:
            reason = reason or "inspector_read_backlog"
        if reason is None:
            try:
                completions, bridge = self.store.for_event(event, self.correlation_window_s)
                return InspectorIndex(
                    True, completions, self.correlation_window_s, bridge
                ).for_call(event)
            except (sqlite3.Error, EvidenceLimitError) as exc:
                reason = (
                    str(exc) if isinstance(exc, EvidenceLimitError) else "inspector_index_error"
                )
        return {
            "inspector_correlation": "none",
            "inspector_kernel_status": "unknown",
            "kernel_execution_status": "unknown",
            "inspector_correlation_reason": reason,
        }
