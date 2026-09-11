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

"""Low-overhead, training-progress-driven heartbeat runtime for worker ranks."""

from __future__ import annotations

import atexit
import json
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _assigned_gpu(cuda_visible_devices: str, local_rank: int) -> dict[str, Any]:
    """Describe the GPU selected by LOCAL_RANK without calling the CUDA runtime."""
    tokens = [token.strip() for token in cuda_visible_devices.split(",") if token.strip()]
    device_token = tokens[local_rank] if 0 <= local_rank < len(tokens) else None
    if device_token is None and local_rank >= 0:
        device_token = str(local_rank)
    return {
        "visible_ordinal": local_rank,
        "device_token": device_token,
    }


@dataclass(frozen=True)
class ProgressSnapshot:
    progress_seq: int
    iteration: int | None
    phase: str
    checkpoint_active: bool
    checkpoint_id: str | None
    last_progress_unix_ns: int | None
    last_progress_mono_ns: int | None


class GpuProgressHeartbeat:
    """Publish process liveness while progress is advanced only by training hooks."""

    def __init__(
        self,
        *,
        run_id: str,
        output_dir: Path,
        rank: int,
        local_rank: int,
        world_size: int,
        publish_interval_s: float,
    ) -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.publish_interval_s = publish_interval_s
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
        self.assigned_gpu = _assigned_gpu(self.cuda_visible_devices, self.local_rank)
        self.path = output_dir / f"rank_{rank}_pid_{self.pid}.heartbeat.jsonl"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._progress_seq = 0
        self._iteration: int | None = None
        self._phase = "setup"
        self._phase_before_checkpoint = "setup"
        self._checkpoint_active = False
        self._checkpoint_id: str | None = None
        self._checkpoint_seq = 0
        self._last_progress_unix_ns: int | None = None
        self._last_progress_mono_ns: int | None = None
        self._record_queue: queue.Queue[tuple[dict[str, Any], threading.Event] | None] = (
            queue.Queue()
        )
        self._thread = threading.Thread(
            target=self._publisher_loop,
            name=f"flagscale-gpu-heartbeat-rank-{rank}",
            daemon=True,
        )

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)

    def mark_progress(self, phase: str, iteration: int | None = None) -> None:
        """Advance progress using CPU-only in-memory operations on the training path."""
        now_unix_ns = time.time_ns()
        now_mono_ns = time.monotonic_ns()
        with self._lock:
            self._progress_seq += 1
            if iteration is not None:
                self._iteration = int(iteration)
            self._phase = str(phase)
            self._last_progress_unix_ns = now_unix_ns
            self._last_progress_mono_ns = now_mono_ns

    def checkpoint_start(self, iteration: int | None = None) -> str:
        """Publish a durable checkpoint transition before checkpoint work starts."""
        with self._lock:
            if not self._checkpoint_active:
                self._checkpoint_seq += 1
                self._phase_before_checkpoint = self._phase
                if iteration is not None:
                    self._iteration = int(iteration)
                iteration_label = str(self._iteration) if self._iteration is not None else "unknown"
                self._checkpoint_id = f"iteration-{iteration_label}-{self._checkpoint_seq}"
                self._checkpoint_active = True
                self._phase = "checkpointing"
            checkpoint_id = self._checkpoint_id

        assert checkpoint_id is not None
        self._publish_transition("checkpoint_start")
        return checkpoint_id

    def checkpoint_end(self) -> str | None:
        """Publish checkpoint completion and restore the preceding training phase."""
        now_unix_ns = time.time_ns()
        now_mono_ns = time.monotonic_ns()
        with self._lock:
            if not self._checkpoint_active:
                return None
            checkpoint_id = self._checkpoint_id
            self._checkpoint_active = False
            self._phase = self._phase_before_checkpoint
            self._progress_seq += 1
            self._last_progress_unix_ns = now_unix_ns
            self._last_progress_mono_ns = now_mono_ns

        self._publish_transition("checkpoint_end")
        return checkpoint_id

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            return ProgressSnapshot(
                progress_seq=self._progress_seq,
                iteration=self._iteration,
                phase=self._phase,
                checkpoint_active=self._checkpoint_active,
                checkpoint_id=self._checkpoint_id,
                last_progress_unix_ns=self._last_progress_unix_ns,
                last_progress_mono_ns=self._last_progress_mono_ns,
            )

    def stop(self) -> None:
        self._stop.set()
        self._record_queue.put(None)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=min(self.publish_interval_s + 0.5, 2.0))

    def _base_record(self, event: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "schema_version": 2,
            "component": "gpu_progress_heartbeat",
            "event": event,
            "run_id": self.run_id,
            "timestamp_unix_ns": time.time_ns(),
            "timestamp_mono_ns": time.monotonic_ns(),
            "hostname": self.hostname,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "pid": self.pid,
            "cuda_visible_devices": self.cuda_visible_devices,
            "assigned_gpu": self.assigned_gpu,
            "progress_seq": snapshot.progress_seq,
            "iteration": snapshot.iteration,
            "phase": snapshot.phase,
            "checkpoint_active": snapshot.checkpoint_active,
            "checkpoint_id": snapshot.checkpoint_id,
            "last_progress_unix_ns": snapshot.last_progress_unix_ns,
            "last_progress_mono_ns": snapshot.last_progress_mono_ns,
        }

    @staticmethod
    def _write_payload(file_obj, record: dict[str, Any]) -> None:
        file_obj.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        file_obj.flush()

    def _write_record(self, file_obj, event: str) -> None:
        self._write_payload(file_obj, self._base_record(event))

    def _publish_transition(self, event: str) -> bool:
        """Wait briefly until the publisher has flushed a lifecycle event."""
        if not self._thread.is_alive():
            return False
        written = threading.Event()
        self._record_queue.put((self._base_record(event), written))
        return written.wait(timeout=1.0)

    def _publisher_loop(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as output:
                self._write_record(output, "process_start")
                self._write_record(output, "heartbeat")
                while True:
                    try:
                        pending = self._record_queue.get(timeout=self.publish_interval_s)
                    except queue.Empty:
                        if self._stop.is_set():
                            break
                        self._write_record(output, "heartbeat")
                        continue
                    if pending is None:
                        if self._stop.is_set():
                            break
                        continue
                    record, written = pending
                    try:
                        self._write_payload(output, record)
                    finally:
                        written.set()
                self._write_record(output, "process_end")
        except Exception:
            # Diagnostics must never make the training worker fail.
            return
        finally:
            while True:
                try:
                    pending = self._record_queue.get_nowait()
                except queue.Empty:
                    break
                if pending is not None:
                    pending[1].set()


_runtime_lock = threading.Lock()
_runtime: GpuProgressHeartbeat | None = None


def initialize_from_env() -> GpuProgressHeartbeat | None:
    """Start the worker-side publisher when enabled by the FlagScale runner."""
    global _runtime
    if not _env_enabled("FLAGSCALE_GPU_HEARTBEAT_ENABLE"):
        return None
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        rank = _env_int("RANK", -1)
        output_dir = os.getenv("FLAGSCALE_GPU_HEARTBEAT_DIR")
        run_id = os.getenv("FLAGSCALE_GPU_HEARTBEAT_RUN_ID")
        if rank < 0 or not output_dir or not run_id:
            return None
        _runtime = GpuProgressHeartbeat(
            run_id=run_id,
            output_dir=Path(output_dir),
            rank=rank,
            local_rank=_env_int("LOCAL_RANK", -1),
            world_size=_env_int("WORLD_SIZE", 0),
            publish_interval_s=_env_float("FLAGSCALE_GPU_HEARTBEAT_INTERVAL_SEC", 5.0),
        )
        _runtime.start()
        return _runtime


def set_phase(phase: str) -> None:
    runtime = _runtime or initialize_from_env()
    if runtime is not None:
        runtime.set_phase(phase)


def mark_progress(phase: str, iteration: int | None = None) -> None:
    runtime = _runtime or initialize_from_env()
    if runtime is not None:
        runtime.mark_progress(phase, iteration)


def mark_training_progress(current_iteration: Any) -> None:
    """Record a completed Megatron iteration without changing its training loop API."""
    try:
        completed_iteration = int(current_iteration) + 1
    except (TypeError, ValueError):
        completed_iteration = None
    mark_progress("train", completed_iteration)


def checkpoint_start(current_iteration: Any = None) -> str | None:
    runtime = _runtime or initialize_from_env()
    if runtime is None:
        return None
    try:
        iteration = int(current_iteration) if current_iteration is not None else None
    except (TypeError, ValueError):
        iteration = None
    return runtime.checkpoint_start(iteration)


def checkpoint_end() -> str | None:
    runtime = _runtime or initialize_from_env()
    if runtime is None:
        return None
    return runtime.checkpoint_end()


def shutdown() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        runtime.stop()


atexit.register(shutdown)
