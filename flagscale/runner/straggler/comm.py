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

"""Communication monitoring helpers for straggler analysis."""

import time
from collections import defaultdict
from typing import Any


class CommStatsCollector:
    """Collect basic communication timings."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.operation_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "rank_times": defaultdict(list),
            }
        )
        self.backend = "unknown"
        self.world_size = 1
        self.rank = 0

    def set_backend_info(self, backend: str, world_size: int, rank: int):
        self.backend = backend
        self.world_size = world_size
        self.rank = rank

    def record_operation(
        self,
        op_type: str,
        op_name: str,
        start_time: float,
        end_time: float,
        data_size: int | None = None,
        target_ranks: list | None = None,
    ):
        if not self.enabled:
            return

        duration = end_time - start_time
        key = f"{op_type}_{op_name}"
        stats = self.operation_stats[key]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        stats["rank_times"][self.rank].append(duration)

        if data_size is not None:
            stats["total_data_size"] = stats.get("total_data_size", 0) + data_size

        if target_ranks is not None:
            stats["target_ranks"] = target_ranks

    def get_operation_stats(self, op_type: str, op_name: str) -> dict[str, Any]:
        return self.operation_stats[f"{op_type}_{op_name}"].copy()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return dict(self.operation_stats)

    def get_straggler_operations(self, threshold: float = 2.0) -> list:
        stragglers = []
        for op_key, stats in self.operation_stats.items():
            if stats["count"] == 0:
                continue
            avg_time = stats["total_time"] / stats["count"]
            max_time = stats["max_time"]
            if avg_time > 0 and max_time / avg_time >= threshold:
                stragglers.append(
                    {
                        "operation": op_key,
                        "avg_time": avg_time,
                        "max_time": max_time,
                        "slowdown_ratio": max_time / avg_time,
                        "count": stats["count"],
                    }
                )
        return stragglers


class NCCLCommHook:
    """Wrap NCCL collectives with timing."""

    def __init__(self, collector: CommStatsCollector):
        self.collector = collector

    def wrap_all_reduce(self, op_func):
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = op_func(*args, **kwargs)
            end_time = time.perf_counter()
            self.collector.record_operation(
                "all_reduce",
                "default",
                start_time,
                end_time,
            )
            return result

        return wrapped

    def wrap_broadcast(self, op_func):
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = op_func(*args, **kwargs)
            end_time = time.perf_counter()
            self.collector.record_operation(
                "broadcast",
                "default",
                start_time,
                end_time,
            )
            return result

        return wrapped


class GlooCommHook:
    """Wrap Gloo collectives with timing."""

    def __init__(self, collector: CommStatsCollector):
        self.collector = collector

    def wrap_all_reduce(self, op_func):
        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = op_func(*args, **kwargs)
            end_time = time.perf_counter()
            self.collector.record_operation(
                "all_reduce",
                "default",
                start_time,
                end_time,
            )
            return result

        return wrapped


class CommProfiler:
    """Backend-aware communication profiler."""

    def __init__(self, backend: str = "auto", enabled: bool = True):
        self.collector = CommStatsCollector(enabled=enabled)
        self.hooks = {}

        if backend == "auto":
            backend = self._detect_backend()

        if backend == "nccl":
            self.hooks["nccl"] = NCCLCommHook(self.collector)
        elif backend == "gloo":
            self.hooks["gloo"] = GlooCommHook(self.collector)

        self.collector.set_backend_info(backend, 1, 0)

    def _detect_backend(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "nccl"
        except ImportError:
            pass
        return "gloo"

    def wrap_operation(self, op_type: str, op_func):
        backend = self.collector.backend
        if backend in self.hooks:
            if op_type == "all_reduce":
                return self.hooks[backend].wrap_all_reduce(op_func)
            if op_type == "broadcast" and hasattr(self.hooks[backend], "wrap_broadcast"):
                return self.hooks[backend].wrap_broadcast(op_func)
        return op_func

    def record_custom_operation(
        self,
        op_type: str,
        op_name: str,
        start_time: float,
        end_time: float,
        data_size: int | None = None,
    ):
        self.collector.record_operation(
            op_type,
            op_name,
            start_time,
            end_time,
            data_size=data_size,
        )
