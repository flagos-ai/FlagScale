"""
Communication layer hooks and statistics collection.
"""

import time

from collections import defaultdict
from typing import Any, Dict, Optional


class CommStatsCollector:
    """
    Collects communication statistics during training.
    This class provides hooks to monitor collective operations
    (all-reduce, broadcast, etc.) and track latencies, bandwidth,
    and potential stragglers in communication.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "rank_times": defaultdict(list),
            }
        )
        self.backend = "unknown"
        self.world_size = 1
        self.rank = 0

    def set_backend_info(self, backend: str, world_size: int, rank: int):
        """Set backend and distributed training info."""
        self.backend = backend
        self.world_size = world_size
        self.rank = rank

    def record_operation(
        self,
        op_type: str,
        op_name: str,
        start_time: float,
        end_time: float,
        data_size: Optional[int] = None,
        target_ranks: Optional[list] = None,
    ):
        """
        Record statistics for a communication operation.
        Args:
            op_type: Type of operation (e.g., "all_reduce", "broadcast", "reduce")
            op_name: Name or identifier for this operation
            start_time: Operation start timestamp
            end_time: Operation end timestamp
            data_size: Size of data transferred in bytes (optional)
            target_ranks: List of ranks involved (optional)
        """
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
            if "total_data_size" not in stats:
                stats["total_data_size"] = 0
            stats["total_data_size"] += data_size

        if target_ranks is not None:
            stats["target_ranks"] = target_ranks

    def get_operation_stats(self, op_type: str, op_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        key = f"{op_type}_{op_name}"
        return self.operation_stats[key].copy()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected statistics."""
        return dict(self.operation_stats)

    def get_straggler_operations(self, threshold: float = 2.0) -> list:
        """
        Identify operations with stragglers based on timing.
        Args:
            threshold: Ratio threshold for identifying stragglers
        Returns:
            List of operations with suspected stragglers
        """
        stragglers = []

        for op_key, stats in self.operation_stats.items():
            if stats["count"] == 0:
                continue

            avg_time = stats["total_time"] / stats["count"]
            max_time = stats["max_time"]

            if max_time / avg_time >= threshold:
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
    """
    Hook for monitoring NCCL communication operations.
    This is a placeholder for NCCL-specific monitoring.
    In practice, this would integrate with NCCL's communication hooks
    or use custom wrappers around NCCL operations.
    """

    def __init__(self, collector: CommStatsCollector):
        self.collector = collector
        self.backend = "nccl"

    def wrap_all_reduce(self, op_func):
        """
        Wrap an all-reduce operation with monitoring.
        Args:
            op_func: The all-reduce function to wrap
        Returns:
            Wrapped function with monitoring
        """

        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = op_func(*args, **kwargs)
            end_time = time.perf_counter()

            self.collector.record_operation(
                op_type="all_reduce", op_name="default", start_time=start_time, end_time=end_time
            )

            return result

        return wrapped

    def wrap_broadcast(self, op_func):
        """Wrap a broadcast operation with monitoring."""

        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = op_func(*args, **kwargs)
            end_time = time.perf_counter()

            self.collector.record_operation(
                op_type="broadcast", op_name="default", start_time=start_time, end_time=end_time
            )

            return result

        return wrapped


class GlooCommHook:
    """
    Hook for monitoring Gloo communication operations.
    Placeholder for Gloo-specific monitoring.
    """

    def __init__(self, collector: CommStatsCollector):
        self.collector = collector
        self.backend = "gloo"

    def wrap_all_reduce(self, op_func):
        """Wrap an all-reduce operation with monitoring."""

        def wrapped(*args, **kwargs):
            start_time = time.perf_counter()
            result = op_func(*args, **kwargs)
            end_time = time.perf_counter()

            self.collector.record_operation(
                op_type="all_reduce", op_name="default", start_time=start_time, end_time=end_time
            )

            return result

        return wrapped


class CommProfiler:
    """
    High-level communication profiler.
    Provides a unified interface for monitoring different backends.
    """

    def __init__(self, backend: str = "auto", enabled: bool = True):
        self.collector = CommStatsCollector(enabled=enabled)
        self.hooks = {}

        if backend == "auto":
            # Detect backend automatically
            backend = self._detect_backend()

        if backend == "nccl":
            self.hooks["nccl"] = NCCLCommHook(self.collector)
        elif backend == "gloo":
            self.hooks["gloo"] = GlooCommHook(self.collector)

        self.collector.set_backend_info(backend, 1, 0)

    def _detect_backend(self) -> str:
        """Auto-detect communication backend."""
        # This is a placeholder - in practice, you'd detect
        # based on available libraries and configuration
        try:
            import torch

            if torch.cuda.is_available():
                return "nccl"
        except ImportError:
            pass
        return "gloo"

    def wrap_operation(self, op_type: str, op_func):
        """Wrap a communication operation with monitoring."""
        backend = self.collector.backend
        if backend in self.hooks:
            if op_type == "all_reduce":
                return self.hooks[backend].wrap_all_reduce(op_func)
            elif op_type == "broadcast":
                return self.hooks[backend].wrap_broadcast(op_func)

        # Default: return unwrapped function
        return op_func

    def record_custom_operation(
        self,
        op_type: str,
        op_name: str,
        start_time: float,
        end_time: float,
        data_size: Optional[int] = None,
    ):
        """Record a custom communication operation."""
        self.collector.record_operation(
            op_type=op_type,
            op_name=op_name,
            start_time=start_time,
            end_time=end_time,
            data_size=data_size,
        )

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all communication statistics."""
        return self.collector.get_all_stats()

    def get_stragglers(self, threshold: float = 2.0) -> list:
        """Get operations with suspected stragglers."""
        return self.collector.get_straggler_operations(threshold)
