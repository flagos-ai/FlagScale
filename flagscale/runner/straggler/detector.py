"""Main detector implementation for FlagScale straggler analysis."""

from collections import defaultdict
import json
import time
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.distributed as dist

    TORCH_DISTRIBUTED_AVAILABLE = True
except ImportError:
    torch = None
    dist = None
    TORCH_DISTRIBUTED_AVAILABLE = False

from .config import StragglerConfig
from .report import StragglerReport


class StragglerDetector:
    """Collect section timings and build straggler reports."""

    def __init__(
        self,
        config: StragglerConfig,
        rank: int = 0,
        world_size: int = 1,
        node_name: Optional[str] = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.node_name = node_name or f"rank-{rank}"
        self.section_timings: Dict[str, List[Tuple[int, float, Optional[float]]]] = defaultdict(list)
        self.current_step = 0
        self.enabled = config.enabled
        self.straggler_threshold = config.straggler_threshold

    def record_section(
        self,
        name: str,
        cpu_time: float,
        gpu_time: Optional[float] = None,
        step: Optional[int] = None,
    ):
        if not self.enabled:
            return
        if step is None:
            step = self.current_step
        if name not in self.config.monitor_sections:
            return
        self.section_timings[name].append((step, cpu_time, gpu_time))

    def increment_step(self):
        self.current_step += 1

    def should_profile(self, step: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        if step is None:
            step = self.current_step
        if step < self.config.warmup_steps:
            return False
        return (step - self.config.warmup_steps) % self.config.profiling_interval == 0

    def should_report(self, step: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        if step is None:
            step = self.current_step
        return step > 0 and step % self.config.report_interval_steps == 0

    def compute_section_scores(
        self,
        section_name: str,
        sample_size: Optional[int] = None,
    ) -> Dict[int, float]:
        if sample_size is None:
            sample_size = self.config.sample_size
        avg_time = self.get_recent_section_time(section_name, num_samples=sample_size)
        if avg_time is None:
            return {}
        return {self.rank: avg_time}

    def compute_all_section_scores(
        self,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Dict[int, float]]:
        all_scores = {}
        for section_name in self.config.monitor_sections:
            scores = self.compute_section_scores(section_name, sample_size)
            if scores:
                all_scores[section_name] = scores
        return all_scores

    def compute_gpu_scores(self, sample_size: Optional[int] = None) -> Dict[int, float]:
        if not self.config.enable_gpu_profile:
            return {}

        if sample_size is None:
            sample_size = self.config.sample_size

        total_time = 0.0
        for section_name in ("forward_backward", "forward", "backward"):
            recent = self.get_recent_section_time(section_name, num_samples=sample_size)
            if recent is not None:
                total_time += recent

        if total_time <= 0:
            return {}
        return {self.rank: 1.0 / total_time}

    def identify_stragglers(
        self,
        section_scores: Optional[Dict[str, Dict[int, float]]] = None,
        threshold: Optional[float] = None,
    ) -> List[int]:
        if threshold is None:
            threshold = self.straggler_threshold
        if section_scores is None:
            section_scores = self.compute_all_section_scores()
        return self._identify_stragglers_from_times(section_scores, threshold)

    def _get_collective_device(self):
        if not TORCH_DISTRIBUTED_AVAILABLE or not dist.is_initialized():
            return None

        backend = ""
        try:
            backend = str(dist.get_backend())
        except Exception:
            backend = ""

        if ("cuda" in backend or "nccl" in backend or "flagcx" in backend) and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _gather_section_times_across_ranks(self) -> Dict[str, Dict[int, float]]:
        if not TORCH_DISTRIBUTED_AVAILABLE or not dist.is_initialized():
            result = {}
            for section_name in self.config.monitor_sections:
                avg_time = self.get_recent_section_time(section_name, num_samples=5)
                if avg_time is not None:
                    result[section_name] = {self.rank: avg_time}
            return result

        device = self._get_collective_device()
        result = {}
        for section_name in self.config.monitor_sections:
            avg_time = self.get_recent_section_time(section_name, num_samples=5)
            local_time = avg_time if avg_time is not None else -1.0
            local_tensor = torch.tensor([local_time], dtype=torch.float64, device=device)
            gathered_tensors = [
                torch.zeros(1, dtype=torch.float64, device=device) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_tensors, local_tensor)

            section_times = {}
            for rank, tensor in enumerate(gathered_tensors):
                time_val = tensor.item()
                if time_val >= 0:
                    section_times[rank] = time_val
            if section_times:
                result[section_name] = section_times

        return result

    def _gather_node_names_across_ranks(self) -> Dict[int, str]:
        if not TORCH_DISTRIBUTED_AVAILABLE or not dist.is_initialized():
            return {self.rank: self.node_name}

        node_names_list = [None] * self.world_size
        dist.all_gather_object(node_names_list, self.node_name)
        return {rank: name for rank, name in enumerate(node_names_list) if name is not None}

    def generate_report(
        self,
        step: Optional[int] = None,
        gather_on_rank0: Optional[bool] = None,
    ) -> StragglerReport:
        if step is None:
            step = self.current_step
        if gather_on_rank0 is None:
            gather_on_rank0 = self.config.gather_on_rank0

        section_scores = self._gather_section_times_across_ranks()

        gpu_scores = {}
        for section_name in ("forward_backward", "forward", "backward"):
            if section_name not in section_scores:
                continue
            for rank, time_val in section_scores[section_name].items():
                if time_val <= 0:
                    continue
                gpu_scores[rank] = gpu_scores.get(rank, 0.0) + time_val
        for rank, total_time in list(gpu_scores.items()):
            gpu_scores[rank] = 1.0 / total_time if total_time > 0 else 0.0

        straggler_ranks = self._identify_stragglers_from_times(section_scores)
        node_names = self._gather_node_names_across_ranks()

        report = StragglerReport(
            step=step,
            section_scores=section_scores,
            gpu_scores=gpu_scores,
            straggler_ranks=straggler_ranks,
            node_names=node_names,
        )
        report.timestamp = time.time()
        return report

    def _identify_stragglers_from_times(
        self,
        section_times: Dict[str, Dict[int, float]],
        threshold: Optional[float] = None,
    ) -> List[int]:
        if threshold is None:
            threshold = self.straggler_threshold
        if not section_times:
            return []

        total_times = defaultdict(float)
        for rank_times in section_times.values():
            for rank, time_val in rank_times.items():
                total_times[rank] += time_val

        if not total_times:
            return []

        fastest_rank = min(total_times.items(), key=lambda item: item[1])[0]
        fastest_time = total_times[fastest_rank]
        if fastest_time <= 0:
            return []

        stragglers = []
        for rank, total_time in total_times.items():
            if rank == fastest_rank:
                continue
            slowdown_ratio = total_time / fastest_time
            if slowdown_ratio >= threshold:
                stragglers.append(rank)
        return sorted(stragglers)

    def save_report(self, report: StragglerReport, filepath: str):
        try:
            with open(filepath, "w") as file_obj:
                json.dump(report.to_dict(), file_obj, indent=2)
        except Exception as exc:
            print(f"Warning: Could not save report to {filepath}: {exc}")

    def get_recent_section_time(
        self,
        section_name: str,
        num_samples: int = 1,
    ) -> Optional[float]:
        timings = self.section_timings.get(section_name, [])
        if not timings:
            return None

        recent = timings[-num_samples:]
        if not recent:
            return None

        total_time = 0.0
        count = 0
        for _, cpu_time, gpu_time in recent:
            total_time += gpu_time if gpu_time is not None else cpu_time
            count += 1
        return total_time / count if count > 0 else None

    def get_section_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for section_name, timings in self.section_timings.items():
            if not timings:
                continue

            cpu_times = []
            gpu_times = []
            for _, cpu_time, gpu_time in timings:
                cpu_times.append(cpu_time)
                if gpu_time is not None:
                    gpu_times.append(gpu_time)

            section_stats = {
                "count": len(timings),
                "cpu_avg": sum(cpu_times) / len(cpu_times),
                "cpu_min": min(cpu_times),
                "cpu_max": max(cpu_times),
            }
            if gpu_times:
                section_stats["gpu_avg"] = sum(gpu_times) / len(gpu_times)
                section_stats["gpu_min"] = min(gpu_times)
                section_stats["gpu_max"] = max(gpu_times)
            stats[section_name] = section_stats
        return stats

    def reset(self):
        self.section_timings.clear()
        self.current_step = 0

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
