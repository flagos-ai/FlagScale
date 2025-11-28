"""
Main StragglerDetector class for detecting and reporting performance stragglers.
"""

import json
import time

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import StragglerConfig
from .report import StragglerReport


class StragglerDetector:
    """
    Main detector class for identifying performance stragglers in distributed training.
    This class collects timing data from various training sections,
    analyzes performance across ranks, and identifies stragglers.
    Args:
        config: StragglerConfig instance
        rank: Current process rank
        world_size: Total number of processes
        node_name: Optional node name for identification
    """

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

        # Section timing data: section_name -> list of (step, cpu_time, gpu_time)
        self.section_timings: Dict[str, List[Tuple[int, float, Optional[float]]]] = defaultdict(
            list
        )

        # Step counter
        self.current_step = 0

        # Whether detector is enabled
        self.enabled = config.enabled

        # Threshold for straggler detection
        self.straggler_threshold = config.straggler_threshold

    def record_section(
        self,
        name: str,
        cpu_time: float,
        gpu_time: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Record timing for a training section.
        Args:
            name: Name of the section (e.g., "forward", "backward")
            cpu_time: CPU elapsed time in seconds
            gpu_time: GPU elapsed time in seconds (optional)
            step: Training step number (uses current_step if not provided)
        """
        if not self.enabled:
            return

        if step is None:
            step = self.current_step

        # Only record if this section is being monitored
        if name not in self.config.monitor_sections:
            return

        self.section_timings[name].append((step, cpu_time, gpu_time))

    def increment_step(self):
        """Increment the current training step."""
        self.current_step += 1

    def should_profile(self, step: Optional[int] = None) -> bool:
        """
        Check if we should profile at the given step.
        Args:
            step: Step to check (uses current_step if None)
        Returns:
            True if profiling should occur at this step
        """
        if not self.enabled:
            return False

        if step is None:
            step = self.current_step

        # Skip warmup steps
        if step < self.config.warmup_steps:
            return False

        return (step - self.config.warmup_steps) % self.config.profiling_interval == 0

    def should_report(self, step: Optional[int] = None) -> bool:
        """
        Check if we should generate a report at the given step.
        Args:
            step: Step to check (uses current_step if None)
        Returns:
            True if a report should be generated
        """
        if not self.enabled:
            return False

        if step is None:
            step = self.current_step

        return step > 0 and step % self.config.report_interval_steps == 0

    def compute_section_scores(
        self, section_name: str, sample_size: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Compute performance scores for a section across all ranks.
        Args:
            section_name: Name of the section
            sample_size: Number of recent samples to use (uses config.sample_size if None)
        Returns:
            Dictionary mapping rank to performance score
        """
        if sample_size is None:
            sample_size = self.config.sample_size

        # Get timing data for this section
        timings = self.section_timings.get(section_name, [])

        if len(timings) < sample_size:
            return {}

        # Use the most recent samples
        recent_timings = timings[-sample_size:]

        # Compute average times
        avg_times = {}
        for step, cpu_time, gpu_time in recent_timings:
            # Use GPU time if available, otherwise CPU time
            elapsed = gpu_time if gpu_time is not None else cpu_time
            avg_times[step] = elapsed

        # In a real implementation, this would gather data from all ranks
        # For now, we just return our local data
        # The actual implementation would use all-reduce to gather data

        return {self.rank: avg_times}

    def compute_all_section_scores(
        self, sample_size: Optional[int] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute scores for all monitored sections.
        Args:
            sample_size: Number of recent samples to use
        Returns:
            Dictionary mapping section name to {rank: score}
        """
        all_scores = {}

        for section_name in self.config.monitor_sections:
            scores = self.compute_section_scores(section_name, sample_size)
            if scores:
                all_scores[section_name] = scores

        return all_scores

    def compute_gpu_scores(self, sample_size: Optional[int] = None) -> Dict[int, float]:
        """
        Compute GPU performance scores across ranks.
        Args:
            sample_size: Number of recent samples to use
        Returns:
            Dictionary mapping rank to GPU score
        """
        # In a real implementation, this would collect GPU metrics
        # from all ranks and compute relative performance

        # Placeholder implementation
        if not self.config.enable_gpu_profile:
            return {}

        # Use forward + backward time as a proxy for GPU performance
        gpu_sections = ["forward", "backward"]
        gpu_time_total = 0.0

        for section in gpu_sections:
            scores = self.compute_section_scores(section, sample_size)
            if self.rank in scores:
                gpu_time_total += scores[self.rank]

        # Higher score = better performance
        # We'll use the inverse of time
        gpu_score = 1.0 / max(gpu_time_total, 1e-6)

        return {self.rank: gpu_score}

    def identify_stragglers(
        self,
        section_scores: Optional[Dict[str, Dict[int, float]]] = None,
        threshold: Optional[float] = None,
    ) -> List[int]:
        """
        Identify straggler ranks based on performance scores.
        Args:
            section_scores: Scores to analyze (computed if not provided)
            threshold: Straggler threshold (uses config.straggler_threshold if None)
        Returns:
            List of straggler ranks
        """
        if threshold is None:
            threshold = self.straggler_threshold

        if section_scores is None:
            section_scores = self.compute_all_section_scores()

        # Combine scores from all sections
        combined_scores = defaultdict(float)
        section_weights = {section: 1.0 for section in section_scores.keys()}

        for section_name, rank_scores in section_scores.items():
            weight = section_weights.get(section_name, 1.0)
            for rank, score in rank_scores.items():
                combined_scores[rank] += score * weight

        if not combined_scores:
            return []

        # Find the fastest rank
        fastest_rank = max(combined_scores.items(), key=lambda x: x[1])[0]
        fastest_score = combined_scores[fastest_rank]

        # Identify stragglers
        stragglers = []
        for rank, score in combined_scores.items():
            if rank == fastest_rank:
                continue

            relative_slowdown = fastest_score / score if score > 0 else float('inf')
            if relative_slowdown >= threshold:
                stragglers.append(rank)

        return sorted(stragglers)

    def generate_report(
        self, step: Optional[int] = None, gather_on_rank0: Optional[bool] = None
    ) -> StragglerReport:
        """
        Generate a straggler detection report.
        Args:
            step: Training step (uses current_step if None)
            gather_on_rank0: Whether to gather on rank 0 (uses config.gather_on_rank0 if None)
        Returns:
            StragglerReport instance
        """
        if step is None:
            step = self.current_step

        if gather_on_rank0 is None:
            gather_on_rank0 = self.config.gather_on_rank0

        # Compute section scores
        section_scores = self.compute_all_section_scores()

        # Compute GPU scores
        gpu_scores = self.compute_gpu_scores()

        # Identify stragglers
        straggler_ranks = self.identify_stragglers(section_scores)

        # Create node names mapping
        node_names = {self.rank: self.node_name}

        # Create report
        report = StragglerReport(
            step=step,
            section_scores=section_scores,
            gpu_scores=gpu_scores,
            straggler_ranks=straggler_ranks,
            node_names=node_names,
        )

        report.timestamp = time.time()

        # In a real implementation, we would gather data from all ranks here
        # using all-reduce or similar collective operations

        return report

    def save_report(self, report: StragglerReport, filepath: str):
        """
        Save a report to file.
        Args:
            report: StragglerReport to save
            filepath: Path to save the report
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save report to {filepath}: {e}")

    def get_recent_section_time(self, section_name: str, num_samples: int = 1) -> Optional[float]:
        """
        Get the most recent timing for a section.
        Args:
            section_name: Name of the section
            num_samples: Number of recent samples to average
        Returns:
            Average time in seconds, or None if no data
        """
        timings = self.section_timings.get(section_name, [])
        if not timings:
            return None

        recent = timings[-num_samples:]
        if not recent:
            return None

        # Average the times
        total_time = 0.0
        count = 0
        for step, cpu_time, gpu_time in recent:
            elapsed = gpu_time if gpu_time is not None else cpu_time
            total_time += elapsed
            count += 1

        return total_time / count if count > 0 else None

    def get_section_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summaries for all sections.
        Returns:
            Dictionary mapping section name to statistics
        """
        stats = {}

        for section_name, timings in self.section_timings.items():
            if not timings:
                continue

            cpu_times = []
            gpu_times = []

            for step, cpu_time, gpu_time in timings:
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
        """Reset all collected data."""
        self.section_timings.clear()
        self.current_step = 0

    def is_enabled(self) -> bool:
        """Check if the detector is enabled."""
        return self.enabled

    def set_enabled(self, enabled: bool):
        """Enable or disable the detector."""
        self.enabled = enabled
