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

"""Serialization and text formatting for straggler reports."""

from typing import Any


class StragglerReport:
    """A lightweight report object for straggler findings."""

    def __init__(
        self,
        step: int,
        section_scores: dict[str, dict[int, float]] | None = None,
        comm_stats: dict[str, Any] | None = None,
        gpu_scores: dict[int, float] | None = None,
        straggler_ranks: list[int] | None = None,
        node_names: dict[int, str] | None = None,
    ):
        self.step = step
        self.section_scores = section_scores or {}
        self.comm_stats = comm_stats or {}
        self.gpu_scores = gpu_scores or {}
        self.straggler_ranks = straggler_ranks or []
        self.node_names = node_names or {}
        self.timestamp = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "section_scores": self.section_scores,
            "comm_stats": self.comm_stats,
            "gpu_scores": self.gpu_scores,
            "straggler_ranks": self.straggler_ranks,
            "node_names": self.node_names,
            "timestamp": self.timestamp,
        }

    def to_text(self) -> str:
        lines = [f"=== Straggler Report at Step {self.step} ==="]

        if self.straggler_ranks:
            lines.append("")
            lines.append(f"Detected stragglers: {self.straggler_ranks}")
            for rank in self.straggler_ranks:
                node_name = self.node_names.get(rank, f"rank-{rank}")
                lines.append(f"  Rank {rank} ({node_name})")
        else:
            lines.append("")
            lines.append("No stragglers detected.")

        if self.section_scores:
            lines.append("")
            lines.append("Section Timings (ms):")
            for section_name, rank_times in self.section_scores.items():
                if not rank_times:
                    continue
                times = list(rank_times.values())
                min_time = min(times) * 1000
                max_time = max(times) * 1000
                avg_time = sum(times) / len(times) * 1000
                slowdown = max_time / min_time if min_time > 0 else 1.0
                lines.append("")
                lines.append(f"  {section_name}:")
                lines.append(
                    f"    Min: {min_time:.2f}ms, Max: {max_time:.2f}ms, Avg: {avg_time:.2f}ms, Slowdown: {slowdown:.2f}x"
                )
                for rank, time_val in sorted(rank_times.items()):
                    node_name = self.node_names.get(rank, f"rank-{rank}")
                    lines.append(f"    Rank {rank} ({node_name}): {time_val * 1000:.2f}ms")

        if self.gpu_scores:
            lines.append("")
            lines.append("GPU Performance Scores (higher is faster):")
            for rank, score in sorted(self.gpu_scores.items()):
                node_name = self.node_names.get(rank, f"rank-{rank}")
                lines.append(f"  Rank {rank} ({node_name}): {score:.4f}")

        if self.comm_stats:
            lines.append("")
            lines.append("Communication Statistics:")
            for op_name, stats in self.comm_stats.items():
                lines.append(f"  {op_name}:")
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        if isinstance(value, float):
                            lines.append(f"    {key}: {value:.4f}")
                        else:
                            lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def identify_stragglers(self, threshold: float = 1.5) -> list[int]:
        stragglers = []
        for _, rank_scores in self.section_scores.items():
            if not rank_scores:
                continue
            fastest_rank, fastest_time = min(rank_scores.items(), key=lambda item: item[1])
            for rank, elapsed_time in rank_scores.items():
                if rank == fastest_rank:
                    continue
                if fastest_time <= 0:
                    continue
                relative_slowdown = elapsed_time / fastest_time
                if relative_slowdown >= threshold and rank not in stragglers:
                    stragglers.append(rank)

        for rank in self.identify_gpu_stragglers(threshold):
            if rank not in stragglers:
                stragglers.append(rank)
        return sorted(stragglers)

    def identify_gpu_stragglers(self, threshold: float = 1.5) -> list[int]:
        if not self.gpu_scores:
            return []

        fastest_rank, fastest_score = max(self.gpu_scores.items(), key=lambda item: item[1])
        stragglers = []
        for rank, score in self.gpu_scores.items():
            if rank == fastest_rank:
                continue
            relative_slowdown = fastest_score / score if score > 0 else float("inf")
            if relative_slowdown >= threshold:
                stragglers.append(rank)
        return sorted(stragglers)

    def get_worst_sections(self, top_k: int = 3) -> list[tuple]:
        section_performance = []
        for section_name, rank_scores in self.section_scores.items():
            if len(rank_scores) < 2:
                continue
            scores = list(rank_scores.values())
            slowest_score = max(scores)
            fastest_score = min(scores)
            worst_rank = max(rank_scores.items(), key=lambda item: item[1])[0]
            section_performance.append((section_name, worst_rank, slowest_score, fastest_score))

        section_performance.sort(
            key=lambda item: item[2] / item[3] if item[3] > 0 else float("inf"),
            reverse=True,
        )
        return section_performance[:top_k]
