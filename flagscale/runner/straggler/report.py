"""
Straggler detection report structures and serialization.
"""

from typing import Any, Dict, List, Optional


class StragglerReport:
    """
    Lightweight report structure for straggler detection results.
    Attributes:
        step: Current training step
        section_scores: Dict mapping section names to {rank: score}
        comm_stats: Communication statistics per operation
        gpu_scores: Dict mapping rank to GPU performance score
        straggler_ranks: List of ranks identified as stragglers
        node_names: Optional dict mapping rank to node name
        timestamp: Report generation timestamp
    """

    def __init__(
        self,
        step: int,
        section_scores: Optional[Dict[str, Dict[int, float]]] = None,
        comm_stats: Optional[Dict[str, Any]] = None,
        gpu_scores: Optional[Dict[int, float]] = None,
        straggler_ranks: Optional[List[int]] = None,
        node_names: Optional[Dict[int, str]] = None,
    ):
        self.step = step
        self.section_scores = section_scores or {}
        self.comm_stats = comm_stats or {}
        self.gpu_scores = gpu_scores or {}
        self.straggler_ranks = straggler_ranks or []
        self.node_names = node_names or {}
        self.timestamp = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary for serialization.
        Returns:
            Dictionary representation of the report
        """
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
        """
        Format report as human-readable text for logging.
        Returns:
            Text representation of the report
        """
        lines = []
        lines.append(f"=== Straggler Report at Step {self.step} ===")

        # Straggler ranks
        if self.straggler_ranks:
            lines.append(f"\nDetected Stragglers: {self.straggler_ranks}")
            for rank in self.straggler_ranks:
                node_name = self.node_names.get(rank, f"rank-{rank}")
                lines.append(f"  - Rank {rank} ({node_name})")
        else:
            lines.append("\nNo stragglers detected.")

        # Section scores summary
        if self.section_scores:
            lines.append("\nSection Scores:")
            for section_name, rank_scores in self.section_scores.items():
                lines.append(f"\n  {section_name}:")
                for rank, score in sorted(rank_scores.items()):
                    node_name = self.node_names.get(rank, f"rank-{rank}")
                    lines.append(f"    Rank {rank} ({node_name}): {score:.4f}")

        # GPU scores summary
        if self.gpu_scores:
            lines.append("\nGPU Performance Scores:")
            for rank, score in sorted(self.gpu_scores.items()):
                node_name = self.node_names.get(rank, f"rank-{rank}")
                lines.append(f"  Rank {rank} ({node_name}): {score:.4f}")

        # Communication stats summary
        if self.comm_stats:
            lines.append("\nCommunication Statistics:")
            for op_name, stats in self.comm_stats.items():
                if isinstance(stats, dict):
                    lines.append(f"\n  {op_name}:")
                    for key, value in stats.items():
                        if isinstance(value, float):
                            lines.append(f"    {key}: {value:.4f}")
                        else:
                            lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def identify_stragglers(self, threshold: float = 1.5) -> List[int]:
        """
        Identify straggler ranks based on performance scores.
        Args:
            threshold: Relative slowdown factor (e.g., 1.5 means 50% slower than fastest)
        Returns:
            List of ranks identified as stragglers
        """
        stragglers = []

        # Check each section for stragglers
        for section_name, rank_scores in self.section_scores.items():
            if not rank_scores:
                continue

            # Find the fastest rank (highest score)
            fastest_rank = max(rank_scores.items(), key=lambda x: x[1])

            # Identify ranks slower than threshold
            for rank, score in rank_scores.items():
                if rank == fastest_rank[0]:
                    continue  # Skip the fastest rank

                # If score is significantly lower, mark as straggler
                relative_slowdown = fastest_rank[1] / score if score > 0 else float('inf')
                if relative_slowdown >= threshold and rank not in stragglers:
                    stragglers.append(rank)

        # Also check GPU scores if available
        if self.gpu_scores:
            gpu_stragglers = self.identify_gpu_stragglers(threshold)
            for rank in gpu_stragglers:
                if rank not in stragglers:
                    stragglers.append(rank)

        # Sort for consistent output
        return sorted(stragglers)

    def identify_gpu_stragglers(self, threshold: float = 1.5) -> List[int]:
        """
        Identify stragglers specifically based on GPU performance.
        Args:
            threshold: Relative slowdown factor
        Returns:
            List of GPU straggler ranks
        """
        if not self.gpu_scores:
            return []

        # Find the fastest rank
        fastest_rank = max(self.gpu_scores.items(), key=lambda x: x[1])
        stragglers = []

        for rank, score in self.gpu_scores.items():
            if rank == fastest_rank[0]:
                continue

            relative_slowdown = fastest_rank[1] / score if score > 0 else float('inf')
            if relative_slowdown >= threshold:
                stragglers.append(rank)

        return sorted(stragglers)

    def get_worst_sections(self, top_k: int = 3) -> List[tuple]:
        """
        Get the top-k worst performing sections across all ranks.
        Args:
            top_k: Number of worst sections to return
        Returns:
            List of tuples (section_name, worst_rank, slowest_score, fastest_score)
        """
        section_performance = []

        for section_name, rank_scores in self.section_scores.items():
            if len(rank_scores) < 2:
                continue

            scores = list(rank_scores.values())
            fastest_score = max(scores)
            slowest_score = min(scores)

            worst_rank = min(rank_scores.items(), key=lambda x: x[1])[0]
            section_performance.append((section_name, worst_rank, slowest_score, fastest_score))

        # Sort by relative slowdown (largest first)
        section_performance.sort(
            key=lambda x: x[2] / x[3] if x[3] > 0 else float('inf'), reverse=True
        )

        return section_performance[:top_k]
