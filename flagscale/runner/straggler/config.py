"""
Straggler detection configuration module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class StragglerConfig:
    """Configuration dataclass for straggler detection."""

    # Enable/disable straggler detection
    enabled: bool = True

    # Score computation mode
    # - "relative": compute scores relative to fastest ranks
    # - "individual": compute individual section scores
    # - "all": compute all scores
    scores_to_compute: Literal["relative", "individual", "all"] = "all"

    # Gather all statistics on rank 0
    gather_on_rank0: bool = True

    # Profile every N steps (sampling interval)
    profiling_interval: int = 10

    # Report interval in steps
    report_interval_steps: int = 100

    # Node name for identification (optional)
    node_name: Optional[str] = None

    # List of sections to monitor
    # Common sections: dataloader, forward, backward, optimizer
    monitor_sections: List[str] = field(
        default_factory=lambda: ["dataloader", "forward", "backward", "optimizer"]
    )

    # Enable communication logging and profiling
    enable_comm_logging: bool = True

    # Enable GPU profiling and timing
    enable_gpu_profile: bool = True

    # Threshold for identifying stragglers (relative slowdown factor)
    straggler_threshold: float = 1.5

    # Maximum number of stragglers to report
    max_stragglers_to_report: int = 5

    # Communication backend to monitor
    # Options: "nccl", "gloo", "mpi", "all"
    comm_backend: Literal["nccl", "gloo", "mpi", "all"] = "all"

    # Sample size for statistical analysis
    sample_size: int = 100

    # Ignore first N steps (warmup)
    warmup_steps: int = 10
