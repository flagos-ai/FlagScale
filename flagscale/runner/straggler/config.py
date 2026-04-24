"""Configuration types for FlagScale straggler detection."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StragglerConfig:
    """Runtime configuration for straggler detection."""

    enabled: bool = True
    scores_to_compute: Literal["relative", "individual", "all"] = "all"
    gather_on_rank0: bool = True
    profiling_interval: int = 10
    report_interval_steps: int = 100
    node_name: str | None = None
    monitor_sections: list[str] = field(
        default_factory=lambda: [
            "dataloader",
            "forward",
            "backward",
            "optimizer",
            "forward_backward",
        ]
    )
    enable_comm_logging: bool = True
    enable_gpu_profile: bool = True
    straggler_threshold: float = 1.5
    max_stragglers_to_report: int = 5
    comm_backend: Literal["nccl", "gloo", "mpi", "all"] = "all"
    sample_size: int = 100
    warmup_steps: int = 10
