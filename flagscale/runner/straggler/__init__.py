"""
Straggler detection module for FlagScale.
This module provides infrastructure for detecting and reporting performance
stragglers in distributed training scenarios.
"""

from .comm import CommProfiler, CommStatsCollector, GlooCommHook, NCCLCommHook
from .config import StragglerConfig
from .detector import StragglerDetector
from .healthcheck import ElasticTrainingHealthChecker, NetworkHealthChecker
from .report import StragglerReport
from .section import (
    OptionalSectionContext,
    SectionContext,
    SectionProfiler,
    create_section_decorator,
)

__all__ = [
    "StragglerConfig",
    "StragglerDetector",
    "StragglerReport",
    "SectionContext",
    "OptionalSectionContext",
    "create_section_decorator",
    "SectionProfiler",
    "CommStatsCollector",
    "CommProfiler",
    "NCCLCommHook",
    "GlooCommHook",
    "NetworkHealthChecker",
    "ElasticTrainingHealthChecker",
]

__version__ = "0.1.0"
