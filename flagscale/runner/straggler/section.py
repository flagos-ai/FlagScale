"""
Section context manager for straggler detection.
"""

import time

from typing import Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SectionContext:
    """
    Context manager for profiling code sections during training.
    This class wraps code sections (e.g., dataloader, forward, backward)
    to measure execution time and report to a StragglerDetector.
    Example:
        with SectionContext(detector, "forward", profile_cuda=True):
            output = model(input)
    Args:
        detector: StragglerDetector instance to report to
        name: Name of the section being profiled
        profile_cuda: Whether to profile CUDA events (if available)
    """

    def __init__(self, detector, name: str, profile_cuda: bool = False):
        self.detector = detector
        self.name = name
        self.profile_cuda = profile_cuda
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cuda_start_event: Optional = None
        self.cuda_end_event: Optional = None

    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.perf_counter()

        if self.profile_cuda and TORCH_AVAILABLE and torch.cuda.is_available():
            # Record CUDA events for more precise GPU timing
            torch.cuda.synchronize()
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_start_event.record()
        else:
            self.cuda_start_event = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop timing and report to detector when exiting the context.
        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self.end_time = time.perf_counter()

        # Calculate elapsed time
        cpu_elapsed = self.end_time - self.start_time

        # Get CUDA elapsed time if available
        cuda_elapsed = None
        if self.cuda_start_event is not None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
                self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                self.cuda_end_event.record()
                cuda_elapsed = self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000.0
            else:
                cuda_elapsed = cpu_elapsed

        # Report to detector
        if hasattr(self.detector, 'record_section'):
            self.detector.record_section(
                name=self.name, cpu_time=cpu_elapsed, gpu_time=cuda_elapsed
            )

        # Don't suppress exceptions
        return False


class OptionalSectionContext:
    """
    Optional context manager that only profiles if enabled.
    This is a wrapper around SectionContext that checks a flag before
    creating the profiling context. This avoids overhead when profiling
    is disabled.
    Example:
        with OptionalSectionContext(detector, "forward", enabled=True):
            output = model(input)
    """

    def __init__(self, detector, name: str, enabled: bool = True, profile_cuda: bool = False):
        self.detector = detector
        self.name = name
        self.enabled = enabled
        self.profile_cuda = profile_cuda
        self.context: Optional[SectionContext] = None

    def __enter__(self):
        if self.enabled:
            self.context = SectionContext(
                detector=self.detector, name=self.name, profile_cuda=self.profile_cuda
            )
            return self.context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context is not None:
            return self.context.__exit__(exc_type, exc_val, exc_tb)
        return False


def create_section_decorator(detector, section_name: str, profile_cuda: bool = False):
    """
    Create a decorator for profiling functions as sections.
    Args:
        detector: StragglerDetector instance
        section_name: Name for the section
        profile_cuda: Whether to profile CUDA
    Returns:
        Decorator function
    Example:
        @create_section_decorator(detector, "preprocess", profile_cuda=True)
        def preprocess_data(data):
            return processed_data
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with SectionContext(detector, section_name, profile_cuda):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class SectionProfiler:
    """
    High-level section profiling manager.
    Provides a simple interface to manage multiple sections and their contexts.
    """

    def __init__(self, detector):
        self.detector = detector
        self.active_sections: Dict[str, SectionContext] = {}

    def start_section(self, name: str, profile_cuda: bool = False) -> SectionContext:
        """
        Start a named section and return its context.
        Args:
            name: Name of the section
            profile_cuda: Whether to profile CUDA
        Returns:
            SectionContext for manual exit
        Example:
            profiler = SectionProfiler(detector)
            ctx = profiler.start_section("data_loading")
            # ... do work ...
            ctx.__exit__(None, None, None)
        """
        if name in self.active_sections:
            raise ValueError(f"Section '{name}' is already active")

        context = SectionContext(self.detector, name, profile_cuda)
        self.active_sections[name] = context
        context.__enter__()
        return context

    def end_section(self, name: str):
        """
        End a named section.
        Args:
            name: Name of the section to end
        """
        if name not in self.active_sections:
            raise ValueError(f"Section '{name}' is not active")

        context = self.active_sections.pop(name)
        context.__exit__(None, None, None)

    def __enter__(self, name: str = None):
        """Enter a section (for use as context manager)."""
        if name is None:
            raise ValueError("Must provide section name")
        self.start_section(name)
        return self

    def __exit__(self, name: str = None, *args):
        """Exit a section (for use as context manager)."""
        if name is None:
            raise ValueError("Must provide section name")
        self.end_section(name)
