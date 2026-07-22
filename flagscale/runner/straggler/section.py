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

"""Section-level timing utilities for straggler detection."""

import time
from functools import wraps

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class SectionContext:
    """Context manager that records a timed section."""

    def __init__(self, detector, name: str, profile_cuda: bool = False):
        self.detector = detector
        self.name = name
        self.profile_cuda = profile_cuda
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.cuda_start_event = None
        self.cuda_end_event = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.profile_cuda and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_start_event.record()
        else:
            self.cuda_start_event = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        cpu_elapsed = self.end_time - self.start_time

        cuda_elapsed = None
        if self.cuda_start_event is not None and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.cuda_end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_end_event.record()
            self.cuda_end_event.synchronize()
            cuda_elapsed = self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000.0

        if hasattr(self.detector, "record_section"):
            self.detector.record_section(
                name=self.name,
                cpu_time=cpu_elapsed,
                gpu_time=cuda_elapsed,
            )
        return False


class OptionalSectionContext:
    """Conditionally enable section profiling."""

    def __init__(
        self,
        detector,
        name: str,
        enabled: bool = True,
        profile_cuda: bool = False,
    ):
        self.detector = detector
        self.name = name
        self.enabled = enabled
        self.profile_cuda = profile_cuda
        self.context: SectionContext | None = None

    def __enter__(self):
        if self.enabled:
            self.context = SectionContext(
                detector=self.detector,
                name=self.name,
                profile_cuda=self.profile_cuda,
            )
            return self.context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context is not None:
            return self.context.__exit__(exc_type, exc_val, exc_tb)
        return False


def create_section_decorator(detector, section_name: str, profile_cuda: bool = False):
    """Create a decorator that measures function runtime as a section."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with SectionContext(detector, section_name, profile_cuda):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class SectionProfiler:
    """Manage multiple active section contexts."""

    def __init__(self, detector):
        self.detector = detector
        self.active_sections: dict[str, SectionContext] = {}

    def start_section(self, name: str, profile_cuda: bool = False) -> SectionContext:
        if name in self.active_sections:
            raise ValueError(f"Section '{name}' is already active")

        context = SectionContext(self.detector, name, profile_cuda)
        self.active_sections[name] = context
        context.__enter__()
        return context

    def end_section(self, name: str):
        if name not in self.active_sections:
            raise ValueError(f"Section '{name}' is not active")

        context = self.active_sections.pop(name)
        context.__exit__(None, None, None)
