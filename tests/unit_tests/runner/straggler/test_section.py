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

"""Unit tests for section profiling helpers."""

import time
from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.straggler.section import (
    OptionalSectionContext,
    SectionContext,
    SectionProfiler,
    create_section_decorator,
)


class TestSectionContext:
    @pytest.fixture
    def mock_detector(self):
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    def test_basic_timing(self, mock_detector):
        with SectionContext(mock_detector, "test_section"):
            time.sleep(0.01)
        call_args = mock_detector.record_section.call_args
        assert call_args.kwargs["name"] == "test_section"
        assert call_args.kwargs["cpu_time"] >= 0.01

    def test_exception_handling(self, mock_detector):
        with pytest.raises(ValueError), SectionContext(mock_detector, "error_section"):
            raise ValueError("Test error")
        mock_detector.record_section.assert_called_once()

    def test_nested_contexts(self, mock_detector):
        with SectionContext(mock_detector, "outer"):
            time.sleep(0.01)
            with SectionContext(mock_detector, "inner"):
                time.sleep(0.01)
        calls = mock_detector.record_section.call_args_list
        assert calls[0].kwargs["name"] == "inner"
        assert calls[1].kwargs["name"] == "outer"


class TestOptionalSectionContext:
    @pytest.fixture
    def mock_detector(self):
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    def test_enabled_profiles(self, mock_detector):
        with OptionalSectionContext(mock_detector, "test", enabled=True):
            time.sleep(0.01)
        mock_detector.record_section.assert_called_once()

    def test_disabled_does_not_profile(self, mock_detector):
        with OptionalSectionContext(mock_detector, "test", enabled=False):
            time.sleep(0.01)
        mock_detector.record_section.assert_not_called()


class TestSectionProfiler:
    def test_start_and_end_section(self):
        detector = MagicMock()
        detector.record_section = MagicMock()
        profiler = SectionProfiler(detector)
        profiler.start_section("test_section")
        time.sleep(0.01)
        profiler.end_section("test_section")
        detector.record_section.assert_called_once()

    def test_start_duplicate_section_raises(self):
        profiler = SectionProfiler(MagicMock())
        profiler.start_section("test")
        with pytest.raises(ValueError, match="already active"):
            profiler.start_section("test")


class TestCreateSectionDecorator:
    def test_decorator_wraps_function(self):
        detector = MagicMock()
        detector.record_section = MagicMock()

        @create_section_decorator(detector, "decorated_section")
        def my_function(a, b):
            return a + b

        assert my_function(1, 2) == 3
        detector.record_section.assert_called_once()


class TestSectionContextWithCuda:
    @patch("flagscale.runner.straggler.section.TORCH_AVAILABLE", True)
    @patch("flagscale.runner.straggler.section.torch")
    def test_cuda_profiling_enabled(self, mock_torch):
        detector = MagicMock()
        detector.record_section = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_event = MagicMock()
        mock_event.elapsed_time.return_value = 10.0
        mock_torch.cuda.Event.return_value = mock_event

        with SectionContext(detector, "cuda_section", profile_cuda=True):
            time.sleep(0.01)

        detector.record_section.assert_called_once()
        assert mock_torch.cuda.synchronize.called
