"""
Unit tests for straggler section profiling module.
"""

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
    """Test cases for SectionContext class."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    def test_basic_timing(self, mock_detector):
        """Test that SectionContext records timing correctly."""
        with SectionContext(mock_detector, "test_section"):
            time.sleep(0.01)  # Small sleep to ensure measurable time

        # Verify record_section was called
        mock_detector.record_section.assert_called_once()

        # Check call arguments
        call_args = mock_detector.record_section.call_args
        assert call_args.kwargs["name"] == "test_section"
        assert call_args.kwargs["cpu_time"] > 0
        assert call_args.kwargs["cpu_time"] >= 0.01  # At least sleep duration

    def test_timing_accuracy(self, mock_detector):
        """Test timing accuracy of SectionContext."""
        sleep_duration = 0.05

        with SectionContext(mock_detector, "timing_test"):
            time.sleep(sleep_duration)

        call_args = mock_detector.record_section.call_args
        recorded_time = call_args.kwargs["cpu_time"]

        # Should be close to sleep duration (within tolerance)
        assert recorded_time >= sleep_duration
        assert recorded_time < sleep_duration + 0.05  # Allow some overhead

    def test_section_name(self, mock_detector):
        """Test that section name is passed correctly."""
        with SectionContext(mock_detector, "my_custom_section"):
            pass

        call_args = mock_detector.record_section.call_args
        assert call_args.kwargs["name"] == "my_custom_section"

    def test_no_cuda_profiling_by_default(self, mock_detector):
        """Test that CUDA profiling is off by default."""
        with SectionContext(mock_detector, "test_section"):
            pass

        call_args = mock_detector.record_section.call_args
        # gpu_time should be None when CUDA profiling is disabled
        assert call_args.kwargs["gpu_time"] is None

    def test_exception_handling(self, mock_detector):
        """Test that exceptions are not suppressed."""
        with pytest.raises(ValueError):
            with SectionContext(mock_detector, "error_section"):
                raise ValueError("Test error")

        # record_section should still be called
        mock_detector.record_section.assert_called_once()

    def test_exception_does_not_affect_timing(self, mock_detector):
        """Test that exceptions don't prevent timing from being recorded."""
        try:
            with SectionContext(mock_detector, "error_section"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should have recorded timing before exception propagated
        call_args = mock_detector.record_section.call_args
        assert call_args.kwargs["cpu_time"] >= 0.01

    def test_nested_contexts(self, mock_detector):
        """Test nested SectionContext usage."""
        with SectionContext(mock_detector, "outer"):
            time.sleep(0.01)
            with SectionContext(mock_detector, "inner"):
                time.sleep(0.01)

        # Both contexts should have recorded
        assert mock_detector.record_section.call_count == 2

        # Check call order (inner finishes first)
        calls = mock_detector.record_section.call_args_list
        assert calls[0].kwargs["name"] == "inner"
        assert calls[1].kwargs["name"] == "outer"

    def test_context_returns_self(self, mock_detector):
        """Test that context manager returns self."""
        with SectionContext(mock_detector, "test") as ctx:
            assert isinstance(ctx, SectionContext)
            assert ctx.name == "test"


class TestOptionalSectionContext:
    """Test cases for OptionalSectionContext class."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    def test_enabled_profiles(self, mock_detector):
        """Test that enabled OptionalSectionContext profiles."""
        with OptionalSectionContext(mock_detector, "test", enabled=True):
            time.sleep(0.01)

        mock_detector.record_section.assert_called_once()

    def test_disabled_does_not_profile(self, mock_detector):
        """Test that disabled OptionalSectionContext does not profile."""
        with OptionalSectionContext(mock_detector, "test", enabled=False):
            time.sleep(0.01)

        mock_detector.record_section.assert_not_called()

    def test_default_is_enabled(self, mock_detector):
        """Test that OptionalSectionContext is enabled by default."""
        with OptionalSectionContext(mock_detector, "test"):
            pass

        mock_detector.record_section.assert_called_once()

    def test_exception_handling_enabled(self, mock_detector):
        """Test exception handling when enabled."""
        with pytest.raises(ValueError):
            with OptionalSectionContext(mock_detector, "test", enabled=True):
                raise ValueError("Test error")

        mock_detector.record_section.assert_called_once()

    def test_exception_handling_disabled(self, mock_detector):
        """Test exception handling when disabled."""
        with pytest.raises(ValueError):
            with OptionalSectionContext(mock_detector, "test", enabled=False):
                raise ValueError("Test error")

        mock_detector.record_section.assert_not_called()


class TestSectionProfiler:
    """Test cases for SectionProfiler class."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    @pytest.fixture
    def profiler(self, mock_detector):
        """Create a SectionProfiler for testing."""
        return SectionProfiler(mock_detector)

    def test_start_and_end_section(self, profiler, mock_detector):
        """Test starting and ending a section."""
        ctx = profiler.start_section("test_section")
        time.sleep(0.01)
        profiler.end_section("test_section")

        mock_detector.record_section.assert_called_once()
        call_args = mock_detector.record_section.call_args
        assert call_args.kwargs["name"] == "test_section"

    def test_multiple_sections(self, profiler, mock_detector):
        """Test multiple sections."""
        profiler.start_section("section1")
        profiler.start_section("section2")

        profiler.end_section("section2")
        profiler.end_section("section1")

        assert mock_detector.record_section.call_count == 2

    def test_start_duplicate_section_raises(self, profiler):
        """Test that starting a duplicate section raises error."""
        profiler.start_section("test")

        with pytest.raises(ValueError, match="already active"):
            profiler.start_section("test")

    def test_end_nonexistent_section_raises(self, profiler):
        """Test that ending a non-existent section raises error."""
        with pytest.raises(ValueError, match="not active"):
            profiler.end_section("nonexistent")

    def test_active_sections_tracking(self, profiler):
        """Test that active sections are tracked correctly."""
        assert len(profiler.active_sections) == 0

        profiler.start_section("section1")
        assert "section1" in profiler.active_sections
        assert len(profiler.active_sections) == 1

        profiler.start_section("section2")
        assert "section2" in profiler.active_sections
        assert len(profiler.active_sections) == 2

        profiler.end_section("section1")
        assert "section1" not in profiler.active_sections
        assert len(profiler.active_sections) == 1

        profiler.end_section("section2")
        assert len(profiler.active_sections) == 0


class TestCreateSectionDecorator:
    """Test cases for create_section_decorator function."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    def test_decorator_wraps_function(self, mock_detector):
        """Test that decorator wraps function correctly."""

        @create_section_decorator(mock_detector, "decorated_section")
        def my_function():
            return "result"

        result = my_function()

        assert result == "result"
        mock_detector.record_section.assert_called_once()

    def test_decorator_preserves_arguments(self, mock_detector):
        """Test that decorator preserves function arguments."""

        @create_section_decorator(mock_detector, "decorated_section")
        def my_function(a, b, c=None):
            return a + b + (c or 0)

        result = my_function(1, 2, c=3)

        assert result == 6
        mock_detector.record_section.assert_called_once()

    def test_decorator_records_timing(self, mock_detector):
        """Test that decorator records timing correctly."""

        @create_section_decorator(mock_detector, "timed_section")
        def slow_function():
            time.sleep(0.02)
            return "done"

        result = slow_function()

        assert result == "done"
        call_args = mock_detector.record_section.call_args
        assert call_args.kwargs["name"] == "timed_section"
        assert call_args.kwargs["cpu_time"] >= 0.02

    def test_decorator_handles_exceptions(self, mock_detector):
        """Test that decorator handles exceptions correctly."""

        @create_section_decorator(mock_detector, "error_section")
        def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_function()

        # Should still record timing
        mock_detector.record_section.assert_called_once()

    def test_multiple_decorated_functions(self, mock_detector):
        """Test multiple decorated functions."""

        @create_section_decorator(mock_detector, "section_a")
        def function_a():
            return "a"

        @create_section_decorator(mock_detector, "section_b")
        def function_b():
            return "b"

        function_a()
        function_b()

        assert mock_detector.record_section.call_count == 2


class TestSectionContextWithCuda:
    """Test cases for CUDA profiling (mocked)."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        detector = MagicMock()
        detector.record_section = MagicMock()
        return detector

    @patch('flagscale.runner.straggler.section.TORCH_AVAILABLE', True)
    @patch('flagscale.runner.straggler.section.torch')
    def test_cuda_profiling_enabled(self, mock_torch, mock_detector):
        """Test CUDA profiling when enabled."""
        # Setup mock
        mock_torch.cuda.is_available.return_value = True
        mock_event = MagicMock()
        mock_event.elapsed_time.return_value = 10.0  # 10ms
        mock_torch.cuda.Event.return_value = mock_event

        with SectionContext(mock_detector, "cuda_section", profile_cuda=True):
            time.sleep(0.01)

        mock_detector.record_section.assert_called_once()
        # Should have synchronized CUDA
        assert mock_torch.cuda.synchronize.called

    @patch('flagscale.runner.straggler.section.TORCH_AVAILABLE', False)
    def test_cuda_profiling_without_torch(self, mock_detector):
        """Test CUDA profiling gracefully handles missing torch."""
        with SectionContext(mock_detector, "section", profile_cuda=True):
            pass

        mock_detector.record_section.assert_called_once()
        call_args = mock_detector.record_section.call_args
        # gpu_time should be None when torch not available
        assert call_args.kwargs["gpu_time"] is None
