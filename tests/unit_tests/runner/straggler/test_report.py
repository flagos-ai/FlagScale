"""
Unit tests for straggler report module.
"""

import json

import pytest

from flagscale.runner.straggler.report import StragglerReport


class TestStragglerReport:
    """Test cases for StragglerReport class."""

    @pytest.fixture
    def sample_section_scores(self):
        """Create sample section scores data."""
        return {
            "forward_backward": {0: 0.100, 1: 0.105, 2: 0.098, 3: 0.102},
            "optimizer": {0: 0.010, 1: 0.011, 2: 0.009, 3: 0.010},
        }

    @pytest.fixture
    def sample_gpu_scores(self):
        """Create sample GPU scores data."""
        return {0: 10.0, 1: 9.5, 2: 10.2, 3: 9.8}

    @pytest.fixture
    def sample_node_names(self):
        """Create sample node names data."""
        return {0: "node0:gpu0", 1: "node0:gpu1", 2: "node1:gpu0", 3: "node1:gpu1"}

    def test_init_basic(self):
        """Test basic report initialization."""
        report = StragglerReport(step=100)

        assert report.step == 100
        assert report.section_scores == {}
        assert report.gpu_scores == {}
        assert report.straggler_ranks == []
        assert report.node_names == {}

    def test_init_with_data(self, sample_section_scores, sample_gpu_scores, sample_node_names):
        """Test report initialization with data."""
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[2],
            node_names=sample_node_names,
        )

        assert report.step == 100
        assert report.section_scores == sample_section_scores
        assert report.gpu_scores == sample_gpu_scores
        assert report.straggler_ranks == [2]
        assert report.node_names == sample_node_names

    def test_to_dict(self, sample_section_scores, sample_gpu_scores, sample_node_names):
        """Test conversion to dictionary."""
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[2],
            node_names=sample_node_names,
        )

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result["step"] == 100
        assert result["section_scores"] == sample_section_scores
        assert result["gpu_scores"] == sample_gpu_scores
        assert result["straggler_ranks"] == [2]
        assert result["node_names"] == sample_node_names
        assert "timestamp" in result

    def test_to_dict_json_serializable(
        self, sample_section_scores, sample_gpu_scores, sample_node_names
    ):
        """Test that to_dict result is JSON serializable."""
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[2, 3],
            node_names=sample_node_names,
        )

        result = report.to_dict()

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["step"] == 100

    def test_to_text_no_stragglers(
        self, sample_section_scores, sample_gpu_scores, sample_node_names
    ):
        """Test text output when no stragglers detected."""
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[],
            node_names=sample_node_names,
        )

        text = report.to_text()

        assert "Step 100" in text
        assert "No stragglers detected" in text
        assert "forward_backward" in text
        assert "optimizer" in text

    def test_to_text_with_stragglers(
        self, sample_section_scores, sample_gpu_scores, sample_node_names
    ):
        """Test text output when stragglers are detected."""
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[2, 3],
            node_names=sample_node_names,
        )

        text = report.to_text()

        assert "Step 100" in text
        assert "STRAGGLERS DETECTED" in text or "Straggler" in text
        # Should mention straggler ranks
        assert "2" in text
        assert "3" in text

    def test_to_text_section_timings(self, sample_node_names):
        """Test that section timings are properly formatted."""
        section_scores = {"forward_backward": {0: 0.100, 1: 0.150, 2: 0.120, 3: 0.110}}

        report = StragglerReport(
            step=100, section_scores=section_scores, node_names=sample_node_names
        )

        text = report.to_text()

        # Should contain section name
        assert "forward_backward" in text
        # Should contain timing statistics
        assert "Min" in text or "min" in text
        assert "Max" in text or "max" in text

    def test_to_text_gpu_scores(self, sample_gpu_scores, sample_node_names):
        """Test that GPU scores are properly formatted."""
        report = StragglerReport(
            step=100, section_scores={}, gpu_scores=sample_gpu_scores, node_names=sample_node_names
        )

        text = report.to_text()

        # Should contain GPU scores section
        assert "GPU" in text or "gpu" in text
        # Should contain node names
        assert "node0:gpu0" in text

    def test_to_text_empty_report(self):
        """Test text output for empty report."""
        report = StragglerReport(step=100)

        text = report.to_text()

        assert "Step 100" in text
        # Should still be valid text
        assert isinstance(text, str)
        assert len(text) > 0


class TestStragglerReportStatistics:
    """Test cases for report statistics calculations."""

    def test_section_timing_statistics(self):
        """Test that section timing statistics are calculated correctly."""
        section_scores = {
            "forward_backward": {0: 0.100, 1: 0.200, 2: 0.150, 3: 0.150}  # Min  # Max
        }

        report = StragglerReport(step=100, section_scores=section_scores)

        text = report.to_text()

        # Check that min/max/avg are displayed
        # Values should be converted to milliseconds in output
        assert "100" in text or "0.1" in text  # Min
        assert "200" in text or "0.2" in text  # Max

    def test_slowdown_calculation(self):
        """Test slowdown ratio calculation in report."""
        # Slowdown = max_time / min_time = 0.200 / 0.100 = 2.0x
        section_scores = {"forward_backward": {0: 0.100, 1: 0.200}}

        report = StragglerReport(step=100, section_scores=section_scores)

        text = report.to_text()

        # Should contain slowdown ratio
        assert "2.0" in text or "Slowdown" in text


class TestStragglerReportEdgeCases:
    """Test edge cases for StragglerReport."""

    def test_single_rank(self):
        """Test report with single rank."""
        report = StragglerReport(
            step=100,
            section_scores={"forward_backward": {0: 0.100}},
            gpu_scores={0: 10.0},
            node_names={0: "single-node:gpu0"},
        )

        text = report.to_text()
        result = report.to_dict()

        assert isinstance(text, str)
        assert result["step"] == 100

    def test_large_number_of_ranks(self):
        """Test report with many ranks."""
        num_ranks = 64
        section_scores = {"forward_backward": {i: 0.1 + i * 0.001 for i in range(num_ranks)}}
        gpu_scores = {i: 10.0 - i * 0.1 for i in range(num_ranks)}
        node_names = {i: f"node{i // 8}:gpu{i % 8}" for i in range(num_ranks)}

        report = StragglerReport(
            step=100, section_scores=section_scores, gpu_scores=gpu_scores, node_names=node_names
        )

        text = report.to_text()
        result = report.to_dict()

        assert isinstance(text, str)
        assert len(result["section_scores"]["forward_backward"]) == num_ranks

    def test_zero_timing_values(self):
        """Test report with zero timing values."""
        report = StragglerReport(step=100, section_scores={"forward_backward": {0: 0.0, 1: 0.0}})

        # Should not raise
        text = report.to_text()
        result = report.to_dict()

        assert isinstance(text, str)
        assert isinstance(result, dict)

    def test_very_large_timing_values(self):
        """Test report with very large timing values."""
        report = StragglerReport(
            step=100, section_scores={"forward_backward": {0: 1000.0, 1: 2000.0}}
        )

        text = report.to_text()
        result = report.to_dict()

        assert isinstance(text, str)
        assert isinstance(result, dict)

    def test_timestamp_is_set(self):
        """Test that timestamp is set in report."""
        report = StragglerReport(step=100)
        report.timestamp = 1234567890.0

        result = report.to_dict()

        assert result["timestamp"] == 1234567890.0
