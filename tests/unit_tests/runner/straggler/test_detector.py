"""
Unit tests for straggler detector module.
"""

import json
import os
import tempfile

from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.straggler.config import StragglerConfig
from flagscale.runner.straggler.detector import StragglerDetector


class TestStragglerDetector:
    """Test cases for StragglerDetector class."""

    @pytest.fixture
    def default_config(self):
        """Create a default config for testing."""
        return StragglerConfig(
            enabled=True,
            profiling_interval=10,
            report_interval_steps=100,
            straggler_threshold=1.5,
            warmup_steps=10,
            monitor_sections=["forward_backward", "optimizer"],
        )

    @pytest.fixture
    def detector(self, default_config):
        """Create a detector instance for testing."""
        return StragglerDetector(
            config=default_config, rank=0, world_size=8, node_name="test-node:gpu0"
        )

    def test_init(self, default_config):
        """Test detector initialization."""
        detector = StragglerDetector(
            config=default_config, rank=0, world_size=8, node_name="test-node:gpu0"
        )

        assert detector.rank == 0
        assert detector.world_size == 8
        assert detector.node_name == "test-node:gpu0"
        assert detector.enabled is True
        assert detector.current_step == 0

    def test_init_default_node_name(self, default_config):
        """Test that node_name defaults to rank-{rank} if not provided."""
        detector = StragglerDetector(config=default_config, rank=3, world_size=8)

        assert detector.node_name == "rank-3"

    def test_is_enabled(self, detector):
        """Test is_enabled method."""
        assert detector.is_enabled() is True

        detector.set_enabled(False)
        assert detector.is_enabled() is False

    def test_set_enabled(self, detector):
        """Test set_enabled method."""
        detector.set_enabled(False)
        assert detector.enabled is False

        detector.set_enabled(True)
        assert detector.enabled is True

    def test_increment_step(self, detector):
        """Test step counter increment."""
        assert detector.current_step == 0

        detector.increment_step()
        assert detector.current_step == 1

        detector.increment_step()
        assert detector.current_step == 2

    def test_record_section(self, detector):
        """Test recording section timing data."""
        detector.record_section("forward_backward", cpu_time=0.5, gpu_time=0.45)

        assert "forward_backward" in detector.section_timings
        assert len(detector.section_timings["forward_backward"]) == 1
        assert detector.section_timings["forward_backward"][0] == (0, 0.5, 0.45)

    def test_record_section_unmonitored(self, detector):
        """Test that unmonitored sections are not recorded."""
        detector.record_section("unmonitored_section", cpu_time=0.5)

        assert "unmonitored_section" not in detector.section_timings

    def test_record_section_disabled(self, default_config):
        """Test that recording is skipped when disabled."""
        default_config.enabled = False
        detector = StragglerDetector(config=default_config, rank=0, world_size=1)

        detector.record_section("forward_backward", cpu_time=0.5)

        assert len(detector.section_timings) == 0

    def test_should_profile_warmup(self, detector):
        """Test should_profile respects warmup period."""
        # During warmup (step < warmup_steps=10)
        for step in range(10):
            detector.current_step = step
            assert detector.should_profile() is False

        # After warmup
        detector.current_step = 10
        assert detector.should_profile() is True

    def test_should_profile_interval(self, detector):
        """Test should_profile respects profiling interval."""
        # profiling_interval=10, warmup_steps=10
        # After warmup, should profile at step 10, 20, 30, ...

        detector.current_step = 10  # First step after warmup
        assert detector.should_profile() is True

        detector.current_step = 15
        assert detector.should_profile() is False

        detector.current_step = 20
        assert detector.should_profile() is True

    def test_should_report(self, detector):
        """Test should_report respects report interval."""
        # report_interval_steps=100

        detector.current_step = 0
        assert detector.should_report() is False

        detector.current_step = 50
        assert detector.should_report() is False

        detector.current_step = 100
        assert detector.should_report() is True

        detector.current_step = 200
        assert detector.should_report() is True

    def test_get_recent_section_time(self, detector):
        """Test getting recent section timing."""
        # Record some timings
        detector.record_section("forward_backward", cpu_time=0.1, step=1)
        detector.record_section("forward_backward", cpu_time=0.2, step=2)
        detector.record_section("forward_backward", cpu_time=0.3, step=3)

        # Get average of last 2 samples
        avg_time = detector.get_recent_section_time("forward_backward", num_samples=2)
        assert avg_time == pytest.approx(0.25, rel=1e-3)

        # Get most recent sample
        recent_time = detector.get_recent_section_time("forward_backward", num_samples=1)
        assert recent_time == pytest.approx(0.3, rel=1e-3)

    def test_get_recent_section_time_no_data(self, detector):
        """Test getting recent section time with no data."""
        result = detector.get_recent_section_time("forward_backward")
        assert result is None

    def test_get_section_statistics(self, detector):
        """Test getting section statistics."""
        # Record some timings
        detector.record_section("forward_backward", cpu_time=0.1)
        detector.record_section("forward_backward", cpu_time=0.2)
        detector.record_section("forward_backward", cpu_time=0.3)

        stats = detector.get_section_statistics()

        assert "forward_backward" in stats
        assert stats["forward_backward"]["count"] == 3
        assert stats["forward_backward"]["cpu_avg"] == pytest.approx(0.2, rel=1e-3)
        assert stats["forward_backward"]["cpu_min"] == pytest.approx(0.1, rel=1e-3)
        assert stats["forward_backward"]["cpu_max"] == pytest.approx(0.3, rel=1e-3)

    def test_reset(self, detector):
        """Test resetting detector state."""
        detector.record_section("forward_backward", cpu_time=0.5)
        detector.increment_step()
        detector.increment_step()

        assert len(detector.section_timings) > 0
        assert detector.current_step == 2

        detector.reset()

        assert len(detector.section_timings) == 0
        assert detector.current_step == 0


class TestStragglerDetection:
    """Test cases specifically for straggler detection logic."""

    @pytest.fixture
    def config(self):
        """Create config for straggler detection tests."""
        return StragglerConfig(
            enabled=True,
            straggler_threshold=1.5,
            monitor_sections=["forward_backward", "optimizer"],
        )

    @pytest.fixture
    def detector(self, config):
        """Create detector for straggler detection tests."""
        return StragglerDetector(config=config, rank=0, world_size=8, node_name="test-node:gpu0")

    def test_identify_stragglers_no_straggler(self, detector):
        """Test straggler identification when all ranks perform similarly."""
        # All ranks have similar timing (within threshold)
        section_times = {
            "forward_backward": {
                0: 0.100,
                1: 0.102,
                2: 0.098,
                3: 0.101,
                4: 0.099,
                5: 0.103,
                6: 0.097,
                7: 0.100,
            }
        }

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert len(stragglers) == 0

    def test_identify_stragglers_single_straggler(self, detector):
        """Test straggler identification with one slow rank."""
        # Rank 2 is 2x slower than fastest (exceeds 1.5x threshold)
        section_times = {
            "forward_backward": {
                0: 0.100,
                1: 0.100,
                2: 0.200,  # Straggler: 2x slower
                3: 0.100,
                4: 0.100,
                5: 0.100,
                6: 0.100,
                7: 0.100,
            }
        }

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert len(stragglers) == 1
        assert 2 in stragglers

    def test_identify_stragglers_multiple_stragglers(self, detector):
        """Test straggler identification with multiple slow ranks."""
        # Rank 2 and 5 are both stragglers
        section_times = {
            "forward_backward": {
                0: 0.100,
                1: 0.100,
                2: 0.200,  # Straggler: 2x slower
                3: 0.100,
                4: 0.100,
                5: 0.180,  # Straggler: 1.8x slower
                6: 0.100,
                7: 0.100,
            }
        }

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert len(stragglers) == 2
        assert 2 in stragglers
        assert 5 in stragglers

    def test_identify_stragglers_at_threshold(self, detector):
        """Test straggler identification at threshold boundary."""
        # Rank 2 is slightly above 1.5x threshold (1.51x)
        section_times = {
            "forward_backward": {
                0: 0.100,
                1: 0.100,
                2: 0.151,  # Slightly above 1.5x threshold
                3: 0.100,
            }
        }

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert 2 in stragglers

    def test_identify_stragglers_below_threshold(self, detector):
        """Test straggler identification below threshold."""
        # Rank 2 is 1.4x slower (below 1.5x threshold)
        section_times = {
            "forward_backward": {0: 0.100, 1: 0.100, 2: 0.140, 3: 0.100}  # Below 1.5x threshold
        }

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert 2 not in stragglers

    def test_identify_stragglers_multiple_sections(self, detector):
        """Test straggler identification with multiple sections."""
        # Rank 3 is slow in forward_backward, rank 5 is slow in optimizer
        section_times = {
            "forward_backward": {0: 0.100, 1: 0.100, 2: 0.100, 3: 0.200},  # Slow here
            "optimizer": {0: 0.010, 1: 0.010, 2: 0.010, 3: 0.010},
        }

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert 3 in stragglers

    def test_identify_stragglers_empty_data(self, detector):
        """Test straggler identification with empty data."""
        stragglers = detector._identify_stragglers_from_times({})
        assert len(stragglers) == 0

    def test_identify_stragglers_custom_threshold(self, config):
        """Test straggler identification with custom threshold."""
        config.straggler_threshold = 2.0
        detector = StragglerDetector(config=config, rank=0, world_size=4)

        # Rank 2 is 1.8x slower (below 2.0 threshold, not a straggler)
        section_times = {"forward_backward": {0: 0.100, 1: 0.100, 2: 0.180, 3: 0.100}}

        stragglers = detector._identify_stragglers_from_times(section_times)
        assert 2 not in stragglers

        # Rank 3 is 2.5x slower (above 2.0 threshold, is a straggler)
        section_times["forward_backward"][3] = 0.250
        stragglers = detector._identify_stragglers_from_times(section_times)
        assert 3 in stragglers


class TestStragglerDetectorReportGeneration:
    """Test cases for report generation without distributed environment."""

    @pytest.fixture
    def config(self):
        """Create config for report tests."""
        return StragglerConfig(
            enabled=True,
            straggler_threshold=1.5,
            monitor_sections=["forward_backward", "optimizer"],
        )

    @pytest.fixture
    def detector(self, config):
        """Create detector for report tests."""
        return StragglerDetector(
            config=config,
            rank=0,
            world_size=1,  # Single rank for non-distributed testing
            node_name="test-node:gpu0",
        )

    def test_generate_report_local(self, detector):
        """Test report generation with local data only."""
        # Record some data
        for i in range(5):
            detector.record_section("forward_backward", cpu_time=0.1 + i * 0.01)
            detector.record_section("optimizer", cpu_time=0.01)
            detector.increment_step()

        detector.current_step = 10
        report = detector.generate_report(step=10)

        assert report.step == 10
        assert report.node_names is not None
        assert 0 in report.node_names

    def test_save_report(self, detector):
        """Test saving report to file."""
        # Record some data
        detector.record_section("forward_backward", cpu_time=0.1)
        detector.increment_step()

        report = detector.generate_report(step=1)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            detector.save_report(report, temp_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert "step" in data
            assert "section_scores" in data
            assert "node_names" in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestStragglerDetectorWithMockedDistributed:
    """Test cases for distributed functionality with mocked torch.distributed."""

    @pytest.fixture
    def config(self):
        """Create config for distributed tests."""
        return StragglerConfig(
            enabled=True,
            straggler_threshold=1.5,
            monitor_sections=["forward_backward", "optimizer"],
        )

    @patch('flagscale.runner.straggler.detector.dist')
    @patch('flagscale.runner.straggler.detector.TORCH_DISTRIBUTED_AVAILABLE', True)
    def test_gather_section_times_across_ranks(self, mock_dist, config):
        """Test gathering section times from all ranks."""
        # Setup mock
        mock_dist.is_initialized.return_value = True

        # Create detector
        detector = StragglerDetector(config=config, rank=0, world_size=4, node_name="node0:gpu0")

        # Record local timing
        detector.record_section("forward_backward", cpu_time=0.1)

        # Mock all_gather to simulate gathering from 4 ranks
        def mock_all_gather(gathered_list, local_tensor):
            # Simulate different times from different ranks
            times = [0.10, 0.11, 0.15, 0.12]
            for i, t in enumerate(times):
                gathered_list[i].fill_(t)

        mock_dist.all_gather.side_effect = mock_all_gather

        # Call gather method
        result = detector._gather_section_times_across_ranks()

        assert "forward_backward" in result
        assert len(result["forward_backward"]) == 4

    @patch('flagscale.runner.straggler.detector.dist')
    @patch('flagscale.runner.straggler.detector.TORCH_DISTRIBUTED_AVAILABLE', True)
    def test_gather_node_names_across_ranks(self, mock_dist, config):
        """Test gathering node names from all ranks."""
        # Setup mock
        mock_dist.is_initialized.return_value = True

        detector = StragglerDetector(config=config, rank=0, world_size=4, node_name="node0:gpu0")

        # Mock all_gather_object
        def mock_all_gather_object(output_list, local_obj):
            names = ["node0:gpu0", "node0:gpu1", "node1:gpu0", "node1:gpu1"]
            for i, name in enumerate(names):
                output_list[i] = name

        mock_dist.all_gather_object.side_effect = mock_all_gather_object

        result = detector._gather_node_names_across_ranks()

        assert len(result) == 4
        assert result[0] == "node0:gpu0"
        assert result[2] == "node1:gpu0"
