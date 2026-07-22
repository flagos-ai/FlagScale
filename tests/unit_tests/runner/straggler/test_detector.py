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

"""Unit tests for straggler detector."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

import flagscale.runner.straggler.detector as detector_module
from flagscale.runner.straggler.config import StragglerConfig
from flagscale.runner.straggler.detector import StragglerDetector


class TestStragglerDetector:
    @pytest.fixture
    def default_config(self):
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
        return StragglerDetector(
            config=default_config,
            rank=0,
            world_size=8,
            node_name="test-node:gpu0",
        )

    def test_init(self, default_config):
        detector = StragglerDetector(
            config=default_config,
            rank=0,
            world_size=8,
            node_name="test-node:gpu0",
        )
        assert detector.rank == 0
        assert detector.world_size == 8
        assert detector.node_name == "test-node:gpu0"
        assert detector.enabled is True
        assert detector.current_step == 0

    def test_init_default_node_name(self, default_config):
        detector = StragglerDetector(config=default_config, rank=3, world_size=8)
        assert detector.node_name == "rank-3"

    def test_set_enabled(self, detector):
        detector.set_enabled(False)
        assert detector.is_enabled() is False
        detector.set_enabled(True)
        assert detector.is_enabled() is True

    def test_increment_step(self, detector):
        detector.increment_step()
        detector.increment_step()
        assert detector.current_step == 2

    def test_record_section(self, detector):
        detector.record_section("forward_backward", cpu_time=0.5, gpu_time=0.45)
        assert detector.section_timings["forward_backward"][0] == (0, 0.5, 0.45)

    def test_record_section_unmonitored(self, detector):
        detector.record_section("unmonitored_section", cpu_time=0.5)
        assert "unmonitored_section" not in detector.section_timings

    def test_record_section_disabled(self, default_config):
        default_config.enabled = False
        detector = StragglerDetector(config=default_config, rank=0, world_size=1)
        detector.record_section("forward_backward", cpu_time=0.5)
        assert len(detector.section_timings) == 0

    def test_record_section_keeps_recent_sample_window(self, default_config):
        default_config.sample_size = 3
        detector = StragglerDetector(config=default_config, rank=0, world_size=1)

        for step in range(5):
            detector.record_section("forward_backward", cpu_time=float(step), step=step)

        assert detector.section_timings["forward_backward"] == [
            (2, 2.0, None),
            (3, 3.0, None),
            (4, 4.0, None),
        ]

    def test_record_section_allows_unbounded_window_when_sample_size_zero(self, default_config):
        default_config.sample_size = 0
        detector = StragglerDetector(config=default_config, rank=0, world_size=1)

        for step in range(5):
            detector.record_section("forward_backward", cpu_time=float(step), step=step)

        assert len(detector.section_timings["forward_backward"]) == 5

    def test_should_profile(self, detector):
        for step in range(10):
            detector.current_step = step
            assert detector.should_profile() is False
        detector.current_step = 10
        assert detector.should_profile() is True
        detector.current_step = 15
        assert detector.should_profile() is False
        detector.current_step = 20
        assert detector.should_profile() is True

    def test_should_report(self, detector):
        detector.current_step = 0
        assert detector.should_report() is False
        detector.current_step = 100
        assert detector.should_report() is True

    def test_get_recent_section_time(self, detector):
        detector.record_section("forward_backward", cpu_time=0.1, step=1)
        detector.record_section("forward_backward", cpu_time=0.2, step=2)
        detector.record_section("forward_backward", cpu_time=0.3, step=3)
        assert detector.get_recent_section_time("forward_backward", num_samples=2) == pytest.approx(
            0.25, rel=1e-3
        )
        assert detector.get_recent_section_time("forward_backward", num_samples=1) == pytest.approx(
            0.3, rel=1e-3
        )

    def test_get_section_statistics(self, detector):
        detector.record_section("forward_backward", cpu_time=0.1)
        detector.record_section("forward_backward", cpu_time=0.2)
        detector.record_section("forward_backward", cpu_time=0.3)
        stats = detector.get_section_statistics()
        assert stats["forward_backward"]["count"] == 3
        assert stats["forward_backward"]["cpu_avg"] == pytest.approx(0.2, rel=1e-3)

    def test_reset(self, detector):
        detector.record_section("forward_backward", cpu_time=0.5)
        detector.increment_step()
        detector.reset()
        assert len(detector.section_timings) == 0
        assert detector.current_step == 0


class TestStragglerDetection:
    @pytest.fixture
    def detector(self):
        return StragglerDetector(
            config=StragglerConfig(
                enabled=True,
                straggler_threshold=1.5,
                monitor_sections=["forward_backward", "optimizer"],
            ),
            rank=0,
            world_size=8,
            node_name="test-node:gpu0",
        )

    def test_identify_stragglers_single_straggler(self, detector):
        section_times = {
            "forward_backward": {
                0: 0.100,
                1: 0.100,
                2: 0.200,
                3: 0.100,
            }
        }
        assert detector._identify_stragglers_from_times(section_times) == [2]

    def test_identify_stragglers_multiple_sections(self, detector):
        section_times = {
            "forward_backward": {0: 0.100, 1: 0.100, 2: 0.200, 3: 0.100},
            "optimizer": {0: 0.010, 1: 0.010, 2: 0.010, 3: 0.010},
        }
        assert detector._identify_stragglers_from_times(section_times) == [2]

    def test_identify_stragglers_empty_data(self, detector):
        assert detector._identify_stragglers_from_times({}) == []


class TestStragglerDetectorReportGeneration:
    @pytest.fixture
    def detector(self):
        return StragglerDetector(
            config=StragglerConfig(
                enabled=True,
                straggler_threshold=1.5,
                monitor_sections=["forward_backward", "optimizer"],
            ),
            rank=0,
            world_size=1,
            node_name="test-node:gpu0",
        )

    def test_generate_report_local(self, detector):
        for idx in range(5):
            detector.record_section("forward_backward", cpu_time=0.1 + idx * 0.01)
            detector.record_section("optimizer", cpu_time=0.01)
            detector.increment_step()

        report = detector.generate_report(step=10)
        assert report.step == 10
        assert report.node_names[0] == "test-node:gpu0"

    def test_save_report(self, detector):
        detector.record_section("forward_backward", cpu_time=0.1)
        report = detector.generate_report(step=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file_obj:
            temp_path = file_obj.name

        try:
            detector.save_report(report, temp_path)
            with open(temp_path, "r") as file_obj:
                data = json.load(file_obj)
            assert "step" in data
            assert "section_scores" in data
            assert "node_names" in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @patch("flagscale.runner.straggler.detector.dist")
    def test_gather_section_times_uses_configured_sample_size(self, mock_dist, detector):
        mock_dist.is_initialized.return_value = False
        detector.config.sample_size = 2
        for idx in range(4):
            detector.record_section("forward_backward", cpu_time=float(idx + 1), step=idx)

        result = detector._gather_section_times_across_ranks()

        assert result == {
            "forward_backward": {0: pytest.approx(3.5)},
        }


class TestStragglerDetectorWithMockedDistributed:
    @pytest.mark.skipif(detector_module.torch is None, reason="torch is not installed")
    @patch("flagscale.runner.straggler.detector.dist")
    @patch("flagscale.runner.straggler.detector.TORCH_DISTRIBUTED_AVAILABLE", True)
    def test_gather_section_times_across_ranks(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_backend.return_value = "gloo"
        detector = StragglerDetector(
            config=StragglerConfig(
                enabled=True, monitor_sections=["forward_backward", "optimizer"]
            ),
            rank=0,
            world_size=4,
            node_name="node0:gpu0",
        )
        detector.record_section("forward_backward", cpu_time=0.1)

        def mock_all_gather(gathered_list, local_tensor):
            for idx, value in enumerate([0.10, 0.11, 0.15, 0.12]):
                gathered_list[idx].fill_(value)

        mock_dist.all_gather.side_effect = mock_all_gather
        result = detector._gather_section_times_across_ranks()
        assert "forward_backward" in result
        assert len(result["forward_backward"]) == 4

    @pytest.mark.skipif(detector_module.torch is None, reason="torch is not installed")
    @patch("flagscale.runner.straggler.detector.dist")
    @patch("flagscale.runner.straggler.detector.TORCH_DISTRIBUTED_AVAILABLE", True)
    def test_gather_node_names_across_ranks(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        detector = StragglerDetector(
            config=StragglerConfig(enabled=True),
            rank=0,
            world_size=4,
            node_name="node0:gpu0",
        )

        def mock_all_gather_object(output_list, local_obj):
            for idx, name in enumerate(["node0:gpu0", "node0:gpu1", "node1:gpu0", "node1:gpu1"]):
                output_list[idx] = name

        mock_dist.all_gather_object.side_effect = mock_all_gather_object
        result = detector._gather_node_names_across_ranks()
        assert result[0] == "node0:gpu0"
        assert result[2] == "node1:gpu0"
