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

"""Unit tests for straggler reports."""

import json

import pytest

from flagscale.runner.straggler.report import StragglerReport


class TestStragglerReport:
    @pytest.fixture
    def sample_section_scores(self):
        return {
            "forward_backward": {0: 0.100, 1: 0.105, 2: 0.098, 3: 0.102},
            "optimizer": {0: 0.010, 1: 0.011, 2: 0.009, 3: 0.010},
        }

    @pytest.fixture
    def sample_gpu_scores(self):
        return {0: 10.0, 1: 9.5, 2: 10.2, 3: 9.8}

    @pytest.fixture
    def sample_node_names(self):
        return {
            0: "node0:gpu0",
            1: "node0:gpu1",
            2: "node1:gpu0",
            3: "node1:gpu1",
        }

    def test_to_dict_json_serializable(
        self, sample_section_scores, sample_gpu_scores, sample_node_names
    ):
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[2, 3],
            node_names=sample_node_names,
        )
        parsed = json.loads(json.dumps(report.to_dict()))
        assert parsed["step"] == 100

    def test_to_text_no_stragglers(
        self, sample_section_scores, sample_gpu_scores, sample_node_names
    ):
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
        report = StragglerReport(
            step=100,
            section_scores=sample_section_scores,
            gpu_scores=sample_gpu_scores,
            straggler_ranks=[2, 3],
            node_names=sample_node_names,
        )
        text = report.to_text()
        assert "Detected stragglers" in text
        assert "node1:gpu0" in text

    def test_slowdown_calculation(self):
        report = StragglerReport(
            step=100,
            section_scores={"forward_backward": {0: 0.100, 1: 0.200}},
        )
        assert "Slowdown" in report.to_text()

    def test_identify_stragglers_treats_section_scores_as_timings(self):
        report = StragglerReport(
            step=100,
            section_scores={"forward_backward": {0: 0.100, 1: 0.210, 2: 0.110}},
        )

        assert report.identify_stragglers(threshold=2.0) == [1]

    def test_identify_gpu_stragglers_treats_higher_scores_as_faster(self):
        report = StragglerReport(
            step=100,
            gpu_scores={0: 10.0, 1: 4.0, 2: 9.0},
        )

        assert report.identify_gpu_stragglers(threshold=2.0) == [1]

    def test_timestamp_is_set(self):
        report = StragglerReport(step=100)
        report.timestamp = 1234567890.0
        assert report.to_dict()["timestamp"] == 1234567890.0
