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

"""Unit tests for straggler health checks."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.straggler.healthcheck import (
    ElasticTrainingHealthChecker,
    NetworkHealthChecker,
)


class TestNetworkHealthChecker:
    @pytest.fixture
    def checker(self):
        return NetworkHealthChecker(rank=0, world_size=4)

    @patch("socket.socket")
    def test_check_node_connectivity_success(self, mock_socket_class, checker):
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket
        results = checker.check_node_connectivity(["192.168.1.1", "192.168.1.2"])
        assert results["192.168.1.1"] is True
        assert results["192.168.1.2"] is True

    @patch("subprocess.run")
    def test_measure_latency_failure(self, mock_run, checker):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        results = checker.measure_latency(["192.168.1.1"])
        assert results["192.168.1.1"] == float("inf")

    @patch("subprocess.run")
    def test_measure_latency_parse_failure_stays_infinite(self, mock_run, checker):
        mock_run.return_value = MagicMock(returncode=0, stdout="ping succeeded without avg line")
        results = checker.measure_latency(["192.168.1.1"])
        assert results["192.168.1.1"] == float("inf")

    def test_identify_unhealthy_nodes(self, checker):
        unhealthy = checker.identify_unhealthy_nodes(
            {
                "192.168.1.1": {"connectivity": True, "latency_ms": 50.0, "bandwidth_mbps": 100.0},
                "192.168.1.2": {"connectivity": False, "latency_ms": 0.0, "bandwidth_mbps": 0.0},
                "192.168.1.3": {"connectivity": True, "latency_ms": 200.0, "bandwidth_mbps": 50.0},
            },
            max_latency_ms=100.0,
            min_bandwidth_mbps=10.0,
        )
        assert unhealthy == ["192.168.1.2", "192.168.1.3"]

    def test_get_network_summary(self, checker):
        with patch.object(checker, "comprehensive_health_check") as mock_check:
            mock_check.return_value = {
                "192.168.1.1": {
                    "connectivity": True,
                    "latency_ms": 10.0,
                    "bandwidth_mbps": 100.0,
                    "healthy": True,
                },
                "192.168.1.2": {
                    "connectivity": True,
                    "latency_ms": 15.0,
                    "bandwidth_mbps": 90.0,
                    "healthy": True,
                },
            }
            summary = checker.get_network_summary(["192.168.1.1", "192.168.1.2"])
            assert summary["total_nodes"] == 2
            assert summary["healthy_nodes"] == 2
            assert summary["network_healthy"] is True

    def test_save_health_report(self, checker):
        health_results = {
            "192.168.1.1": {
                "connectivity": True,
                "latency_ms": 10.0,
                "bandwidth_mbps": 100.0,
                "healthy": True,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file_obj:
            temp_path = file_obj.name

        try:
            checker.save_health_report(health_results, temp_path)
            with open(temp_path, "r") as file_obj:
                content = file_obj.read()
            assert "192.168.1.1" in content
            assert "Network Health Check Report" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestElasticTrainingHealthChecker:
    def test_detect_unstable_nodes(self):
        checker = ElasticTrainingHealthChecker(rank=0, world_size=4)
        unstable = checker.detect_unstable_nodes(
            [
                {
                    "check_id": 0,
                    "timestamp": 1000.0,
                    "health_results": {
                        "192.168.1.1": {"connectivity": True},
                        "192.168.1.2": {"connectivity": False},
                    },
                },
                {
                    "check_id": 1,
                    "timestamp": 1030.0,
                    "health_results": {
                        "192.168.1.1": {"connectivity": True},
                        "192.168.1.2": {"connectivity": False},
                    },
                },
            ],
            instability_threshold=0.5,
        )
        assert unstable == ["192.168.1.2"]
