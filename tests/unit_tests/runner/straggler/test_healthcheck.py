"""
Unit tests for straggler healthcheck module.
"""

import os
import tempfile

from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.straggler.healthcheck import (
    ElasticTrainingHealthChecker,
    NetworkHealthChecker,
)


class TestNetworkHealthChecker:
    """Test cases for NetworkHealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a NetworkHealthChecker for testing."""
        return NetworkHealthChecker(rank=0, world_size=4)

    def test_init(self):
        """Test NetworkHealthChecker initialization."""
        checker = NetworkHealthChecker(rank=2, world_size=8)

        assert checker.rank == 2
        assert checker.world_size == 8
        assert checker.node_health == {}
        assert checker.latency_matrix == {}

    @patch('socket.socket')
    def test_check_node_connectivity_success(self, mock_socket_class, checker):
        """Test successful node connectivity check."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0  # Success
        mock_socket_class.return_value = mock_socket

        results = checker.check_node_connectivity(["192.168.1.1", "192.168.1.2"])

        assert results["192.168.1.1"] is True
        assert results["192.168.1.2"] is True

    @patch('socket.socket')
    def test_check_node_connectivity_failure(self, mock_socket_class, checker):
        """Test failed node connectivity check."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 1  # Failure
        mock_socket_class.return_value = mock_socket

        results = checker.check_node_connectivity(["192.168.1.1"])

        assert results["192.168.1.1"] is False

    @patch('socket.socket')
    def test_check_node_connectivity_exception(self, mock_socket_class, checker):
        """Test node connectivity check with exception."""
        mock_socket = MagicMock()
        mock_socket.connect_ex.side_effect = Exception("Connection error")
        mock_socket_class.return_value = mock_socket

        results = checker.check_node_connectivity(["192.168.1.1"])

        assert results["192.168.1.1"] is False

    @patch('subprocess.run')
    def test_measure_latency_success(self, mock_run, checker):
        """Test successful latency measurement."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="rtt min/avg/max/mdev = 0.5/1.0/1.5/0.2 ms"
        )

        results = checker.measure_latency(["192.168.1.1"])

        assert "192.168.1.1" in results
        # Should have parsed the latency value
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_measure_latency_failure(self, mock_run, checker):
        """Test failed latency measurement."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        results = checker.measure_latency(["192.168.1.1"])

        assert results["192.168.1.1"] == float('inf')

    @patch('subprocess.run')
    def test_measure_latency_exception(self, mock_run, checker):
        """Test latency measurement with exception."""
        mock_run.side_effect = Exception("Ping failed")

        results = checker.measure_latency(["192.168.1.1"])

        assert results["192.168.1.1"] == float('inf')

    def test_identify_unhealthy_nodes(self, checker):
        """Test identifying unhealthy nodes from health results."""
        health_results = {
            "192.168.1.1": {"connectivity": True, "latency_ms": 50.0, "bandwidth_mbps": 100.0},
            "192.168.1.2": {
                "connectivity": False,  # Unhealthy - not connected
                "latency_ms": 0.0,
                "bandwidth_mbps": 0.0,
            },
            "192.168.1.3": {
                "connectivity": True,
                "latency_ms": 200.0,  # Unhealthy - high latency
                "bandwidth_mbps": 50.0,
            },
            "192.168.1.4": {
                "connectivity": True,
                "latency_ms": 30.0,
                "bandwidth_mbps": 5.0,  # Unhealthy - low bandwidth
            },
        }

        unhealthy = checker.identify_unhealthy_nodes(
            health_results, max_latency_ms=100.0, min_bandwidth_mbps=10.0
        )

        assert "192.168.1.2" in unhealthy  # Not connected
        assert "192.168.1.3" in unhealthy  # High latency
        assert "192.168.1.4" in unhealthy  # Low bandwidth
        assert "192.168.1.1" not in unhealthy  # Healthy

    def test_identify_unhealthy_nodes_all_healthy(self, checker):
        """Test when all nodes are healthy."""
        health_results = {
            "192.168.1.1": {"connectivity": True, "latency_ms": 10.0, "bandwidth_mbps": 100.0},
            "192.168.1.2": {"connectivity": True, "latency_ms": 20.0, "bandwidth_mbps": 80.0},
        }

        unhealthy = checker.identify_unhealthy_nodes(health_results)

        assert len(unhealthy) == 0

    def test_get_network_summary(self, checker):
        """Test network summary generation with mocked health check."""
        # Mock comprehensive_health_check
        with patch.object(checker, 'comprehensive_health_check') as mock_check:
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
            assert summary["health_percentage"] == 100.0
            assert summary["network_healthy"] is True

    def test_save_health_report(self, checker):
        """Test saving health report to file."""
        health_results = {
            "192.168.1.1": {
                "connectivity": True,
                "latency_ms": 10.0,
                "bandwidth_mbps": 100.0,
                "healthy": True,
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name

        try:
            checker.save_health_report(health_results, temp_path)

            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                content = f.read()

            assert "192.168.1.1" in content
            assert "Network Health Check Report" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_health_report_error_handling(self, checker):
        """Test save health report handles write errors."""
        health_results = {"192.168.1.1": {"connectivity": True}}

        # Should not raise, just print warning
        checker.save_health_report(health_results, "/nonexistent/path/report.txt")


class TestElasticTrainingHealthChecker:
    """Test cases for ElasticTrainingHealthChecker class."""

    @pytest.fixture
    def elastic_checker(self):
        """Create an ElasticTrainingHealthChecker for testing."""
        return ElasticTrainingHealthChecker(rank=0, world_size=4)

    def test_init(self):
        """Test ElasticTrainingHealthChecker initialization."""
        checker = ElasticTrainingHealthChecker(rank=1, world_size=8)

        assert checker.rank == 1
        assert checker.world_size == 8
        assert checker.health_history == []

    def test_inherits_from_network_checker(self, elastic_checker):
        """Test that ElasticTrainingHealthChecker inherits from NetworkHealthChecker."""
        assert isinstance(elastic_checker, NetworkHealthChecker)

    def test_detect_unstable_nodes(self, elastic_checker):
        """Test detecting unstable nodes from health history."""
        health_history = [
            {
                "check_id": 0,
                "timestamp": 1000.0,
                "health_results": {
                    "192.168.1.1": {"connectivity": True},
                    "192.168.1.2": {"connectivity": False},  # Failed
                },
            },
            {
                "check_id": 1,
                "timestamp": 1030.0,
                "health_results": {
                    "192.168.1.1": {"connectivity": True},
                    "192.168.1.2": {"connectivity": False},  # Failed again
                },
            },
            {
                "check_id": 2,
                "timestamp": 1060.0,
                "health_results": {
                    "192.168.1.1": {"connectivity": True},
                    "192.168.1.2": {"connectivity": True},  # Recovered
                },
            },
        ]

        # 192.168.1.2 failed 2 out of 3 times = 66.7% failure rate
        unstable = elastic_checker.detect_unstable_nodes(
            health_history, instability_threshold=0.5  # 50% threshold
        )

        assert "192.168.1.2" in unstable
        assert "192.168.1.1" not in unstable

    def test_detect_unstable_nodes_all_stable(self, elastic_checker):
        """Test when all nodes are stable."""
        health_history = [
            {
                "check_id": i,
                "timestamp": 1000.0 + i * 30,
                "health_results": {
                    "192.168.1.1": {"connectivity": True},
                    "192.168.1.2": {"connectivity": True},
                },
            }
            for i in range(5)
        ]

        unstable = elastic_checker.detect_unstable_nodes(health_history)

        assert len(unstable) == 0

    def test_detect_unstable_nodes_empty_history(self, elastic_checker):
        """Test with empty health history."""
        unstable = elastic_checker.detect_unstable_nodes([])

        assert len(unstable) == 0

    def test_health_history_accumulation(self, elastic_checker):
        """Test that health history is accumulated."""
        assert len(elastic_checker.health_history) == 0

        # Mock comprehensive_health_check
        with patch.object(elastic_checker, 'comprehensive_health_check') as mock_check:
            mock_check.return_value = {"192.168.1.1": {"connectivity": True}}

            # This would normally take time due to sleep, so we mock the timing
            with patch('time.sleep'):
                elastic_checker.monitor_elastic_health(
                    ["192.168.1.1"], check_interval=0.1, num_checks=3
                )

        assert len(elastic_checker.health_history) == 3


class TestHealthCheckerEdgeCases:
    """Test edge cases for health checker classes."""

    def test_empty_node_list(self):
        """Test with empty node list."""
        checker = NetworkHealthChecker()

        results = checker.check_node_connectivity([])
        assert results == {}

        latencies = checker.measure_latency([])
        assert latencies == {}

    def test_single_node(self):
        """Test with single node."""
        checker = NetworkHealthChecker(rank=0, world_size=1)

        with patch('socket.socket') as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket

            results = checker.check_node_connectivity(["localhost"])
            assert len(results) == 1

    def test_check_bandwidth_exception(self):
        """Test bandwidth check handles exceptions."""
        checker = NetworkHealthChecker()

        with patch('socket.socket') as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect.side_effect = Exception("Connection failed")
            mock_socket_class.return_value = mock_socket

            bandwidths = checker.check_bandwidth(["192.168.1.1"])
            assert bandwidths["192.168.1.1"] == 0.0

    def test_network_summary_empty_nodes(self):
        """Test network summary with no reachable nodes."""
        checker = NetworkHealthChecker()

        with patch.object(checker, 'comprehensive_health_check') as mock_check:
            mock_check.return_value = {
                "192.168.1.1": {
                    "connectivity": False,
                    "latency_ms": float('inf'),
                    "bandwidth_mbps": 0.0,
                    "healthy": False,
                }
            }

            summary = checker.get_network_summary(["192.168.1.1"])

            assert summary["healthy_nodes"] == 0
            assert summary["network_healthy"] is False
