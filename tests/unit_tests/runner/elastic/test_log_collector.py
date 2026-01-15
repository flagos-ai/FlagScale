import os
import tempfile
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from flagscale.runner.elastic.log_collector import _log_offsets, collect_logs


class TestLogCollector:
    """Test cases for log collector module"""

    @pytest.fixture
    def mock_config(self):
        """mock config"""
        return OmegaConf.create(
            {
                "train": {"system": {"logging": {"log_dir": "/tmp/test_logs"}}},
                "experiment": {"runner": {"no_shared_fs": False, "ssh_port": 22}},
            }
        )

    @pytest.fixture
    def mock_config_no_shared_fs(self):
        """mock config with no_shared_fs"""
        return OmegaConf.create(
            {
                "train": {"system": {"logging": {"log_dir": "/tmp/test_logs"}}},
                "experiment": {"runner": {"no_shared_fs": True, "ssh_port": 22}},
            }
        )

    def setup_method(self):
        """Reset log offsets before each test"""
        _log_offsets.clear()

    def test_collect_logs_localhost_initial(self, mock_config):
        """Test initial log collection from localhost"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Initial log content\nLine 2\nLine 3\n")
            src_log_path = f.name

        # Mock the source log file path
        expected_src = "/tmp/test_logs/host_0_localhost.output"

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=expected_src,
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs"),
            patch(
                "flagscale.runner.elastic.log_collector.run_local_command"
            ) as mock_run_local_command,
        ):
            result = collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

            # Should call run_local_command for localhost
            mock_run_local_command.assert_called()

            # Should return a destination file path
            assert result is not None
            assert result.endswith(".log")

        # Cleanup
        try:
            os.unlink(src_log_path)
        except:
            pass

    def test_collect_logs_localhost_incremental(self, mock_config):
        """Test incremental log collection"""
        # Set initial offset
        _log_offsets["localhost_0"] = 50

        expected_src = "/tmp/test_logs/host_0_localhost.output"

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=expected_src,
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=200),
            patch("os.makedirs"),
            patch(
                "flagscale.runner.elastic.log_collector.run_local_command"
            ) as mock_run_local_command,
        ):
            collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

            mock_run_local_command.assert_called()
            args, kwargs = mock_run_local_command.call_args
            command = args[0]
            assert "tail -c +51" in command  # offset + 1

            assert _log_offsets["localhost_0"] == 200

    def test_collect_logs_no_shared_fs(self, mock_config_no_shared_fs):
        """Test that collect logs with no_shared_fs"""
        expected_src = "/tmp/test_logs/host.output"

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=expected_src,
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs"),
            patch(
                "flagscale.runner.elastic.log_collector.run_local_command"
            ) as mock_run_local_command,
        ):
            collect_logs(mock_config_no_shared_fs, "localhost", 0, "/tmp/dest", dryrun=False)

            mock_run_local_command.assert_called()

    def test_collect_logs_file_not_found(self, mock_config):
        """Test that logs file is not found"""
        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
            patch("os.remove"),
            patch("flagscale.runner.elastic.log_collector.run_local_command"),
            patch("flagscale.runner.elastic.log_collector.logger") as mock_logger,
        ):
            result = collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

            assert result is None
            mock_logger.debug.assert_called()

    def test_collect_logs_empty_file(self, mock_config):
        """Test that logs file is empty"""
        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=0),
            patch("os.makedirs"),
            patch("os.remove"),
            patch("flagscale.runner.elastic.log_collector.run_local_command"),
            patch("flagscale.runner.elastic.log_collector.logger") as mock_logger,
        ):
            result = collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

            assert result is None
            mock_logger.debug.assert_called()

    def test_collect_logs_dryrun(self, mock_config):
        """Test that collect logs with dryrun mode"""
        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs"),
            patch("os.remove"),
            patch(
                "flagscale.runner.elastic.log_collector.run_local_command"
            ) as mock_run_local_command,
        ):
            collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=True)

            # Verify that run_local_command was called with dryrun=True
            mock_run_local_command.assert_called()
            call_args = mock_run_local_command.call_args
            # Check that dryrun=True was passed (as second positional argument)
            assert len(call_args[0]) >= 2, (
                "run_local_command should be called with command and dryrun arguments"
            )
            assert call_args[0][1] is True, "dryrun=True should be passed as second argument"

    def test_collect_logs_exception_handling(self, mock_config):
        """Test that collect logs with exception handling"""
        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs"),
            patch("os.remove"),
            patch(
                "flagscale.runner.elastic.log_collector.run_local_command",
                side_effect=Exception("Test error"),
            ),
            patch("flagscale.runner.elastic.log_collector.logger") as mock_logger,
        ):
            result = collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

            assert result is None
            mock_logger.error.assert_called()

    def test_log_offsets_management(self, mock_config):
        """Test that managing log's offsets"""
        assert "localhost_0" not in _log_offsets

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs"),
            patch("flagscale.runner.elastic.log_collector.run_local_command"),
        ):
            collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)
            assert _log_offsets["localhost_0"] == 100

            with patch("os.path.getsize", return_value=200):
                collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)
                assert _log_offsets["localhost_0"] == 200

    def test_destination_file_cleanup_on_failure(self, mock_config):
        """Test that failing to cleanup destination file"""
        dest_file = "/tmp/dest/host_0_localhost_temp_test.log"

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.path.exists", return_value=False),
            patch("os.makedirs"),
            patch("os.path.exists") as mock_exists,
            patch("os.remove"),
        ):
            # Mock that dest file exists
            mock_exists.side_effect = lambda path: path == dest_file

            result = collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

            assert result is None
