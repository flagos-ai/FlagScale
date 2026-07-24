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

import os
import tempfile
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from flagscale.runner.elastic.log_collector import (
    _log_offsets,
    collect_logs,
    find_actual_log_file,
    get_file_size,
)


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

    def test_get_file_size_local_and_remote(self):
        with (
            patch("os.path.exists", side_effect=[True, False]),
            patch("os.path.getsize", return_value=123),
            patch(
                "flagscale.runner.elastic.log_collector.get_remote_file_size",
                return_value=456,
            ) as remote_size,
        ):
            assert get_file_size("localhost", "/tmp/local.log") == 123
            assert get_file_size("localhost", "/tmp/missing.log") == -1
            assert get_file_size("worker0", "/tmp/remote.log", port=2222) == 456

        remote_size.assert_called_once_with("worker0", "/tmp/remote.log", 2222)

    def test_find_actual_log_file_exact_match_glob_and_fallback(self, tmp_path):
        exact = tmp_path / "host_0_worker0.output"
        exact.write_text("log")
        assert find_actual_log_file(str(tmp_path), 0, "worker0") == str(exact)

        exact.unlink()
        discovered = tmp_path / "host_0_10.0.0.1.output"
        discovered.write_text("log")
        assert find_actual_log_file(str(tmp_path), 0, "worker0") == str(discovered)

        discovered.unlink()
        assert find_actual_log_file(str(tmp_path), 0, "worker0") == str(
            tmp_path / "host_0_worker0.output"
        )
        assert find_actual_log_file(str(tmp_path), 0, "worker0", no_shared_fs=True) == str(
            tmp_path / "host.output"
        )

    def test_collect_logs_remote_uses_ssh_tail_command_and_updates_offset(self, mock_config):
        src_log = "/tmp/test_logs/host_0_worker0.output"
        dest_log = "/tmp/dest/host_0_worker0_current.log"

        def fake_exists(path):
            return path in {src_log, dest_log}

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=src_log,
            ),
            patch("os.path.exists", side_effect=fake_exists),
            patch("os.path.getsize", return_value=10),
            patch("os.makedirs"),
            patch(
                "flagscale.runner.elastic.log_collector.get_remote_file_size",
                return_value=100,
            ),
            patch("flagscale.runner.elastic.log_collector.run_local_command") as run_local,
        ):
            result = collect_logs(mock_config, "worker0", 0, "/tmp/dest", dryrun=False)

        assert result == dest_log
        command = run_local.call_args.args[0]
        assert "ssh -p 22 worker0" in command
        assert "tail -c +1" in command
        assert src_log in command
        assert _log_offsets["worker0_0"] == 100

    def test_collect_logs_removes_empty_destination_after_command_success(self, mock_config):
        src_log = "/tmp/test_logs/host_0_localhost.output"
        dest_log = "/tmp/dest/host_0_localhost_current.log"

        def fake_exists(path):
            return path in {src_log, dest_log}

        def fake_size(path):
            return 100 if path == src_log else 0

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=src_log,
            ),
            patch("os.path.exists", side_effect=fake_exists),
            patch("os.path.getsize", side_effect=fake_size),
            patch("os.makedirs"),
            patch("flagscale.runner.elastic.log_collector.run_local_command"),
            patch("os.remove") as remove,
        ):
            result = collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

        assert result is None
        remove.assert_called_once_with(dest_log)

    def test_collect_logs_makedirs_permission_error(self, mock_config):
        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value="/tmp/test_logs/host_0_localhost.output",
            ),
            patch("os.makedirs", side_effect=PermissionError("cannot create destination")),
            pytest.raises(PermissionError),
        ):
            collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False)

    def test_collect_logs_remote_source_has_no_content_removes_destination(self, mock_config):
        src_log = "/tmp/test_logs/host_0_worker0.output"
        dest_log = "/tmp/dest/host_0_worker0_current.log"

        def fake_exists(path):
            return path in {src_log, dest_log}

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=src_log,
            ),
            patch("os.path.exists", side_effect=fake_exists),
            patch("os.path.getsize", return_value=1),
            patch("os.makedirs"),
            patch(
                "flagscale.runner.elastic.log_collector.get_remote_file_size",
                return_value=-1,
            ),
            patch("flagscale.runner.elastic.log_collector.run_local_command"),
            patch("os.remove") as remove,
        ):
            result = collect_logs(mock_config, "worker0", 0, "/tmp/dest", dryrun=False)

        assert result is None
        remove.assert_called_once_with(dest_log)

    def test_collect_logs_no_new_bytes_returns_none(self, mock_config):
        src_log = "/tmp/test_logs/host_0_localhost.output"
        _log_offsets["localhost_0"] = 100

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=src_log,
            ),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs"),
            patch("flagscale.runner.elastic.log_collector.run_local_command"),
        ):
            assert collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False) is None
            assert _log_offsets["localhost_0"] == 100

    def test_collect_logs_exception_removes_existing_destination(self, mock_config):
        src_log = "/tmp/test_logs/host_0_localhost.output"
        dest_log = "/tmp/dest/host_0_localhost_current.log"

        def fake_exists(path):
            return path in {src_log, dest_log}

        with (
            patch(
                "flagscale.runner.elastic.log_collector.find_actual_log_file",
                return_value=src_log,
            ),
            patch("os.path.exists", side_effect=fake_exists),
            patch("os.makedirs"),
            patch(
                "flagscale.runner.elastic.log_collector.run_local_command",
                side_effect=RuntimeError("tail failed"),
            ),
            patch("os.remove") as remove,
        ):
            assert collect_logs(mock_config, "localhost", 0, "/tmp/dest", dryrun=False) is None

        remove.assert_called_once_with(dest_log)
