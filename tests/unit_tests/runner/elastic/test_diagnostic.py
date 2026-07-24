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
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from flagscale.runner.elastic.diagnostic import (
    _diagnostic_offsets,
    error_types,
    find_error_lines,
    format_line_range,
    generate_diagnostic_report,
)


class TestDiagnostic:
    """Test cases for diagnostic module"""

    def setup_method(self):
        _diagnostic_offsets.clear()

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Mock config object"""
        return OmegaConf.create({"train": {"system": {"logging": {"log_dir": str(tmp_path)}}}})

    @pytest.fixture
    def sample_log_content(self):
        """Sample log content with various errors"""
        return """
        [INFO] Starting training...
        [ERROR] CUDA out of memory
        Traceback (most recent call last):
          File "train.py", line 100, in <module>
            model.forward()
        torch.distributed.elastic.rendezvous.api.RendezvousConnectionError: Connection failed
        OutOfMemoryError: GPU memory exhausted
        [ERROR] Training failed
        """

    def test_error_types_dict_exists(self):
        """Test that error_types dictionary is properly defined"""
        assert isinstance(error_types, dict)
        assert len(error_types) > 0

        expected_keys = [
            "out of memory",
            "rendezvousconnectionerror",
            "traceback (most recent call last)",
            "cuda error",
            "hanging",
        ]
        for key in expected_keys:
            assert key in error_types

    def test_generate_diagnostic_report_empty_file(self, mock_config):
        """Test that report's format"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )
            assert (
                report == ""
                or "Diagnostic Report for localhost (node 0)" in report
                or "Log file is empty" in report
            )
        finally:
            os.unlink(temp_path)

    def test_generate_diagnostic_report_with_errors(self, mock_config, sample_log_content):
        """Test that diagnostic report is generated"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(sample_log_content)
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )

            assert "OutOfMemoryError" in report
            assert "RendezvousConnectionError" in report
            assert "CodeError" in report
        finally:
            os.unlink(temp_path)

    def test_generate_diagnostic_report_nonexistent_file(self, mock_config):
        """Test that log file is empty or does not exist"""
        report = generate_diagnostic_report(
            mock_config, "localhost", 0, "/nonexistent/file.log", return_content=True
        )

        assert "Log file is empty or does not exist" in report or "" in report

    def test_generate_diagnostic_report_no_errors(self, mock_config):
        """Test that no errors or unknown error"""
        content = """
        [INFO] Training started successfully
        [INFO] Epoch 1/100 completed
        [INFO] Training finished successfully
        """

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=True
            )

            assert "No errors or unknown error detected" in report or "" in report
        finally:
            os.unlink(temp_path)

    def test_generate_diagnostic_report_file_output(self, mock_config, sample_log_content):
        """Test that generate diagnostic report file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(sample_log_content)
            temp_path = f.name

        try:
            # Test file output mode
            result_path = generate_diagnostic_report(
                mock_config, "localhost", 0, temp_path, return_content=False
            )

            assert result_path is not None
            assert "diagnostic" in result_path or result_path.endswith(".txt")
        finally:
            os.unlink(temp_path)

    @patch("flagscale.runner.elastic.diagnostic.logger")
    def test_generate_diagnostic_report_read_error(self, mock_logger, mock_config):
        """Test diagnostic report generation with file read error"""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=10),
            patch("builtins.open", side_effect=PermissionError("Access denied")),
        ):
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, "/some/file.log", return_content=True
            )

            assert "Error analyzing log file" in report
            assert "Access denied" in report
            mock_logger.error.assert_called_once()

    def test_find_error_lines_and_format_line_range_helpers(self):
        lines = ["ok", "CUDA error happened", "still ok", "cuda ERROR again"]

        assert find_error_lines(lines, "cuda error") == [2, 4]
        assert find_error_lines(lines, "cuda error", start_line=2) == [4]
        assert format_line_range([]) == "unknown"
        assert format_line_range([3]) == "3"
        assert format_line_range([2, 5, 4]) == "2-5"

    def test_generate_diagnostic_report_writes_header_and_errors_incrementally(self, tmp_path):
        config = OmegaConf.create({"train": {"system": {"logging": {"log_dir": str(tmp_path)}}}})
        log_file = tmp_path / "host_0_localhost.output"
        log_file.write_text("start\nCUDA error happened\n", encoding="utf-8")

        diagnostic_path = generate_diagnostic_report(
            config, "localhost", 0, str(log_file), return_content=False
        )

        assert diagnostic_path.endswith("host_0_localhost_diagnostic.txt")
        diagnostic_content = open(diagnostic_path, encoding="utf-8").read()
        assert "Diagnostic Report for localhost (node 0)" in diagnostic_content
        assert "CUDAError" in diagnostic_content
        assert _diagnostic_offsets["localhost_0"] == 2

        assert (
            generate_diagnostic_report(config, "localhost", 0, str(log_file), return_content=True)
            == ""
        )

        log_file.write_text(
            "start\nCUDA error happened\nPermission denied while writing\n",
            encoding="utf-8",
        )
        incremental = generate_diagnostic_report(
            config, "localhost", 0, str(log_file), return_content=True
        )
        assert "PermissionError" in incremental
        assert "CUDAError" not in incremental

    def test_generate_diagnostic_report_handles_write_error(self, tmp_path):
        config = OmegaConf.create({"train": {"system": {"logging": {"log_dir": str(tmp_path)}}}})
        log_file = tmp_path / "host_0_localhost.output"
        log_file.write_text("fatal error\n", encoding="utf-8")
        read_handle = MagicMock()
        read_handle.__enter__.return_value.readlines.return_value = ["fatal error\n"]

        with patch("builtins.open", side_effect=[read_handle, PermissionError("readonly")]):
            report = generate_diagnostic_report(
                config, "localhost", 0, str(log_file), return_content=True
            )

        assert "Error analyzing log file" in report
        assert "readonly" in report
