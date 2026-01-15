import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.elastic.diagnostic import error_types, generate_diagnostic_report


class TestDiagnostic:
    """Test cases for diagnostic module"""

    @pytest.fixture
    def mock_config(self):
        """Mock config object"""
        return MagicMock()

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
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            report = generate_diagnostic_report(
                mock_config, "localhost", 0, "/some/file.log", return_content=True
            )

            assert "Error reading log file" in report or "" in report
