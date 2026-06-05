"""Tests for config error wrapping in flagscale/run.py and flagscale/cli.py.

Verifies that OmegaConf errors are caught and re-raised with detailed context
including the config content and actionable hints.
"""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import (
    ConfigAttributeError,
    InterpolationResolutionError,
    MissingMandatoryValue,
)

# ---------------------------------------------------------------------------
# Tests for _main() error wrapping in run.py
# ---------------------------------------------------------------------------


class TestMainErrorWrapping:
    """Test that OmegaConf errors in _main() propagate correctly,
    and validate_task rejects bad inputs."""

    def test_struct_config_missing_experiment_raises(self):
        """_main() on a struct config missing 'experiment' raises ConfigAttributeError."""
        from flagscale.run import _main

        config = OmegaConf.create({"action": "run"})
        OmegaConf.set_struct(config, True)

        with pytest.raises(ConfigAttributeError, match="experiment"):
            _main(config)

    def test_valid_task_type_and_action(self):
        """Valid task_type and action should pass validate_task without error."""
        from flagscale.run import validate_task

        for task in ("train", "serve", "compress", "rl", "inference"):
            validate_task(task, "run")

    def test_invalid_task_type(self):
        """Invalid task_type should raise ValueError."""
        from flagscale.run import validate_task

        with pytest.raises(ValueError, match="Invalid task_type"):
            validate_task("unknown_task", "run")

    def test_invalid_action(self):
        """Invalid action for a valid task_type should raise ValueError."""
        from flagscale.run import validate_task

        with pytest.raises(ValueError, match="not allowed"):
            validate_task("train", "invalid_action")

    def test_missing_mandatory_value(self):
        """Accessing a ??? value raises MissingMandatoryValue."""
        config = OmegaConf.create({"key": "???"})

        with pytest.raises(MissingMandatoryValue):
            _ = config.key

    def test_bad_interpolation(self):
        """A broken ${...} reference raises InterpolationResolutionError."""
        config = OmegaConf.create({"key": "${nonexistent}"})

        with pytest.raises(InterpolationResolutionError):
            _ = config.key


# ---------------------------------------------------------------------------
# Tests for _handle_config_error() in cli.py
# ---------------------------------------------------------------------------


class TestCliErrorHandling:
    """Test that CLI error handler formats errors with context."""

    def test_handle_config_error_exits_with_context(self, capsys):
        """_handle_config_error should print config file, error, hint and exit."""
        from click.exceptions import Exit

        from flagscale.cli import _handle_config_error

        with pytest.raises(Exit) as exc_info:
            _handle_config_error(
                ValueError("test error"),
                "/path/to/config",
                "train",
            )
        assert exc_info.value.exit_code == 1

        captured = capsys.readouterr()
        assert "Config error in: /path/to/config/train.yaml" in captured.err
        assert "test error" in captured.err
        assert "Hint:" in captured.err

    def test_run_task_catches_exception(self):
        """run_task should catch exceptions from run_main and format them."""
        from click.exceptions import Exit

        from flagscale.cli import run_task

        with patch("flagscale.run.main", side_effect=RuntimeError("bad config")):
            with pytest.raises(Exit) as exc_info:
                run_task("/nonexistent/path", "bad_config", "run")
            assert exc_info.value.exit_code == 1
