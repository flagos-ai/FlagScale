"""
Unit tests for straggler detection configuration module.
"""

import pytest

from flagscale.runner.straggler.config import StragglerConfig


class TestStragglerConfig:
    """Test cases for StragglerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = StragglerConfig()

        assert config.enabled is True
        assert config.profiling_interval == 10
        assert config.report_interval_steps == 100
        assert config.straggler_threshold == 1.5
        assert config.warmup_steps == 10
        assert config.sample_size == 100
        assert config.gather_on_rank0 is True
        assert config.enable_gpu_profile is True

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = StragglerConfig(
            enabled=False,
            profiling_interval=5,
            report_interval_steps=50,
            straggler_threshold=2.0,
            warmup_steps=20,
            sample_size=5,
            gather_on_rank0=False,
            enable_gpu_profile=False,
        )

        assert config.enabled is False
        assert config.profiling_interval == 5
        assert config.report_interval_steps == 50
        assert config.straggler_threshold == 2.0
        assert config.warmup_steps == 20
        assert config.sample_size == 5
        assert config.gather_on_rank0 is False
        assert config.enable_gpu_profile is False

    def test_monitor_sections_default(self):
        """Test that monitor_sections has default values."""
        config = StragglerConfig()

        assert isinstance(config.monitor_sections, list)
        assert len(config.monitor_sections) > 0
        # Check some expected sections
        assert "forward_backward" in config.monitor_sections
        assert "optimizer" in config.monitor_sections

    def test_monitor_sections_custom(self):
        """Test that monitor_sections can be customized."""
        custom_sections = ["custom_section1", "custom_section2"]
        config = StragglerConfig(monitor_sections=custom_sections)

        assert config.monitor_sections == custom_sections

    def test_straggler_threshold_positive(self):
        """Test straggler threshold with positive values."""
        config = StragglerConfig(straggler_threshold=1.0)
        assert config.straggler_threshold == 1.0

        config = StragglerConfig(straggler_threshold=3.0)
        assert config.straggler_threshold == 3.0

    def test_config_immutability(self):
        """Test that config values can be modified after creation."""
        config = StragglerConfig()

        # Modify values
        config.enabled = False
        config.straggler_threshold = 2.5

        assert config.enabled is False
        assert config.straggler_threshold == 2.5
