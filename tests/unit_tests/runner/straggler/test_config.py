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

"""Unit tests for straggler configuration."""

from flagscale.runner.straggler.config import StragglerConfig


class TestStragglerConfig:
    def test_default_values(self):
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
        config = StragglerConfig()
        assert isinstance(config.monitor_sections, list)
        assert "forward_backward" in config.monitor_sections
        assert "optimizer" in config.monitor_sections

    def test_monitor_sections_custom(self):
        custom_sections = ["custom_section1", "custom_section2"]
        config = StragglerConfig(monitor_sections=custom_sections)
        assert config.monitor_sections == custom_sections
