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

"""
Test override of get_device in plugin_flagscale.

Run from Megatron-LM-FL root with FlagScale in PYTHONPATH:
    PYTHONPATH=/share/project/lixianduo/override/FlagScale/flagscale/train/megatron:$PYTHONPATH \
        python -m pytest megatron/plugin_flagscale/test_override.py -v

Or standalone:
    python megatron/plugin_flagscale/test_override.py
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure both Megatron-LM-FL and FlagScale's megatron paths are importable
MEGATRON_LM_FL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if MEGATRON_LM_FL_ROOT not in sys.path:
    sys.path.insert(0, MEGATRON_LM_FL_ROOT)

from megatron.plugin.decorators import (
    register,
    register_override_method,
    overridable,
    get_override_method,
    _lazy_registry,
    _plugin_registry,
    _plugin_impl_cache,
    _original_impl_cache,
)


def _clear_registry():
    """Helper to reset all override state between tests."""
    _plugin_registry.clear()
    _plugin_impl_cache.clear()
    _original_impl_cache.clear()
    _lazy_registry.clear()


class TestGetDeviceOverride(unittest.TestCase):
    """Test that get_device can be overridden via the plugin mechanism."""

    def setUp(self):
        _clear_registry()
        _plugin_impl_cache.clear()
        _original_impl_cache.clear()

    def tearDown(self):
        _clear_registry()
        _plugin_impl_cache.clear()
        _original_impl_cache.clear()

    def test_no_override_uses_original(self):
        """Without override registered, original get_device is called."""
        mock_platform = MagicMock()
        mock_platform.device_name.return_value = "cuda"

        @overridable
        def get_device(local_rank=None):
            import torch

            # Simulate original logic
            if local_rank is None:
                return torch.device(mock_platform.device_name())
            else:
                return torch.device(f"{mock_platform.device_name()}:{local_rank}")

        with patch("torch.distributed.get_backend", return_value="nccl"):
            import torch

            device = get_device()
            self.assertEqual(device, torch.device("cuda"))

            device = get_device(local_rank=2)
            self.assertEqual(device, torch.device("cuda:2"))

    def test_override_replaces_get_device(self):
        """After registering override, the plugin implementation is used."""
        mock_platform = MagicMock()
        mock_platform.device_name.return_value = "cuda"

        @overridable
        def get_device(local_rank=None):
            import torch

            # Original: hardcoded cuda
            if local_rank is None:
                return torch.device("cuda")
            else:
                return torch.device(f"cuda:{local_rank}")

        # Simulate the plugin override (uses cur_platform)
        def get_device_override(local_rank=None):
            # Return string representation to avoid torch.device validation
            # for non-standard device names; tests the override dispatch mechanism
            if local_rank is None:
                return mock_platform.device_name()
            else:
                return f"{mock_platform.device_name()}:{local_rank}"

        # The method_key for module-level function is "module_basename.func_name"
        # Since get_device is defined in this test, module is __main__ or test file
        module_parts = get_device.__module__.split(".")
        module_name = module_parts[-1]
        register_override_method(f"{module_name}.get_device", get_device_override)

        # Now test with different platform device names
        mock_platform.device_name.return_value = "musa"
        device = get_device()
        self.assertEqual(device, "musa")

        device = get_device(local_rank=3)
        self.assertEqual(device, "musa:3")

        # Also works for cuda
        mock_platform.device_name.return_value = "cuda"
        device = get_device()
        self.assertEqual(device, "cuda")

    def test_override_via_lazy_register(self):
        """Test override via register() with target/impl string paths."""
        import types
        import torch

        # Create a mock platform
        mock_platform = MagicMock()
        mock_platform.device_name.return_value = "npu"

        # Create the plugin module in sys.modules
        plugin_mod = types.ModuleType("megatron.plugin_flagscale._test_get_device")

        def get_device_plugin(local_rank=None):
            # Return string to avoid torch.device validation for non-standard names
            if local_rank is None:
                return mock_platform.device_name()
            else:
                return f"{mock_platform.device_name()}:{local_rank}"

        plugin_mod.get_device = get_device_plugin
        sys.modules["megatron.plugin_flagscale._test_get_device"] = plugin_mod

        try:
            # Register lazily
            register(
                target="megatron.core.training.dist_signal_handler.get_device",
                impl="megatron.plugin_flagscale._test_get_device.get_device",
            )

            # The method_key should be "dist_signal_handler.get_device"
            key = "dist_signal_handler.get_device"
            self.assertIn(key, _lazy_registry)

            # Resolve and verify
            result = get_override_method(key)
            self.assertIsNotNone(result)

            device = result()
            self.assertEqual(device, "npu")

            device = result(local_rank=1)
            self.assertEqual(device, "npu:1")
        finally:
            del sys.modules["megatron.plugin_flagscale._test_get_device"]

    def test_gloo_backend_returns_cpu(self):
        """Override should still return cpu device for gloo backend."""
        import torch

        def get_device_plugin(local_rank=None):
            backend = torch.distributed.get_backend()
            if backend == "nccl":
                return torch.device("cuda")
            elif backend == "gloo":
                return torch.device("cpu")
            else:
                raise RuntimeError

        @overridable
        def get_device(local_rank=None):
            return torch.device("cuda")  # dummy original

        module_parts = get_device.__module__.split(".")
        module_name = module_parts[-1]
        register_override_method(f"{module_name}.get_device", get_device_plugin)

        with patch("torch.distributed.get_backend", return_value="gloo"):
            device = get_device()
            self.assertEqual(device, torch.device("cpu"))

    def test_unsupported_backend_raises(self):
        """Override should raise RuntimeError for unsupported backends."""
        import torch

        def get_device_plugin(local_rank=None):
            backend = torch.distributed.get_backend()
            if backend == "nccl":
                return torch.device("cuda")
            elif backend == "gloo":
                return torch.device("cpu")
            else:
                raise RuntimeError(f"Unsupported backend: {backend}")

        @overridable
        def get_device(local_rank=None):
            return torch.device("cuda")

        module_parts = get_device.__module__.split(".")
        module_name = module_parts[-1]
        register_override_method(f"{module_name}.get_device", get_device_plugin)

        with patch("torch.distributed.get_backend", return_value="mpi"):
            with self.assertRaises(RuntimeError):
                get_device()


if __name__ == "__main__":
    unittest.main(verbosity=2)
