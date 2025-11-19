import os
import sys

from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.estimator.meta_base import MetaModule
from flagscale.runner.estimator.meta_registry import (
    ModelStatsRegistry,
    get_registry,
    register_model,
)


class TestModelStatsRegistry:
    """Test suite for ModelStatsRegistry class."""

    def test_init(self):
        """Test initialization."""
        # With model_id
        registry = ModelStatsRegistry("test_model")
        assert registry.model_id == "test_model"
        assert registry.total_flops == 0
        assert registry.total_params == 0
        assert registry.total_acts == 0
        assert registry.module_ids == {}
        assert registry.flops_counter == 0

        # Without model_id (should use default)
        registry = ModelStatsRegistry()
        assert registry.model_id == "default"

    def test_add_flops(self):
        """Test adding FLOPs."""
        registry = ModelStatsRegistry()

        # Add with path
        registry.add_flops(100, "op1")
        assert registry.total_flops == 100
        assert len(registry.flops_logs) == 1
        assert registry.flops_logs[0][0] == 100
        assert registry.flops_logs[0][1] == "op1"
        assert registry.flops_logs[0][2] == 0  # level derived from path
        assert "op1" in registry.flops_by_module
        assert registry.flops_by_module["op1"] == 100

        # Add without path (should generate one)
        registry.add_flops(50)
        assert registry.total_flops == 150
        assert len(registry.flops_logs) == 2
        assert registry.flops_logs[1][0] == 50
        assert "unnamed_flop" in registry.flops_logs[1][1]

        # Test negative value handling
        with pytest.raises(ValueError):
            registry.add_flops(-10, "op2")

        # Test path already exists handling - should now raise ValueError
        with pytest.raises(ValueError):
            registry.add_flops(30, "op1")

    def test_add_params(self):
        """Test adding parameters."""
        registry = ModelStatsRegistry()

        # Add with path
        registry.add_params(1000, "layer1")
        assert registry.total_params == 1000
        assert len(registry.params_logs) == 1
        assert registry.params_logs[0][0] == 1000
        assert registry.params_logs[0][1] == "layer1"
        assert registry.params_logs[0][2] == 0  # level derived from path
        assert "layer1" in registry.params_by_module
        assert registry.params_by_module["layer1"] == 1000

        # Add without path (should generate one)
        registry.add_params(500)
        assert registry.total_params == 1500
        assert len(registry.params_logs) == 2
        assert registry.params_logs[1][0] == 500
        assert "unnamed_param" in registry.params_logs[1][1]

        # Test negative value handling
        with pytest.raises(ValueError):
            registry.add_params(-100, "layer2")

    def test_add_acts(self):
        """Test adding activations."""
        registry = ModelStatsRegistry()

        # Add with path
        registry.add_acts(200, "activation1")
        assert registry.total_acts == 200
        assert len(registry.acts_logs) == 1
        assert registry.acts_logs[0][0] == 200
        assert registry.acts_logs[0][1] == "activation1"
        assert registry.acts_logs[0][2] == 0  # level derived from path
        assert "activation1" in registry.acts_by_module
        assert registry.acts_by_module["activation1"] == 200

        # Add without path (should generate one)
        registry.add_acts(300)
        assert registry.total_acts == 500
        assert len(registry.acts_logs) == 2
        assert registry.acts_logs[1][0] == 300
        assert "unnamed_act" in registry.acts_logs[1][1]

        # Test negative value handling
        with pytest.raises(ValueError):
            registry.add_acts(-50, "activation2")

    def test_reset(self):
        """Test resetting the registry."""
        registry = ModelStatsRegistry()

        # Add some metrics
        registry.add_flops(100, "op1")
        registry.add_params(200, "layer1")
        registry.add_acts(300, "activation1")

        # Verify that metrics were added
        assert registry.total_flops == 100
        assert registry.total_params == 200
        assert registry.total_acts == 300
        assert len(registry.flops_logs) == 1
        assert len(registry.params_logs) == 1
        assert len(registry.acts_logs) == 1
        assert "op1" in registry.flops_by_module
        assert "layer1" in registry.params_by_module
        assert "activation1" in registry.acts_by_module
        assert registry.flops_counter == 1
        assert registry.params_counter == 1
        assert registry.acts_counter == 1
        assert len(registry.module_ids) == 3

        # Reset
        registry.reset()

        # Verify that everything was reset
        assert registry.total_flops == 0
        assert registry.total_params == 0
        assert registry.total_acts == 0
        assert len(registry.flops_logs) == 0
        assert len(registry.params_logs) == 0
        assert len(registry.acts_logs) == 0
        assert len(registry.flops_by_module) == 0
        assert len(registry.params_by_module) == 0
        assert len(registry.acts_by_module) == 0
        assert registry.flops_counter == 0
        assert registry.params_counter == 0
        assert registry.acts_counter == 0
        assert len(registry.module_ids) == 0

    def test_hierarchical_paths(self):
        """Test handling of hierarchical paths."""
        registry = ModelStatsRegistry()

        # Add metrics with hierarchical paths
        registry.add_flops(100, "parent")
        registry.add_params(200, "parent/child1")
        registry.add_acts(300, "parent/child1/grandchild")
        registry.add_flops(150, "parent/child2")

        # Verify modules are tracked correctly
        assert "parent" in registry.flops_by_module
        assert "parent/child1" in registry.params_by_module
        assert "parent/child1/grandchild" in registry.acts_by_module
        assert "parent/child2" in registry.flops_by_module

        # Verify levels are derived correctly
        assert registry.flops_logs[0][2] == 0  # parent: 0 slashes
        assert registry.params_logs[0][2] == 1  # parent/child1: 1 slash
        assert registry.acts_logs[0][2] == 2  # parent/child1/grandchild: 2 slashes
        assert registry.flops_logs[1][2] == 1  # parent/child2: 1 slash

    @patch("builtins.print")
    def test_print_logs(self, mock_print):
        """Test print_logs method."""
        registry = ModelStatsRegistry("test_model")

        # Add metrics with hierarchical structure
        registry.add_flops(100, "parent")
        registry.add_params(200, "parent/child1")
        registry.add_acts(300, "parent/child1/grandchild")
        registry.add_flops(150, "parent/child2")

        # Test printing all metrics
        registry.print_logs()
        mock_print.assert_called()

        # Test printing specific metric type
        mock_print.reset_mock()
        registry.print_logs(metric_type="flops")
        mock_print.assert_called()

        # Test printing with summary
        mock_print.reset_mock()
        registry.print_logs(include_summary=True)
        mock_print.assert_called()

        # Test with invalid metric type
        with pytest.raises(ValueError):
            registry.print_logs(metric_type="invalid")

    def test_hierarchical_module_tracking(self):
        """Test hierarchical module tracking across registries."""
        # Register model
        register_model("hierarchy_test")

        # Create hierarchical module structure with a more complex structure
        class RootModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="hierarchy_test")
                self.child1 = ChildModule()
                self.child2 = ChildModule()

            def add_flops(self, *args, **kwargs):
                return 100

            def forward(self, *args, **kwargs):
                # Call children
                self.child1()
                self.child2()
                return None

        class ChildModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="hierarchy_test")
                self.grandchild = GrandchildModule()

            def add_params(self, *args, **kwargs):
                return 200

            def forward(self, *args, **kwargs):
                # Call grandchild
                self.grandchild()
                return None

        class GrandchildModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="hierarchy_test")

            def add_acts(self, *args, **kwargs):
                return 300

            def forward(self, *args, **kwargs):
                return None

        # Create root module and call it
        root = RootModule()
        root()

        # Get registry
        registry = get_registry("hierarchy_test")

        # Verify hierarchical paths were created correctly
        root_path = None
        child_paths = []
        grandchild_paths = []

        for log in registry.flops_logs + registry.params_logs + registry.acts_logs:
            path = log[1]
            if "RootModule_" in path and "/" not in path:
                root_path = path
            elif "ChildModule_" in path and "/" in path and path.count("/") == 1:
                if path not in child_paths:
                    child_paths.append(path)
            elif "GrandchildModule_" in path and path.count("/") == 2:
                if path not in grandchild_paths:
                    grandchild_paths.append(path)

        assert root_path is not None, "Root module path not found"
        assert (
            len(child_paths) == 2
        ), f"Expected 2 child paths, got {len(child_paths)}: {child_paths}"
        assert (
            len(grandchild_paths) == 2
        ), f"Expected 2 grandchild paths, got {len(grandchild_paths)}: {grandchild_paths}"

        # Verify each child path has correct parent
        for child_path in child_paths:
            assert child_path.startswith(
                root_path + "/"
            ), f"Child path {child_path} doesn't start with {root_path}/"

        # Verify each grandchild path has correct parent
        for grandchild_path in grandchild_paths:
            # Extract the parent path by removing the last component
            parent_path = "/".join(grandchild_path.split("/")[:-1])
            assert (
                parent_path in child_paths
            ), f"Grandchild's parent {parent_path} not in child_paths: {child_paths}"

        # Verify levels are derived correctly (now based on path)
        for log in registry.flops_logs:
            path = log[1]
            level = log[2]
            assert level == path.count(
                "/"
            ), f"Path {path} has wrong level {level}, expected {path.count('/')}"

        for log in registry.params_logs:
            path = log[1]
            level = log[2]
            assert level == path.count(
                "/"
            ), f"Path {path} has wrong level {level}, expected {path.count('/')}"

        for log in registry.acts_logs:
            path = log[1]
            level = log[2]
            assert level == path.count(
                "/"
            ), f"Path {path} has wrong level {level}, expected {path.count('/')}"

        # Verify correct metrics were added at each level
        for log in registry.flops_logs:
            path = log[1]
            if "RootModule_" in path and "/" not in path:
                assert log[0] == 100, f"Root module should have 100 FLOPs, got {log[0]}"

        for log in registry.params_logs:
            print(log)
            path = log[1]
            if "RootModule_1/ChildModule_2" == path:
                assert log[0] == 200, f"Child module should have 200 params, got {log[0]}"

        for log in registry.acts_logs:
            path = log[1]
            if "GrandchildModule_" in path:
                assert log[0] == 300, f"Grandchild module should have 300 acts, got {log[0]}"
