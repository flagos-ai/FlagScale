import os
import sys

from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.estimator.meta_base import MetaModule
from flagscale.runner.estimator.meta_registry import get_registry, register_model, reset_registry


class TestMetaModule:
    """Test suite for MetaModule class."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Reset registries before each test
        import flagscale.runner.estimator.meta_registry as meta_registry

        meta_registry._model_registries = {}

        # Reset module counter and path for predictable tests
        MetaModule._counter = 0
        MetaModule._path = None

        # Register models used in tests
        register_model("default")
        register_model("test_model")

    def test_init(self):
        """Test initialization."""
        # Basic init
        module = MetaModule()
        assert module.shard_specs is None
        assert module.model_id == "default"

        # With shard_specs and model_id
        module = MetaModule([[1, 2], [3, 4]], "test_model")
        assert module.shard_specs == [[1, 2], [3, 4]]
        assert module.model_id == "test_model"

    def test_compute_methods(self):
        """Test default compute_* methods."""
        module = MetaModule()

        # Default implementations return 0
        assert module.add_flops() == 0
        assert module.add_params() == 0
        assert module.add_acts() == 0

    @patch("flagscale.runner.estimator.meta_base.get_registry")
    def test_update_registry(self, mock_get_registry):
        """Test update_registry method."""
        # Setup mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry

        # Create module with compute methods that return non-zero values
        class TestModule(MetaModule):
            def add_flops(self, *args, **kwargs):
                return 100

            def add_params(self, *args, **kwargs):
                return 200

            def add_acts(self, *args, **kwargs):
                return 300

            def forward(self, *args, **kwargs):
                return "result"

        module = TestModule()

        # Important: We need to set the path manually since we're bypassing __call__
        MetaModule._path = "TestModule_1"

        # Call update_registry
        module.update_registry("arg1", arg2="value")

        # Verify registry methods were called with correct values
        mock_registry.add_flops.assert_called_once_with(100, path="TestModule_1")
        mock_registry.add_params.assert_called_once_with(200, path="TestModule_1")
        mock_registry.add_acts.assert_called_once_with(300, path="TestModule_1")

        # Reset path to avoid side effects on other tests
        MetaModule._path = None

    @patch("flagscale.runner.estimator.meta_base.get_registry")
    def test_call(self, mock_get_registry):
        """Test __call__ method."""
        # Setup mock registry
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry

        # Create module with mocked methods
        class TestModule(MetaModule):
            def add_flops(self, *args, **kwargs):
                return 100

            def add_params(self, *args, **kwargs):
                return 200

            def add_acts(self, *args, **kwargs):
                return 300

            def forward(self, *args, **kwargs):
                return "result"

        # Important: We need to set a deterministic path before testing
        module = TestModule()

        # Reset counter for consistent path naming
        MetaModule._counter = 0

        # Call the module - this will set _path, increment _counter, call update_registry,
        # then call forward, then restore _path to its previous value (None)
        result = module("arg1", arg2="value")

        # Verify registry methods were called with the correct path
        mock_registry.add_flops.assert_called_once_with(100, path="TestModule_1")
        mock_registry.add_params.assert_called_once_with(200, path="TestModule_1")
        mock_registry.add_acts.assert_called_once_with(300, path="TestModule_1")

        # Verify result is from forward method
        assert result == "result"

    def test_forward_not_implemented(self):
        """Test forward method raises NotImplementedError if not overridden."""
        module = MetaModule()
        with pytest.raises(NotImplementedError):
            module()

    def test_zero_metrics_handling(self):
        """Test handling of zero metric values."""
        # Create test registry to avoid using mock
        register_model("zero_test")
        registry = get_registry("zero_test")

        # Create module that returns zero for some metrics
        class ZeroModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="zero_test")

            def add_flops(self, *args, **kwargs):
                return 0

            def add_params(self, *args, **kwargs):
                return 100

            def add_acts(self, *args, **kwargs):
                return 0

            def forward(self, *args, **kwargs):
                return None

        module = ZeroModule()
        module()

        # Verify metrics are added to logs
        assert len(registry.flops_logs) == 1
        assert len(registry.params_logs) == 1
        assert len(registry.acts_logs) == 1

        # Verify registry totals
        assert registry.total_flops == 0
        assert registry.total_params == 100
        assert registry.total_acts == 0

    def test_nested_module_paths(self):
        """Test hierarchical path generation for nested modules."""
        register_model("nested_test")
        registry = get_registry("nested_test")

        # Define nested modules
        class InnerModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="nested_test")

            def add_flops(self, *args, **kwargs):
                return 50

            def forward(self, *args, **kwargs):
                return "inner_result"

        class OuterModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="nested_test")
                self.inner = InnerModule()

            def add_flops(self, *args, **kwargs):
                return 100

            def forward(self, *args, **kwargs):
                # Call inner module, which should create a nested path
                inner_result = self.inner()
                return f"outer_result: {inner_result}"

        # Reset counter for consistent path naming
        MetaModule._counter = 0

        # Create and call the outer module
        module = OuterModule()
        result = module()

        # Verify result
        assert result == "outer_result: inner_result"

        # Verify paths in logs - should have OuterModule_1 and OuterModule_1/InnerModule_2
        expected_paths = ["OuterModule_1", "OuterModule_1/InnerModule_2"]
        actual_paths = [log[1] for log in registry.flops_logs]

        assert set(expected_paths) == set(actual_paths)

        # Verify flop values
        assert registry.total_flops == 150  # 100 from outer + 50 from inner

    def test_shared_parameters(self):
        """Test shared parameter handling."""
        register_model("shared_test")
        registry = get_registry("shared_test")

        # Create modules that share parameters
        class SharedParamModule(MetaModule):
            def __init__(self, shared=False):
                super().__init__(model_id="shared_test")
                if shared:
                    self.share_params()

            def add_params(self, *args, **kwargs):
                return 200

            def forward(self, *args, **kwargs):
                return "shared_param_result"

        # Reset counter for consistent paths
        MetaModule._counter = 0

        # Create one original module and one shared module
        original = SharedParamModule()
        shared = SharedParamModule(shared=True)

        # Call both modules
        original()
        shared()

        # Verify only one set of parameters was counted
        assert registry.total_params == 200  # Only the original module's params

        # Verify that params_logs shows two entries but only one with non-zero value
        assert len(registry.params_logs) == 2
        values = [log[0] for log in registry.params_logs]
        assert values == [200, 0]  # Original has 200, shared has 0

    def test_get_metrics(self):
        """Test methods to get previously computed metrics."""
        register_model("get_metrics_test")

        # Create module that adds specific metric values
        class MetricsModule(MetaModule):
            def __init__(self):
                super().__init__(model_id="get_metrics_test")

            def add_flops(self, *args, **kwargs):
                return 123

            def add_params(self, *args, **kwargs):
                return 456

            def add_acts(self, *args, **kwargs):
                return 789

            def forward(self, *args, **kwargs):
                return "metrics_result"

        # Create and call module to register metrics
        module = MetricsModule()
        module()

        # Verify get_* methods return correct values
        assert module.get_flops() == 123
        assert module.get_params() == 456
        assert module.get_acts() == 789

    def test_multiple_model_registration(self):
        """Test using multiple models with separate registries."""
        # Register two test models
        register_model("model_a")
        register_model("model_b")

        registry_a = get_registry("model_a")
        registry_b = get_registry("model_b")

        # Create similar modules but for different models
        class TestModuleA(MetaModule):
            def __init__(self):
                super().__init__(model_id="model_a")

            def add_flops(self, *args, **kwargs):
                return 100

            def forward(self, *args, **kwargs):
                return "a_result"

        class TestModuleB(MetaModule):
            def __init__(self):
                super().__init__(model_id="model_b")

            def add_flops(self, *args, **kwargs):
                return 200

            def forward(self, *args, **kwargs):
                return "b_result"

        # Reset counter for consistent path naming
        MetaModule._counter = 0

        # Create and call both modules
        module_a = TestModuleA()
        module_b = TestModuleB()
        module_a()
        module_b()

        # Verify metrics are tracked separately
        assert registry_a.total_flops == 100
        assert registry_b.total_flops == 200

    def test_model_id_consistency_checks(self):
        """Test model_id consistency checks in __call__."""
        from flagscale.runner.estimator.meta_tensor import MetaTensor

        register_model("model_a")
        register_model("model_b")
        register_model("model_c")

        # Create a test module that accepts tensors and returns them
        class TestModule(MetaModule):
            def __init__(self, model_id="default"):
                super().__init__(model_id=model_id)

            def forward(self, *args, **kwargs):
                # Return the first arg for testing model_id propagation
                if args:
                    return args[0]
                return list(kwargs.values())[0] if kwargs else None

        # Test Case 1: Module with default model_id accepts tensors with non-default model_id
        module = TestModule()
        tensor_a = MetaTensor([10, 10], model_id="model_a")

        # Module should adopt tensor's model_id
        result = module(tensor_a)
        assert module.model_id == "model_a"
        assert result.model_id == "model_a"

        # Test Case 2: Module with specific model_id only accepts matching tensors
        module = TestModule(model_id="model_b")
        tensor_b = MetaTensor([10, 10], model_id="model_b")

        # Should succeed when model_ids match
        result = module(tensor_b)
        assert module.model_id == "model_b"
        assert result.model_id == "model_b"

        # Test Case 3: Error when tensors have different non-default model_ids
        tensor_a = MetaTensor([10, 10], model_id="model_a")
        tensor_c = MetaTensor([10, 10], model_id="model_c")

        # Using args
        with pytest.raises(ValueError) as excinfo:
            module = TestModule()
            module(tensor_a, tensor_c)
        assert "inconsistent model_ids" in str(excinfo.value)

        # Using kwargs
        with pytest.raises(ValueError) as excinfo:
            module = TestModule()
            module(tensor1=tensor_a, tensor2=tensor_c)
        assert "inconsistent model_ids" in str(excinfo.value)

        # Test Case 4: Error when module has specific model_id but tensor has different model_id
        module = TestModule(model_id="model_b")
        tensor_c = MetaTensor([10, 10], model_id="model_c")

        with pytest.raises(ValueError) as excinfo:
            module(tensor_c)
        assert "has model_id 'model_b', but received inputs with different model_ids" in str(
            excinfo.value
        )

        # Test Case 5: Model ID propagation to output tensors
        module = TestModule(model_id="model_a")
        tensor_default = MetaTensor([10, 10])  # default model_id

        # Output should get module's model_id
        result = module(tensor_default)
        assert result.model_id == "model_a"

    def test_model_id_propagation_warning(self):
        """Test warning when an operation produces a tensor with inconsistent model_id."""
        import warnings

        from flagscale.runner.estimator.meta_tensor import MetaTensor

        register_model("model_a")
        register_model("model_b")

        # Create a module that returns a tensor with a different model_id
        class InconsistentModule(MetaModule):
            def __init__(self, model_id="model_a"):
                super().__init__(model_id=model_id)

            def forward(self, tensor):
                # Create new tensor with a different model_id
                result = MetaTensor([10, 10], model_id="model_b")
                return result

        module = InconsistentModule(model_id="model_a")
        input_tensor = MetaTensor([5, 5], model_id="model_a")

        # Should raise warning about inconsistent model_id but still correct it
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = module(input_tensor)

            # Check warning was emitted
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "with different model_id=" in str(w[0].message)

            # Check model_id was corrected despite the warning
            assert result.model_id == "model_a"

    def test_nested_structure_model_id_handling(self):
        """Test model_id handling with nested tensor structures."""
        from flagscale.runner.estimator.meta_tensor import MetaTensor

        register_model("nested_a")
        register_model("nested_b")

        # Create a module that returns a list of tensors
        class ListReturnModule(MetaModule):
            def __init__(self, model_id="nested_a"):
                super().__init__(model_id=model_id)

            def forward(self, tensor):
                # Return a list of tensors
                return [
                    tensor,
                    MetaTensor([5, 5]),  # default model_id
                    MetaTensor(
                        [3, 3], model_id="nested_b"
                    ),  # different model_id (will trigger warning)
                ]

        module = ListReturnModule(model_id="nested_a")
        input_tensor = MetaTensor([10, 10], model_id="nested_a")

        # Process the nested structure and check all model_ids
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_list = module(input_tensor)

            # Check all tensors have correct model_id
            for i, tensor in enumerate(result_list):
                assert tensor.model_id == "nested_a", f"Tensor at index {i} has wrong model_id"

            # Check warning was emitted for the tensor with initially incorrect model_id
            assert len(w) == 1
            assert "with different model_id=" in str(w[0].message)

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global state to avoid affecting other tests
        MetaModule._path = None
        MetaModule._counter = 0
