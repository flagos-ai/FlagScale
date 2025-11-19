from unittest.mock import MagicMock

import pytest

from flagscale.runner.estimator.meta_mlp import MLP, SwiGLUMLP
from flagscale.runner.estimator.meta_modules import GELU, Linear, SwiGLU
from flagscale.runner.estimator.meta_registry import get_registry, register_model
from flagscale.runner.estimator.meta_tensor import MetaTensor


# Setup function to be run before any tests
def setup_module():
    """Register default model ID for all tests."""
    try:
        register_model("default")
    except ValueError:
        pass  # Already registered


@pytest.fixture
def config():
    """Fixture to create a mock config object."""
    config = MagicMock()
    config.hidden_size = 768
    config.ffn_hidden_size = 3072
    config.ffn_hidden_size_swiglu = 2048
    config.add_linear_bias = True
    config.tensor_parallel_size = 1
    config.activation_func = "gelu"
    return config


@pytest.fixture
def input_tensor():
    """Fixture to create a standard input tensor for testing."""
    # Create tensor with correct shard_spec format (one element per dimension)
    return MetaTensor([32, 128, 768], shard_spec=[1, 1, 1])


class TestMLP:
    """Test suite for MLP module."""

    def test_init(self, config):
        """Test initialization of MLP module."""
        # Create a new MLP with the config fixture
        get_registry("default").reset()
        mlp = MLP(config)

        # Check instance attributes
        assert mlp.config == config
        assert mlp.model_id == "default"

        # Check sub-modules
        assert isinstance(mlp.fc1, Linear)
        assert mlp.fc1.in_features == config.hidden_size
        assert mlp.fc1.out_features == config.ffn_hidden_size
        assert mlp.fc1.bias == config.add_linear_bias
        assert mlp.fc1.shard_specs == [[1, config.tensor_parallel_size]]

        assert isinstance(mlp.gelu, GELU)
        assert mlp.gelu.approximate == config.activation_func

        assert isinstance(mlp.fc2, Linear)
        assert mlp.fc2.in_features == config.ffn_hidden_size
        assert mlp.fc2.out_features == config.hidden_size
        assert mlp.fc2.bias == config.add_linear_bias
        assert mlp.fc2.shard_specs == [[config.tensor_parallel_size, 1]]

    def test_forward(self, config, input_tensor):
        """Test forward pass through MLP."""
        # Reset registry and create MLP
        get_registry("default").reset()
        mlp = MLP(config)

        # Call forward method
        output = mlp(input_tensor)

        # Verify output shape
        assert output.shape == [32, 128, 768]
        # Verify output shard_spec matches input
        assert output.shard_spec == input_tensor.shard_spec

    def test_get_flops(self, config, input_tensor):
        """Test FLOPs computation for MLP."""
        # Reset registry and create MLP
        registry = get_registry("default")
        registry.reset()
        mlp = MLP(config)

        # Call forward to populate registry
        output = mlp(input_tensor)

        # Verify FLOPs are non-zero and reasonable
        assert registry.total_flops > 0

    def test_get_params(self, config, input_tensor):
        """Test parameter count computation for MLP."""
        # Reset registry and create MLP
        registry = get_registry("default")
        registry.reset()

        mlp = MLP(config)

        # Call forward to populate registry
        output = mlp(input_tensor)

        # Get params from registry
        params = registry.total_params

        # Expected parameters:
        # - fc1: 768*3072 + 3072 (bias) = 2,362,368
        # - fc2: 3072*768 + 768 (bias) = 2,360,064
        # Total: ~4.7M
        expected_params = (768 * 3072 + 3072) + (3072 * 768 + 768)

        # Allow some flexibility for how bias terms are calculated
        assert params == expected_params

    def test_get_acts(self, config, input_tensor):
        """Test activation memory computation for MLP."""
        # Reset registry and create MLP
        registry = get_registry("default")
        registry.reset()
        mlp = MLP(config)

        # Call forward to populate registry
        output = mlp(input_tensor)

        # Get activations from registry
        acts = registry.total_acts

        # Verify activations are non-zero and reasonable
        assert acts > 0
