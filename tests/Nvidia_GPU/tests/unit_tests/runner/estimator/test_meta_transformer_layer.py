from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.estimator.meta_attention import SelfAttention
from flagscale.runner.estimator.meta_mlp import MLP, SwiGLUMLP
from flagscale.runner.estimator.meta_modules import LayerNorm, RMSNorm
from flagscale.runner.estimator.meta_registry import get_registry, register_model
from flagscale.runner.estimator.meta_tensor import MetaTensor
from flagscale.runner.estimator.meta_transformer_layer import TransformerLayer


# Set up model registry for tests
def setup_module():
    """Register test models for all tests."""
    try:
        register_model("default")
        register_model("test_model")
    except ValueError:
        pass  # Already registered


@pytest.fixture
def config():
    """Fixture to provide a standard transformer config."""

    class TransformerConfig:
        def __init__(self):
            # Core architecture
            self.hidden_size = 768
            self.num_attention_heads = 12
            self.num_query_groups = 12  # For standard attention

            # MLP configuration
            self.ffn_hidden_size = 3072  # 4x hidden_size
            self.activation_func = "gelu"

            # Normalization
            self.pre_normalization = True
            self.norm_type = "layernorm"
            self.layernorm_epsilon = 1e-5

            # Attention configuration
            self.use_rotary_position_embeddings = True
            self.rotary_embedding_dim = 64
            self.rotary_embedding_base = 10000
            self.rotary_embedding_max_seq_len = 2048
            self.attention_dropout_prob = 0.1
            self.hidden_dropout = 0.1

            # Misc
            self.tensor_parallel_size = 1
            self.bias = True
            self.add_linear_bias = True
            self.add_qkv_bias = True

    return TransformerConfig()


@pytest.fixture
def alt_config():
    """Fixture to provide alternative transformer config (RMSNorm, SwiGLU, post-norm)."""

    class TransformerAltConfig:
        def __init__(self):
            # Core architecture
            self.hidden_size = 1024
            self.num_attention_heads = 16
            self.num_query_groups = 4  # GQA configuration

            # MLP configuration
            self.ffn_hidden_size = 4096
            self.activation_func = "swiglu"

            # Normalization
            self.pre_normalization = False  # Post-normalization
            self.norm_type = "rmsnorm"
            self.layernorm_epsilon = 1e-5

            # Attention configuration
            self.use_rotary_position_embeddings = True
            self.rotary_embedding_dim = 64
            self.rotary_embedding_base = 10000
            self.rotary_embedding_max_seq_len = 2048
            self.attention_dropout_prob = 0.0
            self.hidden_dropout = 0.0

            # Misc
            self.tensor_parallel_size = 1
            self.bias = False
            self.add_linear_bias = False
            self.add_qkv_bias = False

    return TransformerAltConfig()


@pytest.fixture
def input_tensor():
    """Create a sample input tensor."""
    batch_size, seq_len, hidden_size = 2, 16, 768
    return MetaTensor([batch_size, seq_len, hidden_size])


@pytest.fixture
def large_input_tensor():
    """Create a larger input tensor for the alternative config."""
    batch_size, seq_len, hidden_size = 2, 16, 1024
    return MetaTensor([batch_size, seq_len, hidden_size])


@pytest.fixture
def attention_mask():
    """Create a sample attention mask."""
    batch_size, seq_len = 2, 16
    return MetaTensor([batch_size, 1, seq_len, seq_len])


@pytest.fixture
def position_ids():
    """Create position IDs for the attention mechanism."""
    batch_size, seq_len = 2, 16
    return MetaTensor([batch_size, seq_len])


class TestTransformerLayer:
    """Test suite for TransformerLayer."""

    def test_init_standard(self, config):
        """Test initialization with standard configuration."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(config, layer_number=0, model_id="test_model")

        # Check basic attributes
        assert layer.hidden_size == config.hidden_size
        assert layer.layer_number == 0
        assert layer.pre_normalization is True
        assert layer.ffn_hidden_size == config.ffn_hidden_size

        # Check correct normalization type
        assert layer.norm_type == "layernorm"
        assert isinstance(layer.attention_norm, LayerNorm)
        assert isinstance(layer.mlp_norm, LayerNorm)

        # Check components
        assert isinstance(layer.self_attention, SelfAttention)
        assert isinstance(layer.mlp, MLP)  # Should be regular MLP with GELU

        # Check model_id propagation
        assert layer.model_id == "test_model"
        assert layer.attention_norm.model_id == "test_model"
        assert layer.mlp_norm.model_id == "test_model"
        assert layer.self_attention.model_id == "test_model"
        assert layer.mlp.model_id == "test_model"

    def test_init_alternative(self, alt_config):
        """Test initialization with alternative configuration."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(alt_config, layer_number=1, model_id="test_model")

        # Check basic attributes
        assert layer.hidden_size == alt_config.hidden_size
        assert layer.layer_number == 1
        assert layer.pre_normalization is False  # Post-normalization
        assert layer.ffn_hidden_size == alt_config.ffn_hidden_size

        # Check correct normalization type
        assert layer.norm_type == "rmsnorm"
        assert isinstance(layer.attention_norm, RMSNorm)
        assert isinstance(layer.mlp_norm, RMSNorm)

        # Check components
        assert isinstance(layer.self_attention, SelfAttention)
        assert isinstance(layer.mlp, SwiGLUMLP)  # Should be SwiGLU MLP

        # Check model_id propagation
        assert layer.model_id == "test_model"
        assert layer.attention_norm.model_id == "test_model"
        assert layer.mlp_norm.model_id == "test_model"
        assert layer.self_attention.model_id == "test_model"
        assert layer.mlp.model_id == "test_model"

    def test_forward_prenorm(self, config, input_tensor, attention_mask, position_ids):
        """Test forward pass with pre-normalization architecture."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(config, layer_number=0, model_id="test_model")

        # Forward pass
        output = layer(
            hidden_states=input_tensor, attention_mask=attention_mask, position_ids=position_ids
        )

        # Check output shape matches input shape
        assert output.shape == input_tensor.shape

        # Verify resources are tracked in registry
        assert len(registry.flops_logs) > 0
        assert len(registry.params_logs) > 0
        assert len(registry.acts_logs) > 0

    def test_forward_postnorm(self, alt_config, large_input_tensor, attention_mask, position_ids):
        """Test forward pass with post-normalization architecture."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(alt_config, layer_number=0, model_id="test_model")

        # Forward pass
        output = layer(
            hidden_states=large_input_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Check output shape matches input shape
        assert output.shape == large_input_tensor.shape

        # Verify resources are tracked in registry
        assert len(registry.flops_logs) > 0
        assert len(registry.params_logs) > 0
        assert len(registry.acts_logs) > 0

    def test_get_flops(self, config, input_tensor):
        """Test computation of direct FLOPs."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(config, layer_number=0, model_id="test_model")

        layer(input_tensor)

        # Compute FLOPs directly
        flops = layer.get_flops()

        assert flops > 0

    def test_get_params(self, config, input_tensor):
        """Test counting of direct parameters."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(config, layer_number=0, model_id="test_model")

        layer(input_tensor)

        # Count parameters directly
        params = layer.get_params()

        assert params > 0

    def test_get_acts(self, config, input_tensor):
        """Test measurement of activation memory."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(config, layer_number=0, model_id="test_model")

        # First do a forward pass to ensure the module is properly initialized
        layer(input_tensor)

        # Get activation memory using the registry method
        acts = layer.get_acts()

        # Expected memory: 2 * batch_size * seq_len * hidden_size (for residual connections)
        # plus any activations from submodules
        batch_size, seq_len, hidden_size = input_tensor.shape
        min_expected_acts = 2 * batch_size * seq_len * hidden_size

        # The activation count should be at least the residual connections
        assert acts >= min_expected_acts

    def test_registry_updates(self, config, input_tensor, attention_mask):
        """Test that the registry is properly updated during forward pass."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create transformer layer
        layer = TransformerLayer(config, layer_number=0, model_id="test_model")

        # Forward pass
        output = layer(hidden_states=input_tensor, attention_mask=attention_mask)

        # Print logs to verify hierarchical structure
        registry.print_logs(include_summary=True)

    def test_with_mocked_submodules(self, config):
        """Test TransformerLayer with mocked submodules to verify interactions."""
        # Create mocks for submodules
        mock_attention_norm = MagicMock()
        mock_attention_norm.return_value = MetaTensor([2, 16, 768])

        mock_self_attention = MagicMock()
        mock_self_attention.return_value = MetaTensor([2, 16, 768])

        mock_mlp_norm = MagicMock()
        mock_mlp_norm.return_value = MetaTensor([2, 16, 768])

        mock_mlp = MagicMock()
        mock_mlp.return_value = MetaTensor([2, 16, 768])

        # Create layer and replace submodules with mocks
        layer = TransformerLayer(config)
        layer.attention_norm = mock_attention_norm
        layer.self_attention = mock_self_attention
        layer.mlp_norm = mock_mlp_norm
        layer.mlp = mock_mlp

        # Input tensor
        hidden_states = MetaTensor([2, 16, 768])
        attention_mask = MetaTensor([2, 1, 16, 16])

        # Forward pass
        output = layer(hidden_states, attention_mask)

        # Verify submodules were called in the correct order for pre-normalization
        mock_attention_norm.assert_called_once_with(hidden_states)
        mock_self_attention.assert_called_once()
        mock_mlp_norm.assert_called_once()
        mock_mlp.assert_called_once()

        # Verify output shape
        assert output.shape == hidden_states.shape
