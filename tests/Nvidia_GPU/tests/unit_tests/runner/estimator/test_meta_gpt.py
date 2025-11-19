from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.estimator.meta_gpt import GPTModel
from flagscale.runner.estimator.meta_modules import Dropout, Embedding, LayerNorm, Linear, RMSNorm
from flagscale.runner.estimator.meta_registry import get_registry, register_model
from flagscale.runner.estimator.meta_tensor import MetaTensor, ShardedDim
from flagscale.runner.estimator.meta_transformer_layer import TransformerLayer


# Set up model registry for tests
def setup_module():
    """Register test models for all tests."""
    try:
        register_model("default")
        register_model("test_model")
    except ValueError:
        pass  # Already registered


class GPTConfig:
    """Configuration class for GPT model tests."""

    def __init__(self):
        # Core architecture parameters
        self.hidden_size = 768
        self.num_layers = 3  # Small number for faster tests
        self.vocab_size = 50257
        self.max_position_embeddings = 1024
        self.pad_token_id = 0

        # Attention parameters
        self.num_attention_heads = 12
        self.num_query_groups = 12  # Standard attention (not GQA)
        self.kv_channels = self.hidden_size // self.num_attention_heads  # Per head size

        # MLP configuration
        self.ffn_hidden_size = 3072
        self.activation_func = "gelu"

        # Normalization
        self.pre_normalization = True
        self.norm_type = "layernorm"
        self.layernorm_epsilon = 1e-5
        self.qk_layernorm = False
        self.qk_layernorm_dim = 0

        # Attention configuration
        self.use_rotary_position_embeddings = False
        self.rotary_embedding_dim = 0
        self.rotary_embedding_base = 10000
        self.rotary_embedding_max_seq_len = 2048
        self.attention_dropout_prob = 0.1
        self.hidden_dropout = 0.1
        self.embedding_dropout = 0.1
        self.untie_embeddings_and_output_weights = False

        # Parallelism
        self.tensor_parallel_size = 1
        self.sequence_parallel = False

        # Misc
        self.bias = True
        self.add_linear_bias = True
        self.add_qkv_bias = True
        self.no_bias_in_output = True  # Most GPT models don't use bias in output layer
        self.initialize_at_zero = False


@pytest.fixture
def config():
    """Fixture to provide a standard GPT model config."""
    return GPTConfig()


@pytest.fixture
def rotary_config():
    """Fixture to provide a GPT model config with rotary embeddings."""
    config = GPTConfig()
    config.use_rotary_position_embeddings = True
    config.rotary_embedding_dim = 64
    config.rotary_embedding_base = 10000
    config.rotary_embedding_max_seq_len = 2048
    config.position_embedding_type = "rotary"  # Explicitly set the position embedding type
    return config


@pytest.fixture
def input_ids():
    """Create sample input token IDs."""
    batch_size, seq_len = 2, 16
    return MetaTensor([batch_size, seq_len])


@pytest.fixture
def attention_mask():
    """Create a sample attention mask."""
    batch_size, seq_len = 2, 16
    return MetaTensor([batch_size, 1, seq_len, seq_len])


@pytest.fixture
def position_ids():
    """Create sample position IDs."""
    batch_size, seq_len = 2, 16
    return MetaTensor([batch_size, seq_len])


class TestGPTModel:
    """Test suite for GPTModel."""

    def test_init_standard(self, config):
        """Test initialization with standard configuration."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Check basic attributes
        assert model.hidden_size == config.hidden_size
        assert len(model.layers) == config.num_layers
        assert model.vocab_size == config.vocab_size

        # Check embedding components
        assert isinstance(model.word_embeddings, Embedding)
        assert model.word_embeddings.num_embeddings == config.vocab_size
        assert model.word_embeddings.embedding_dim == config.hidden_size

        # Check position embeddings
        assert not model.use_rotary_position_embeddings
        assert isinstance(model.position_embeddings, Embedding)
        assert model.position_embeddings.num_embeddings == config.max_position_embeddings
        assert model.position_embeddings.embedding_dim == config.hidden_size

        # Check transformer layers
        assert len(model.layers) == config.num_layers
        for i, layer in enumerate(model.layers):
            assert isinstance(layer, TransformerLayer)
            assert layer.layer_number == i

        # Check final normalization
        assert isinstance(model.final_norm, LayerNorm)
        assert model.final_norm.hidden_size == config.hidden_size

        # Check output layer
        assert isinstance(model.output_layer, Linear)
        assert model.output_layer.in_features == config.hidden_size
        assert model.output_layer.out_features == config.vocab_size
        assert model.output_layer.bias is False  # GPT typically doesn't use bias in output

        # Check model_id propagation
        assert model.model_id == "test_model"
        assert model.word_embeddings.model_id == "test_model"
        assert model.position_embeddings.model_id == "test_model"
        assert model.final_norm.model_id == "test_model"
        assert model.output_layer.model_id == "test_model"
        for layer in model.layers:
            assert layer.model_id == "test_model"

    def test_init_rotary(self, rotary_config):
        """Test initialization with rotary position embeddings."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model with rotary embeddings
        model = GPTModel(rotary_config, model_id="test_model")

        # Check rotary embedding configuration
        assert model.use_rotary_position_embeddings is True
        assert model.position_embeddings is None

        # Check that layers have rotary embeddings configured
        for layer in model.layers:
            assert layer.self_attention.rotary_embedding is True

    def test_forward(self, config, input_ids, attention_mask, position_ids):
        """Test forward pass through GPT model."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Forward pass
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )

        # Check output shape
        batch_size, seq_len = input_ids.shape
        assert logits.shape == [batch_size, seq_len, config.vocab_size]

        # Verify resources are tracked in registry
        assert len(registry.flops_logs) > 0
        assert len(registry.params_logs) > 0
        assert len(registry.acts_logs) > 0

    def test_forward_without_position_ids(self, config, input_ids, attention_mask):
        """Test forward pass without explicitly providing position IDs."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Forward pass without position_ids
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output shape
        batch_size, seq_len = input_ids.shape
        assert logits.shape == [batch_size, seq_len, config.vocab_size]

    def test_forward_rotary(self, rotary_config, input_ids, attention_mask, position_ids):
        """Test forward pass with rotary position embeddings."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model with rotary embeddings
        model = GPTModel(rotary_config, model_id="test_model")

        # Forward pass
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )

        # Check output shape
        batch_size, seq_len = input_ids.shape
        assert logits.shape == [batch_size, seq_len, rotary_config.vocab_size]

    def test_get_params(self, config, input_ids):
        """Test parameter counting for GPT model."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Run a forward pass to register parameters
        model(input_ids=input_ids)

        # Get parameter count
        params = model.get_params()

        # Calculate expected parameter count
        # Word embeddings: vocab_size * hidden_size
        word_emb_params = config.vocab_size * config.hidden_size

        # Position embeddings: max_seq_len * hidden_size
        pos_emb_params = config.max_position_embeddings * config.hidden_size

        # Final layer norm: 2 * hidden_size (weights + bias)
        norm_params = 2 * config.hidden_size

        # We don't calculate transformer layers here as they're tested separately

        # The total should be at least this much
        min_expected_params = word_emb_params + pos_emb_params + norm_params

        # Verify parameter count is reasonable
        assert params > min_expected_params

    def test_get_acts(self, config, input_ids):
        """Test activation memory estimation for GPT model."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Run a forward pass to register activations
        model(input_ids=input_ids)

        # Get activation count
        acts = model.get_acts()

        assert acts > 0

    def test_get_flops(self, config, input_ids, attention_mask):
        """Test FLOPs estimation for GPT model."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Run a forward pass to register FLOPs
        model(input_ids=input_ids, attention_mask=attention_mask)

        # Get FLOP count
        flops = model.get_flops()

        # We don't calculate expected FLOPs in detail here
        # Just verify that the count is non-zero and reasonable
        assert flops > 0

        # For a typical GPT model, FLOPs should be at least billions
        # for even small batch sizes, but since we're using a tiny test model
        # we just check it's positive

    def test_registry_updates(self, config, input_ids, attention_mask):
        """Test registry updates during forward pass."""
        # Reset registry for clean test
        registry = get_registry("test_model")
        registry.reset()

        # Create GPT model
        model = GPTModel(config, model_id="test_model")

        # Run a forward pass
        model(input_ids=input_ids, attention_mask=attention_mask)

        # Check that registry has entries for important components
        paths = [log[1] for log in registry.flops_logs]

        # Print summary for visual inspection
        registry.print_logs(include_summary=True)

    def test_with_mocked_layers(self, config, input_ids):
        """Test GPTModel with mocked layers to verify interactions."""
        # Create mocks for components
        mock_word_embeddings = MagicMock()
        mock_word_embeddings.return_value = MetaTensor([2, 16, 768])

        mock_pos_embeddings = MagicMock()
        mock_pos_embeddings.return_value = MetaTensor([2, 16, 768])

        mock_layer1 = MagicMock()
        mock_layer1.return_value = MetaTensor([2, 16, 768])

        mock_layer2 = MagicMock()
        mock_layer2.return_value = MetaTensor([2, 16, 768])

        mock_final_norm = MagicMock()
        mock_final_norm.return_value = MetaTensor([2, 16, 768])

        mock_output_layer = MagicMock()
        mock_output_layer.return_value = MetaTensor([2, 16, 1000])

        # Create model and replace components with mocks
        model = GPTModel(config)
        model.word_embeddings = mock_word_embeddings
        model.position_embeddings = mock_pos_embeddings
        model.layers = [mock_layer1, mock_layer2]
        model.final_norm = mock_final_norm
        model.output_layer = mock_output_layer

        # Forward pass
        output = model(input_ids=input_ids)

        # Verify component interactions
        mock_word_embeddings.assert_called_once()
        mock_pos_embeddings.assert_called_once()
        mock_layer1.assert_called_once()
        mock_layer2.assert_called_once()
        mock_final_norm.assert_called_once()
        mock_output_layer.assert_called_once()

        # Verify output shape
        assert output.shape == [2, 16, 1000]
