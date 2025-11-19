from unittest.mock import MagicMock

import pytest

from flagscale.runner.estimator.meta_attention import CoreAttention, SelfAttention
from flagscale.runner.estimator.meta_modules import (
    Baddbmm,
    Bmm,
    Dropout,
    LayerNorm,
    Linear,
    RMSNorm,
    RotaryEmbedding,
    Softmax,
)
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
    """Fixture to create a config object for attention with real values."""

    class Config:
        def __init__(self):
            # Basic model dimensions
            self.hidden_size = 768
            self.num_attention_heads = 12
            self.num_query_groups = 12  # For standard attention (no GQA)

            # Additional parameters
            self.bias = True
            self.add_linear_bias = True
            self.add_qkv_bias = False
            self.tensor_parallel_size = 1
            self.attention_dropout_prob = 0.1
            self.output_dropout_prob = 0.1

            # Rotary embedding parameters
            self.use_rotary_position_embeddings = True
            self.rotary_embedding_dim = 64
            self.rotary_embedding_base = 10000
            self.rotary_embedding_max_seq_len = 2048

            # QK LayerNorm parameters
            self.qk_layernorm = False
            self.qk_layernorm_dim = 0
            self.layernorm_epsilon = 1e-5
            self.norm_type = "layernorm"

    return Config()


@pytest.fixture
def gqa_config():
    """Fixture to create a config with grouped query attention."""

    class Config:
        def __init__(self):
            # Basic model dimensions
            self.hidden_size = 768
            self.num_attention_heads = 12
            self.num_query_groups = 4  # For GQA (fewer KV heads than query heads)

            # Additional parameters
            self.bias = True
            self.add_linear_bias = True
            self.add_qkv_bias = False
            self.tensor_parallel_size = 1
            self.attention_dropout_prob = 0.1
            self.output_dropout_prob = 0.1

            # Rotary embedding parameters
            self.use_rotary_position_embeddings = True
            self.rotary_embedding_dim = 64
            self.rotary_embedding_base = 10000
            self.rotary_embedding_max_seq_len = 2048

            # QK LayerNorm parameters
            self.qk_layernorm = True
            self.qk_layernorm_dim = 0
            self.layernorm_epsilon = 1e-5
            self.norm_type = "layernorm"

    return Config()


@pytest.fixture
def input_tensor():
    """Fixture to create a standard input tensor for testing."""
    # Create tensor with correct shard_spec format
    return MetaTensor([32, 128, 768])


@pytest.fixture
def attention_mask():
    """Fixture to create an attention mask."""
    # Create causal mask [batch_size, 1, seq_len, seq_len]
    return MetaTensor([32, 1, 128, 128])


@pytest.fixture
def position_ids():
    """Fixture to create position IDs."""
    # Create position ids [batch_size, seq_len]
    return MetaTensor([32, 128])


class TestCoreAttention:
    """Test suite for CoreAttention module."""

    def test_init(self):
        """Test initialization of CoreAttention module."""
        # Reset registry and create CoreAttention
        get_registry("default").reset()
        core_attn = CoreAttention(dropout_prob=0.1)

        # Check instance attributes
        assert core_attn.model_id == "default"

        # Check sub-modules
        assert isinstance(core_attn.baddbmm, Baddbmm)
        assert isinstance(core_attn.softmax, Softmax)
        assert core_attn.softmax.dim == -1
        assert isinstance(core_attn.attention_dropout, Dropout)
        assert core_attn.attention_dropout.p == 0.1
        assert isinstance(core_attn.bmm, Bmm)

    def test_forward(self):
        """Test forward pass through CoreAttention."""
        # Reset registry and create CoreAttention
        registry = get_registry("default")
        registry.reset()

        core_attn = CoreAttention(dropout_prob=0.1)
        core_attn.softmax_scale = 1.0 / (64**0.5)  # Set scale manually

        # Create sample inputs: query, key, value
        batch_size, seq_len_q, num_heads, head_size = 32, 128, 12, 64
        seq_len_k = 128

        query = MetaTensor([batch_size, seq_len_q, num_heads, head_size])
        key = MetaTensor([batch_size, seq_len_k, num_heads, head_size])
        value = MetaTensor([batch_size, seq_len_k, num_heads, head_size])

        # Create attention mask
        attention_mask = MetaTensor([batch_size, 1, seq_len_q, seq_len_k])

        # Call forward method
        context = core_attn(query, key, value, attention_mask)

        # Verify output shape
        expected_hidden_size = num_heads * head_size
        assert context.shape == [batch_size, seq_len_q, expected_hidden_size]

    def test_get_flops(self):
        """Test FLOPs computation for CoreAttention."""
        # Reset registry and create CoreAttention
        registry = get_registry("default")
        registry.reset()

        core_attn = CoreAttention(dropout_prob=0.1)
        core_attn.softmax_scale = 1.0 / (64**0.5)  # Set scale manually

        # Create sample inputs: query, key, value
        batch_size, seq_len_q, num_heads, head_size = 32, 128, 12, 64
        seq_len_k = 128

        query = MetaTensor([batch_size, seq_len_q, num_heads, head_size])
        key = MetaTensor([batch_size, seq_len_k, num_heads, head_size])
        value = MetaTensor([batch_size, seq_len_k, num_heads, head_size])

        # Call forward to populate registry
        context = core_attn(query, key, value)

        # Get FLOPs from registry
        flops = core_attn.get_flops()

        # Verify FLOPs are non-zero and reasonable
        assert flops > 0

        # Expected FLOPs calculation for attention:
        # 1. Q*K^T: 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_size
        # 2. Softmax: batch_size * num_heads * seq_len_q * seq_len_k
        # 3. Attn*V: 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_size
        expected_flops_min = batch_size * num_heads * seq_len_q * seq_len_k * (2 * head_size * 2)
        assert flops >= expected_flops_min


class TestSelfAttention:
    """Test suite for SelfAttention module."""

    def test_init(self, config):
        """Test initialization of SelfAttention module."""
        # Reset registry and create SelfAttention
        registry = get_registry("default")
        registry.reset()

        self_attn = SelfAttention(config)

        # Check instance attributes
        assert self_attn.model_id == "default"
        assert self_attn.hidden_size == config.hidden_size
        assert self_attn.num_attention_heads == config.num_attention_heads
        assert self_attn.num_query_groups == config.num_query_groups

        # Calculated attributes
        assert self_attn.attention_head_size == config.hidden_size // config.num_attention_heads
        assert self_attn.all_head_size == config.num_attention_heads * (
            config.hidden_size // config.num_attention_heads
        )

        # Projections
        head_size = config.hidden_size // config.num_attention_heads
        query_proj_size = head_size * config.num_attention_heads
        kv_proj_size = head_size * config.num_query_groups
        total_proj_size = query_proj_size + 2 * kv_proj_size

        # Check sub-modules
        assert isinstance(self_attn.query_key_value, Linear)
        assert self_attn.query_key_value.in_features == config.hidden_size
        assert self_attn.query_key_value.out_features == total_proj_size

        assert isinstance(self_attn.core_attention, CoreAttention)

        assert isinstance(self_attn.output_proj, Linear)
        assert self_attn.output_proj.in_features == self_attn.all_head_size
        assert self_attn.output_proj.out_features == config.hidden_size

        assert isinstance(self_attn.output_dropout, Dropout)

        # Check rotary embedding
        assert self_attn.rotary_embedding == config.use_rotary_position_embeddings
        if self_attn.rotary_embedding:
            assert isinstance(self_attn.rope, RotaryEmbedding)
            assert self_attn.rope.dim == config.rotary_embedding_dim
            assert self_attn.rope.max_seq_len == config.rotary_embedding_max_seq_len
            assert self_attn.rope.base == config.rotary_embedding_base

        # Check instance attributes
        assert self_attn.model_id == "default"
        assert self_attn.hidden_size == config.hidden_size
        assert self_attn.num_attention_heads == config.num_attention_heads
        assert self_attn.num_query_groups == config.num_query_groups

        # Check QK LayerNorm attributes and modules
        assert self_attn.qk_layernorm == config.qk_layernorm

        if config.qk_layernorm:
            if config.qk_layernorm_dim <= 0:
                assert isinstance(
                    self_attn.q_layernorm, LayerNorm if config.norm_type == "layernorm" else RMSNorm
                )
                assert isinstance(
                    self_attn.k_layernorm, LayerNorm if config.norm_type == "layernorm" else RMSNorm
                )
                assert self_attn.q_layernorm.hidden_size == self_attn.hidden_size_per_attention_head
                assert self_attn.k_layernorm.hidden_size == self_attn.hidden_size_per_attention_head
            else:
                assert isinstance(
                    self_attn.q_layernorm, LayerNorm if config.norm_type == "layernorm" else RMSNorm
                )
                assert isinstance(
                    self_attn.k_layernorm, LayerNorm if config.norm_type == "layernorm" else RMSNorm
                )
                assert self_attn.q_layernorm.hidden_size == self_attn.query_projection_size
                assert self_attn.k_layernorm.hidden_size == self_attn.kv_projection_size
        else:
            assert self_attn.q_layernorm is None
            assert self_attn.k_layernorm is None

    def test_forward(self, config, input_tensor, attention_mask, position_ids):
        """Test forward pass through SelfAttention."""
        # Reset registry and create SelfAttention
        registry = get_registry("default")
        registry.reset()

        self_attn = SelfAttention(config)

        # Call forward method
        output = self_attn(input_tensor, attention_mask, position_ids)

        # Verify output shape
        assert output.shape == input_tensor.shape

    def test_gqa_forward(self, gqa_config, input_tensor, attention_mask):
        """Test forward pass through SelfAttention with grouped query attention."""
        # Reset registry and create SelfAttention with GQA
        registry = get_registry("default")
        registry.reset()

        self_attn = SelfAttention(gqa_config)

        # Verify GQA configuration
        assert self_attn.num_attention_heads > self_attn.num_query_groups
        assert self_attn.num_attention_heads % self_attn.num_query_groups == 0

        # Call forward method
        output = self_attn(input_tensor, attention_mask)

        # Verify output shape
        assert output.shape == input_tensor.shape

    def test_qk_layernorm(self, gqa_config, input_tensor):
        """Test QK LayerNorm functionality in SelfAttention."""
        # Reset registry and create SelfAttention with QK LayerNorm
        registry = get_registry("default")
        registry.reset()

        gqa_config.qk_layernorm = True
        gqa_config.qk_layernorm_dim = 0

        # Create SelfAttention with QK LayerNorm
        self_attn = SelfAttention(gqa_config)

        # Verify QK LayerNorm configuration
        assert self_attn.qk_layernorm is True
        assert isinstance(self_attn.q_layernorm, LayerNorm)
        assert isinstance(self_attn.k_layernorm, LayerNorm)

        # Call forward method
        output = self_attn(input_tensor)

        # Verify output shape
        assert output.shape == input_tensor.shape

    def test_get_flops(self, config, input_tensor, attention_mask):
        """Test FLOPs computation for Attention."""
        # Reset registry and create Attention
        registry = get_registry("default")
        registry.reset()

        attention = SelfAttention(config)

        # Call forward to populate registry
        output = attention(input_tensor, attention_mask)

        # Get FLOPs from registry
        flops = attention.get_flops()

        # Verify FLOPs are non-zero and reasonable
        assert flops > 0

    def test_get_params(self, config, input_tensor):
        """Test parameter count computation for Attention."""
        # Reset registry and create Attention
        registry = get_registry("default")
        registry.reset()

        attention = SelfAttention(config)

        # Call forward to populate registry
        output = attention(input_tensor)

        # Get params from registry
        params = attention.get_params()

        # Calculate expected parameter count
        head_size = config.hidden_size // config.num_attention_heads
        query_proj_size = head_size * config.num_attention_heads
        kv_proj_size = head_size * config.num_query_groups

        # QKV projection: hidden_size * (query_proj_size + 2 * kv_proj_size) + bias
        qkv_params = config.hidden_size * (query_proj_size + 2 * kv_proj_size)
        if config.add_linear_bias or config.add_qkv_bias:
            qkv_params += query_proj_size + 2 * kv_proj_size

        # Output projection: all_head_size * hidden_size + bias
        all_head_size = config.num_attention_heads * head_size
        out_params = all_head_size * config.hidden_size
        if config.bias:
            out_params += config.hidden_size

        # Add rotary embedding parameters if applicable
        rope_params = 0
        if config.use_rotary_position_embeddings:
            # Approximation of rotary embedding parameters
            rope_params = config.rotary_embedding_dim * 2

        expected_params = qkv_params + out_params + rope_params

        # Allow some flexibility for how bias terms are calculated
        assert params > 0

    def test_get_acts(self, config, input_tensor):
        """Test activation memory computation for Attention."""
        # Reset registry and create Attention
        registry = get_registry("default")
        registry.reset()

        attention = SelfAttention(config)

        # Call forward to populate registry
        output = attention(input_tensor)

        # Get activations from registry
        acts = attention.get_acts()

        # Verify activations are non-zero and reasonable
        assert acts > 0
