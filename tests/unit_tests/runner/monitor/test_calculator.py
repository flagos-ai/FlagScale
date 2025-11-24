import math
import unittest

from flagscale.runner.monitor.flops_calculator import FLOPSFormulas


class TestFLOPSFormulas(unittest.TestCase):
    """Test FLOPS calculation formulas."""

    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 8
        self.seq_length = 2048
        self.hidden_size = 4096
        self.num_heads = 32
        self.ffn_hidden_size = 16384
        self.vocab_size = 50000

    def test_attention_flops(self):
        """Test standard attention FLOPS calculation."""
        flops = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        # Verify calculation components
        head_dim = self.hidden_size // self.num_heads

        # QKV projections: 3 * 2 * batch * seq * hidden^2
        qkv_flops = 3 * 2 * self.batch_size * self.seq_length * self.hidden_size * self.hidden_size

        # Q @ K^T: 2 * batch * heads * seq * seq * head_dim
        attention_score_flops = (
            2 * self.batch_size * self.num_heads * self.seq_length * self.seq_length * head_dim
        )

        # scores @ V: 2 * batch * heads * seq * seq * head_dim
        attention_output_flops = (
            2 * self.batch_size * self.num_heads * self.seq_length * self.seq_length * head_dim
        )

        # Output projection: 2 * batch * seq * hidden^2
        output_proj_flops = (
            2 * self.batch_size * self.seq_length * self.hidden_size * self.hidden_size
        )

        expected = qkv_flops + attention_score_flops + attention_output_flops + output_proj_flops
        self.assertEqual(flops, expected)

    def test_gqa_attention_flops(self):
        """Test Grouped Query Attention FLOPS calculation."""
        num_query_groups = 8

        flops = FLOPSFormulas.gqa_attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_query_groups=num_query_groups,
        )

        self.assertGreater(flops, 0)

        # GQA should be less than standard attention
        standard_flops = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )
        self.assertLess(flops, standard_flops)

    def test_flash_attention_flops(self):
        """Test Flash Attention FLOPS calculation."""
        flops = FLOPSFormulas.flash_attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        self.assertGreater(flops, 0)

        # Flash attention has same theoretical FLOPS as standard
        standard_flops = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )
        self.assertEqual(flops, standard_flops)

    def test_ffn_flops(self):
        """Test FFN FLOPS calculation."""
        # Standard FFN (GELU activation)
        flops_gelu = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )

        # Expected: 2 * batch * seq * hidden * ffn_hidden * 2 (up and down)
        expected = (
            2 * 2 * self.batch_size * self.seq_length * self.hidden_size * self.ffn_hidden_size
        )
        self.assertEqual(flops_gelu, expected)

        # SwiGLU FFN
        flops_swiglu = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=True,
        )

        # SwiGLU has 3 projections instead of 2
        self.assertGreater(flops_swiglu, flops_gelu)

    def test_moe_ffn_flops(self):
        """Test MoE FFN FLOPS calculation."""
        num_experts = 8
        expert_capacity_factor = 1.25
        moe_router_topk = 2

        flops = FLOPSFormulas.moe_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_experts=num_experts,
            top_k=moe_router_topk,
        )

        self.assertGreater(flops, 0)

        # MoE should be more efficient than dense FFN * num_experts
        dense_flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )
        self.assertLess(flops, dense_flops * num_experts)

    def test_layer_norm_flops(self):
        """Test LayerNorm FLOPS calculation."""
        flops = FLOPSFormulas.layernorm_flops(
            batch_size=self.batch_size, seq_length=self.seq_length, hidden_size=self.hidden_size
        )

        # LayerNorm is relatively cheap: ~2 * batch * seq * hidden
        expected = 7 * self.batch_size * self.seq_length * self.hidden_size
        self.assertEqual(flops, expected)

    def test_embedding_flops(self):
        """Test embedding layer FLOPS calculation."""
        flops = FLOPSFormulas.embedding_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )

        # Embedding is a simple lookup and projection
        expected = 2 * self.batch_size * self.seq_length * self.hidden_size * (self.vocab_size + 1)
        self.assertEqual(flops, expected)

    def test_cross_entropy_loss_flops(self):
        """Test cross-entropy loss FLOPS calculation."""
        flops = FLOPSFormulas.cross_entropy_flops(
            batch_size=self.batch_size, seq_length=self.seq_length, vocab_size=self.vocab_size
        )

        # Cross-entropy involves softmax and log
        expected = self.batch_size * self.seq_length * (self.vocab_size * 3 + 1)
        self.assertEqual(flops, expected)

    def test_zero_values(self):
        """Test handling of zero values."""
        # Zero batch size
        flops = FLOPSFormulas.attention_flops(
            batch_size=0,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )
        self.assertEqual(flops, 0)

        # Zero sequence length
        flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=0,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )
        self.assertEqual(flops, 0)

    def test_edge_cases(self):
        """Test edge cases."""
        # Single batch, single token
        flops = FLOPSFormulas.attention_flops(
            batch_size=1, seq_length=1, hidden_size=128, num_attention_heads=8
        )
        self.assertGreater(flops, 0)

        # Very small model
        flops = FLOPSFormulas.embedding_flops(
            batch_size=1, seq_length=1, vocab_size=100, hidden_size=128
        )
        self.assertGreater(flops, 0)

    def test_proportionality(self):
        """Test that FLOPS scale correctly with parameters."""
        # Double batch size should double FLOPS
        flops1 = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        flops2 = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size * 2,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        self.assertAlmostEqual(flops2 / flops1, 2.0, places=5)

        # Doubling sequence length increases attention FLOPS but not exactly 4x
        # due to the mix of quadratic (attention) and linear (projection) terms
        flops3 = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length * 2,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        # Fix: Changed assertion values based on actual calculation
        # The increase is around 2.4x, not 3.5x
        self.assertGreater(flops3 / flops1, 2.0)  # Changed from 3.5 to 2.0
        self.assertLess(flops3 / flops1, 3.0)  # Changed from 4.5 to 3.0


if __name__ == '__main__':
    unittest.main()
