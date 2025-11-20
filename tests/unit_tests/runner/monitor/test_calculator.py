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

        expected_flops = (
            qkv_flops + attention_score_flops + attention_output_flops + output_proj_flops
        )

        self.assertEqual(flops, expected_flops)
        self.assertGreater(flops, 0)

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

        # GQA should have fewer FLOPS than standard attention due to KV sharing
        standard_flops = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        self.assertLess(flops, standard_flops)

    def test_ffn_flops_standard(self):
        """Test standard FFN FLOPS calculation."""
        flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )

        # Standard FFN: hidden -> ffn_hidden -> hidden
        up_proj = 2 * self.batch_size * self.seq_length * self.hidden_size * self.ffn_hidden_size
        down_proj = 2 * self.batch_size * self.seq_length * self.ffn_hidden_size * self.hidden_size
        expected_flops = up_proj + down_proj

        self.assertEqual(flops, expected_flops)
        self.assertGreater(flops, 0)

    def test_ffn_flops_swiglu(self):
        """Test SwiGLU FFN FLOPS calculation."""
        flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=True,
        )

        # SwiGLU has gate, up, and down projections
        gate_proj = 2 * self.batch_size * self.seq_length * self.hidden_size * self.ffn_hidden_size
        up_proj = 2 * self.batch_size * self.seq_length * self.hidden_size * self.ffn_hidden_size
        swish_gate = self.batch_size * self.seq_length * self.ffn_hidden_size
        down_proj = 2 * self.batch_size * self.seq_length * self.ffn_hidden_size * self.hidden_size

        expected_flops = gate_proj + up_proj + swish_gate + down_proj

        self.assertEqual(flops, expected_flops)

        # SwiGLU should have more FLOPS than standard FFN
        standard_flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )

        self.assertGreater(flops, standard_flops)

    def test_moe_flops(self):
        """Test Mixture of Experts FLOPS calculation."""
        num_experts = 8
        top_k = 2

        flops = FLOPSFormulas.moe_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            use_swiglu=False,
        )

        self.assertGreater(flops, 0)

        # MoE with top_k < num_experts should have fewer FLOPS than
        # num_experts independent FFNs
        single_ffn_flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )

        # Should be less than all experts active
        self.assertLess(flops, num_experts * single_ffn_flops)

    def test_embedding_flops(self):
        """Test embedding FLOPS calculation."""
        flops = FLOPSFormulas.embedding_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )

        # Input embedding + output projection
        input_emb = 2 * self.batch_size * self.seq_length * self.hidden_size
        output_proj = 2 * self.batch_size * self.seq_length * self.hidden_size * self.vocab_size
        expected_flops = input_emb + output_proj

        self.assertEqual(flops, expected_flops)
        self.assertGreater(flops, 0)

    def test_layernorm_flops(self):
        """Test LayerNorm FLOPS calculation."""
        flops = FLOPSFormulas.layernorm_flops(
            batch_size=self.batch_size, seq_length=self.seq_length, hidden_size=self.hidden_size
        )

        self.assertGreater(flops, 0)

        # LayerNorm should be much smaller than attention/FFN
        attention_flops = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        self.assertLess(flops, attention_flops * 0.01)  # Less than 1% of attention

    def test_cross_entropy_flops(self):
        """Test cross-entropy loss FLOPS calculation."""
        flops = FLOPSFormulas.cross_entropy_flops(
            batch_size=self.batch_size, seq_length=self.seq_length, vocab_size=self.vocab_size
        )

        self.assertGreater(flops, 0)

        # Softmax is the dominant component
        elements = self.batch_size * self.seq_length
        # exp + sum + div for softmax, plus log and mul for loss
        min_ops = elements * self.vocab_size * 3

        self.assertGreater(flops, min_ops)

    def test_rotary_embedding_flops(self):
        """Test Rotary Position Embedding FLOPS calculation."""
        flops = FLOPSFormulas.rotary_embedding_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            num_heads=self.num_heads,
            head_dim=self.hidden_size // self.num_heads,
        )

        self.assertGreater(flops, 0)

        # RoPE applies to Q and K, with rotation operations
        elements = (
            self.batch_size
            * self.seq_length
            * self.num_heads
            * (self.hidden_size // self.num_heads)
        )
        expected_flops = 6 * elements  # 4 muls + 2 adds per element

        self.assertEqual(flops, expected_flops)

    def test_flash_attention_flops(self):
        """Test Flash Attention FLOPS calculation."""
        flash_flops = FLOPSFormulas.flash_attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        standard_flops = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        # Flash attention has same FLOPS as standard attention
        self.assertEqual(flash_flops, standard_flops)

    def test_gradient_accumulation_factor(self):
        """Test gradient accumulation factor calculation."""
        micro_batch_size = 4
        global_batch_size = 32

        factor = FLOPSFormulas.gradient_accumulation_factor(
            micro_batch_size=micro_batch_size, global_batch_size=global_batch_size
        )

        self.assertEqual(factor, 8)  # 32 / 4 = 8

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with batch_size = 1
        flops = FLOPSFormulas.attention_flops(
            batch_size=1,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )
        self.assertGreater(flops, 0)

        # Test with seq_length = 1
        flops = FLOPSFormulas.ffn_flops(
            batch_size=self.batch_size,
            seq_length=1,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            use_swiglu=False,
        )
        self.assertGreater(flops, 0)

        # Test with minimal dimensions
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

        # Quadruple sequence length should ~quadruple attention FLOPS (due to seq^2 term)
        flops3 = FLOPSFormulas.attention_flops(
            batch_size=self.batch_size,
            seq_length=self.seq_length * 2,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
        )

        # Not exactly 4x due to linear projection terms
        self.assertGreater(flops3 / flops1, 3.5)
        self.assertLess(flops3 / flops1, 4.5)


if __name__ == '__main__':
    unittest.main()
