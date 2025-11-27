"""
FLOPS calculation formulas for different model architectures.
This module provides accurate FLOPS calculations for various neural network
components and architectures commonly used in large language models.
(NOTE: Temporarily still in the testing procedure, just a experiment feature)
"""

from typing import Optional


class FLOPSFormulas:
    """
    Collection of FLOPS calculation formulas for different model components.
    All formulas follow the convention:
    - Forward pass FLOPS only (multiply by 3 for forward + backward)
    - Includes both multiplication and addition operations
    - Based on theoretical FLOPS, not hardware-specific optimizations
    """

    @staticmethod
    def attention_flops(
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: Optional[int] = None,
    ) -> float:
        """
        Calculate FLOPS for standard multi-head attention.
        The attention mechanism involves:
        1. QKV projection: 3 * (batch * seq * hidden * hidden)
        2. Attention scores: batch * heads * seq * seq * (hidden/heads)
        3. Attention softmax: negligible compared to matmul
        4. Value projection: batch * heads * seq * seq * (hidden/heads)
        5. Output projection: batch * seq * hidden * hidden
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            kv_channels: Optional custom dimension for keys/values
        Returns:
            Total FLOPS for attention layer
        """
        head_dim = (kv_channels if kv_channels else hidden_size) // num_attention_heads

        # QKV projections
        qkv_flops = 3 * 2 * batch_size * seq_length * hidden_size * hidden_size

        # Attention scores computation (Q @ K^T)
        attention_score_flops = (
            2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        )

        # Attention output (scores @ V)
        attention_output_flops = (
            2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        )

        # Output projection
        output_proj_flops = 2 * batch_size * seq_length * hidden_size * hidden_size

        total_flops = qkv_flops + attention_score_flops + attention_output_flops + output_proj_flops

        return total_flops

    @staticmethod
    def gqa_attention_flops(
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_attention_heads: int,
        num_query_groups: int,
    ) -> float:
        """
        Calculate FLOPS for Grouped Query Attention (GQA).
        GQA reduces KV computation by sharing key-value pairs across
        multiple query heads.
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Hidden dimension
            num_attention_heads: Number of query heads
            num_query_groups: Number of KV groups
        Returns:
            Total FLOPS for GQA layer
        """
        head_dim = hidden_size // num_attention_heads
        kv_hidden_size = head_dim * num_query_groups

        # Q projection
        q_flops = 2 * batch_size * seq_length * hidden_size * hidden_size

        # KV projections (reduced dimension due to grouping)
        kv_flops = 2 * 2 * batch_size * seq_length * hidden_size * kv_hidden_size

        # Attention scores (Q @ K^T)
        attention_score_flops = (
            2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        )

        # Attention output (scores @ V)
        attention_output_flops = (
            2 * batch_size * num_attention_heads * seq_length * seq_length * head_dim
        )

        # Output projection
        output_proj_flops = 2 * batch_size * seq_length * hidden_size * hidden_size

        total_flops = (
            q_flops + kv_flops + attention_score_flops + attention_output_flops + output_proj_flops
        )

        return total_flops

    @staticmethod
    def ffn_flops(
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        ffn_hidden_size: int,
        use_swiglu: bool = False,
    ) -> float:
        """
        Calculate FLOPS for Feed-Forward Network.
        Standard FFN: hidden -> ffn_hidden -> hidden
        SwiGLU FFN: Uses gated linear units with Swish activation
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Model hidden dimension
            ffn_hidden_size: FFN intermediate dimension
            use_swiglu: Whether using SwiGLU activation
        Returns:
            Total FLOPS for FFN layer
        """
        if use_swiglu:
            # SwiGLU has three linear projections: gate, up, and down
            # gate and up project to ffn_hidden_size, down projects back
            gate_proj_flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden_size
            up_proj_flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden_size
            # Element-wise multiplication of gate and up projections
            swish_gate_flops = batch_size * seq_length * ffn_hidden_size
            # Down projection
            down_proj_flops = 2 * batch_size * seq_length * ffn_hidden_size * hidden_size

            total_flops = gate_proj_flops + up_proj_flops + swish_gate_flops + down_proj_flops
        else:
            # Standard FFN with two linear layers
            # Up projection: hidden -> ffn_hidden
            up_proj_flops = 2 * batch_size * seq_length * hidden_size * ffn_hidden_size
            # Down projection: ffn_hidden -> hidden
            down_proj_flops = 2 * batch_size * seq_length * ffn_hidden_size * hidden_size

            total_flops = up_proj_flops + down_proj_flops

        return total_flops

    @staticmethod
    def moe_flops(
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        ffn_hidden_size: int,
        num_experts: int,
        top_k: int,
        use_swiglu: bool = False,
    ) -> float:
        """
        Calculate FLOPS for Mixture of Experts FFN.
        MoE involves:
        1. Router computation to select top-k experts
        2. FFN computation for selected experts only
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Model hidden dimension
            ffn_hidden_size: Expert FFN intermediate dimension
            num_experts: Total number of experts
            top_k: Number of experts selected per token
            use_swiglu: Whether experts use SwiGLU activation
        Returns:
            Total FLOPS for MoE layer
        """
        # Router FLOPS (hidden -> num_experts scores)
        router_flops = 2 * batch_size * seq_length * hidden_size * num_experts

        # Expert FFN FLOPS (only top-k experts are activated per token)
        tokens_total = batch_size * seq_length
        active_tokens = tokens_total * top_k

        # Each expert processes its assigned tokens
        if use_swiglu:
            # SwiGLU FFN for each active token
            expert_flops = (
                3 * 2 * active_tokens * hidden_size * ffn_hidden_size  # gate, up, down projections
                + active_tokens * ffn_hidden_size  # element-wise multiplication
            ) / num_experts  # Assuming uniform distribution
        else:
            # Standard FFN for each active token
            expert_flops = (2 * 2 * active_tokens * hidden_size * ffn_hidden_size) / num_experts

        total_flops = router_flops + expert_flops * num_experts

        return total_flops

    @staticmethod
    def embedding_flops(
        batch_size: int, seq_length: int, vocab_size: int, hidden_size: int
    ) -> float:
        """
        Calculate FLOPS for embedding layers.
        Includes both input embedding and output projection.
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            vocab_size: Vocabulary size
            hidden_size: Embedding dimension
        Returns:
            Total FLOPS for embedding operations
        """
        # Input embedding (lookup is negligible, mainly for gradient computation)
        input_embedding_flops = 2 * batch_size * seq_length * hidden_size

        # Output projection (hidden -> vocab)
        output_projection_flops = 2 * batch_size * seq_length * hidden_size * vocab_size

        return input_embedding_flops + output_projection_flops

    @staticmethod
    def layernorm_flops(batch_size: int, seq_length: int, hidden_size: int) -> float:
        """
        Calculate FLOPS for LayerNorm.
        LayerNorm involves:
        1. Computing mean and variance
        2. Normalization
        3. Scale and shift
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Hidden dimension
        Returns:
            Total FLOPS for LayerNorm
        """
        elements = batch_size * seq_length * hidden_size

        # Mean computation: hidden_size - 1 additions per element
        mean_flops = elements

        # Variance computation: 2 * hidden_size operations per element
        variance_flops = 2 * elements

        # Normalization: 2 operations per element (subtract mean, divide by std)
        norm_flops = 2 * elements

        # Scale and shift: 2 operations per element
        affine_flops = 2 * elements

        return mean_flops + variance_flops + norm_flops + affine_flops

    @staticmethod
    def cross_entropy_flops(batch_size: int, seq_length: int, vocab_size: int) -> float:
        """
        Calculate FLOPS for cross-entropy loss computation.
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            vocab_size: Vocabulary size
        Returns:
            Total FLOPS for loss computation
        """
        # Softmax computation
        # exp: vocab_size operations per token
        # sum: vocab_size - 1 additions per token
        # div: vocab_size operations per token
        softmax_flops = (3 * vocab_size - 1) * batch_size * seq_length

        # Log and multiplication for cross-entropy
        loss_flops = 2 * batch_size * seq_length

        return softmax_flops + loss_flops

    @staticmethod
    def rotary_embedding_flops(
        batch_size: int, seq_length: int, num_heads: int, head_dim: int
    ) -> float:
        """
        Calculate FLOPS for Rotary Position Embeddings (RoPE).
        RoPE applies rotation matrices to query and key vectors.
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
        Returns:
            Total FLOPS for RoPE
        """
        # Rotation involves complex multiplication
        # For each position: 4 multiplications + 2 additions per dimension pair
        # Applied to both Q and K
        elements = batch_size * seq_length * num_heads * head_dim
        rope_flops = 6 * elements  # 4 muls + 2 adds per element

        return rope_flops

    @staticmethod
    def flash_attention_flops(
        batch_size: int, seq_length: int, hidden_size: int, num_attention_heads: int
    ) -> float:
        """
        Calculate FLOPS for Flash Attention.
        Flash Attention is algorithmically equivalent to standard attention
        in terms of FLOPS, but with better memory efficiency.
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
        Returns:
            Total FLOPS (same as standard attention)
        """
        return FLOPSFormulas.attention_flops(
            batch_size, seq_length, hidden_size, num_attention_heads
        )

    @staticmethod
    def gradient_accumulation_factor(micro_batch_size: int, global_batch_size: int) -> float:
        """
        Calculate the gradient accumulation factor.
        This doesn't add FLOPS but affects memory usage.
        Args:
            micro_batch_size: Micro-batch size per GPU
            global_batch_size: Global batch size
        Returns:
            Number of accumulation steps
        """
        return global_batch_size / micro_batch_size
