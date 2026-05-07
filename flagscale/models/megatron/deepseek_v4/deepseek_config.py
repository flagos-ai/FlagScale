## built-in
from dataclasses import dataclass
from typing import List, Optional
import warnings

## megatron-core
from megatron.core.transformer import MLATransformerConfig


@dataclass
class DeepSeekConfig(MLATransformerConfig):
    ####################
    # General Configuration
    ####################
    moe_n_hash_layers: int = 0
    """Number of leading transformer layers that use hash-based MoE routing.
    Layers with layer_number <= moe_n_hash_layers use a pre-computed tid2eid
    lookup table for expert selection instead of learned top-k routing."""

    actual_vocab_size: Optional[int] = None
    """Padded actual vocabulary size. Required when moe_n_hash_layers > 0 for the
    tid2eid lookup buffer in hash-based MoE routing."""

    dense_grouped_gemm: bool = False
    """Use GroupedLinear(num_groups=1) for dense MLP to trigger the
    ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8 fusion on SM100+ with MXFP8 recipe.
    Requires ``use_te_op_fuser=True`` and SwiGLU activation.
    """

    log_moe_overload_factor: bool = False
    """When True, log MoE overload metrics (avg/max vs balanced token count per step; max cum
    overload = peak cumulative actual tokens / peak cumulative balanced count over interleaved
    fwd/bwd) to TensorBoard/W&B and console. Records tokens_per_expert.sum() after dispatch;
    use for debugging."""
    
    ####################
    # Engram Configuration
    ####################
    use_engram: bool = False
    engram_tokenizer_name_or_path: str | None = None
    engram_vocab_size: list[int] | None = None
    max_ngram_size: int = 1
    n_embed_per_ngram: int | None = None
    n_head_per_ngram: int = 1
    engram_layer_ids: list[int] | None = None
    engram_pad_id: int = 0
    engram_seed: int = 0
    engram_kernel_size: int = 1
    engram_embedding_parallel_size: int | None = 1
    engram_embedding_parallel_method: str = "alltoall"
    engram_offload_embedding_optimizer_states: bool = False

    ####################
    # Hyper-Connection Configuration
    ####################
    enable_hyper_connections: bool = False
    """Enable mHC residual connections."""

    num_residual_streams: int = 4
    """Number of residual streams (n in paper)."""

    mhc_sinkhorn_iterations: int = 20
    """Number of Sinkhorn-Knopp iterations for doubly stochastic projection."""

    mhc_init_gating_factor: float = 0.01
    """Initial value of Gating Factor (alpha in paper)."""

    use_fused_mhc: bool = False
    """Use cuTile fused kernels for mHC operations.

    When True, attempts to replace the reference mHC modules (SinkhornKnopp,
    H_aggregate, H_post_bda, ProjRms) with fused cuda.tile (cuTile) autograd
    functions for better performance on supported GPUs.  Requires cuTile to be
    installed; if cuTile is unavailable the flag is silently reset to False and
    a warning is emitted.
    """

    mhc_recompute_layer_num: Optional[int] = None
    """Number of layers per MHC recompute block.
    
    When set, every `mhc_recompute_layer_num` layers form a recompute block. The last layer
    in each recompute block (i.e., layer_number % mhc_recompute_layer_num == 0 or the final
    layer in the transformer block) will:
    - NOT checkpoint its final MLP BDA
    - Register the unified recompute hook on its MLP BDA output
    - A new CheckpointManager is created for subsequent layers
    
    If None, all layers in the transformer block share a single recompute block.

    Must be a positive integer when set."""

    _EXTRA_RECOMPUTE_MODULES = {"mhc"}
    """
    Extra modules that can be recomputed. Because the TransformerConfig only validates choice in post_init, we do not need to override recompute_modules here,
    just need to add the extra allowed module names to the validation logic in post_init.
    """

    def __post_init__(self):
        # Validate recompute_modules except _EXTRA_RECOMPUTE_MODULES, which will be validated after super().__post_init__()
        original = self.recompute_modules

        if original is not None:
            base_allowed = {
                "core_attn", "moe_act", "layernorm", "mla_up_proj",
                "mlp", "moe", "shared_experts"
            }
            self.recompute_modules = [m for m in original if m in base_allowed]

        super().__post_init__()
        self.recompute_modules = original

        if self.recompute_modules is not None:
            all_allowed = base_allowed.union(self._EXTRA_RECOMPUTE_MODULES)
            invalid = set(self.recompute_modules) - all_allowed
            assert not invalid, (
                f"[BusinessConfig] Invalid recompute_modules: {invalid}\n"
                f"All allowed: {sorted(all_allowed)}"
            )
        
        # Validation for use_fused_mhc
        if self.use_fused_mhc:
            if not self.enable_hyper_connections:
                raise ValueError("use_fused_mhc requires enable_hyper_connections=True.")
            try:
                from megatron.core.fusions.fused_mhc_kernels import is_cutile_available

                if not is_cutile_available():
                    warnings.warn(
                        "use_fused_mhc is enabled but cuda.tile (cuTile) is not installed. "
                        "Falling back to reference mHC implementations.",
                        UserWarning,
                    )
                    self.use_fused_mhc = False
            except ImportError:
                warnings.warn(
                    "use_fused_mhc is enabled but fused_mhc_kernels module could not be "
                    "imported. Falling back to reference mHC implementations.",
                    UserWarning,
                )
                self.use_fused_mhc = False

        # Validation for hyper_connections with MTP
        if self.enable_hyper_connections and self.mtp_num_layers is not None:
            raise ValueError(
                "enable_hyper_connections is not compatible with Multi-Token Prediction (MTP). "
                "Please disable MTP (set mtp_num_layers=None) when using hyper connections."
            )
