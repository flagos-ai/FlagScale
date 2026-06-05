from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3EngramConfig(PretrainedConfig):
    r"""
    Configuration class for Qwen3 with Engram modules.

    This configuration adds Engram-specific parameters to the base Qwen3 architecture.
    Engram modules enhance the model with n-gram based contextual embeddings using
    hash mappings and gating mechanisms.
    """

    model_type = "qwen3engram"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        max_ngram_size=3,  # The maximum n-gram size in engram module
        hidden_size=2048,
        hc_mult=1,  # The residual channel number in hyper-connection
        emb_size_enngram=512,  # The embedding size per n-gram in engram module
        intermediate_size=11008,
        num_hidden_layers=40,
        engram_layer_ids=[1, 13],  # The layer ids to insert engram modules
        num_attention_heads=16,
        num_key_value_heads=2,
        num_engram_heads=8,  # The number of heads in engram module
        kernel_size=4,  # The kernel size of shortconv in engram module
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=True,
        use_sliding_window=False,
        sliding_window=32768,
        attention_dropout=0.0,
        pad_id=2,
        seed=0,
        **kwargs,
    ):
        # Overall arguments
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.hc_mult = hc_mult

        # Attention arguments
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.num_key_value_heads = num_key_value_heads
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Layer types for attention mechanism (required by Qwen3Attention)
        # Can be "sliding_attention" or "full_attention" for each layer
        if use_sliding_window:
            self.layer_types = ["sliding_attention"] * num_hidden_layers
        else:
            self.layer_types = ["full_attention"] * num_hidden_layers

        # Validate the correctness of rotary position embeddings parameters
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Engram arguments
        self.pad_id = pad_id
        self.max_ngram_size = max_ngram_size
        self.emb_size_enngram = emb_size_enngram
        self.num_engram_heads = num_engram_heads
        self.engram_layer_ids = engram_layer_ids
        self.tokenizer_name_or_path = kwargs.get(
            "tokenizer_name_or_path", "Qwen/Qwen2.5-7B-Instruct"
        )
        # Vocabulary size for each n-gram level
        self.engram_vocab_size = [vocab_size * 5] * (max_ngram_size - 1)
        self.kernel_size = kernel_size
        self.seed = seed

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3EngramConfig"]
