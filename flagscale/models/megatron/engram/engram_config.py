## built-in
from dataclasses import dataclass

## megatron-core
from megatron.core.transformer import MLATransformerConfig


@dataclass
class EngramConfig(MLATransformerConfig):
    engram_tokenizer_name_or_path: str | None = None
    engram_vocab_size: list[int] | None = None
    max_ngram_size: int = 1
    n_embed_per_ngram: int | None = None
    n_head_per_ngram: int = 1
    engram_layer_ids: list[int] | None = None
    engram_pad_id: int = 0
    engram_seed: int = 0
    engram_kernel_size: int = 1
    engram_hc_mult: int = 1
