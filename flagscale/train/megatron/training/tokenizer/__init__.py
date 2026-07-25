# Copyright (c) 2025, FlagScale Team. All rights reserved.

"""FlagScale tokenizer extensions.

This module provides FlagScale-specific tokenizer types that are not available
in upstream Megatron-LM. Standard tokenizer types (GPT2BPE, SentencePiece,
HuggingFace, TikToken, etc.) are handled by the upstream factory at
megatron.core.tokenizers.utils.build_tokenizer.
"""

from .tokenizer import build_tokenizer, register_tokenizer_factory

__all__ = ["build_tokenizer", "register_tokenizer_factory"]
