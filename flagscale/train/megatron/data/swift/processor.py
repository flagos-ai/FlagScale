# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""A minimal stand-in for ``swift.model.get_processor``.

``swift.model`` exists to resolve any of ~200 model families, load weights, patch
them and attach tuners. The SFT data path needs none of that: it only tokenizes.
Importing it anyway pulls in ~39 modules / ~12k LOC plus ``peft`` (which
``swift.model`` imports at module scope for three ``isinstance(model, PeftModel)``
checks) and, through ``swift.model.patcher``, ``swift.sequence_parallel`` and its
ring-attention implementations.

The template only reads six attributes off the processor::

    model_info.model_dir       (deepseek templates: dynamic-module lookups)
    model_info.config
    model_info.torch_dtype
    model_info.task_type
    model_info.max_model_len   (only when ``max_length`` is left unset)
    model_meta.is_multimodal

so a plain ``AutoTokenizer`` with these two records attached drives ``encode``
identically. ``tests/unit_tests/train/megatron/test_swift_sft_align.py`` pins that
equivalence against ms-swift.

Text-only by design: ``is_multimodal`` is hardcoded ``False``. DeepSeek-V4 carries
no vision/audio sub-config and ``DeepseekV4ForCausalLM`` has no multimodal tower
(see ``test_deepseek_v4_flash_is_text_only`` in ms-swift). A multimodal model would
also need ``AutoProcessor`` and the image/video plumbing this deliberately omits,
so :func:`build_processor` rejects one rather than silently mis-encoding it.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# Mirrors swift.utils.HfConfigFactory.get_max_model_len: different families spell
# the context length differently, and the smallest value present wins.
_MAX_LEN_KEYS = (
    "seq_length",
    "max_position_embeddings",
    "n_positions",
    "model_max_length",
    "seq_len",
    "max_seq_len",
    "max_sequence_length",
    "max_seq_length",
)

# A config declaring any of these is multimodal, which this shim does not support.
_MULTIMODAL_CONFIG_KEYS = ("vision_config", "audio_config", "visual", "image_token_id")


@dataclass
class ShimModelInfo:
    """The ``processor.model_info`` fields the template reads."""

    model_type: str
    model_dir: str
    config: Any
    torch_dtype: Optional[torch.dtype] = None
    max_model_len: Optional[int] = None
    task_type: str = "causal_lm"
    is_multimodal: bool = False
    # swift.model.ModelInfo.__post_init__ derives this; nothing on the data path
    # reads it, but keep the attribute so duck-typing against ModelInfo holds.
    model_name: Optional[str] = None

    def __post_init__(self):
        if self.model_name is None:
            self.model_name = os.path.basename(self.model_dir.rstrip("/"))


@dataclass
class ShimModelMeta:
    """The ``processor.model_meta`` fields the template reads."""

    model_type: str
    template: str
    is_multimodal: bool = False


def _get_max_model_len(config) -> Optional[int]:
    lengths = [
        value
        for value in (getattr(config, key, None) for key in _MAX_LEN_KEYS)
        if isinstance(value, int) and value > 0
    ]
    return min(lengths) if lengths else None


def _get_torch_dtype(config) -> Optional[torch.dtype]:
    # transformers>=5 renamed `torch_dtype` to `dtype`; accept either.
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype, None)
    return None


def build_processor(model_dir: str, model_type: str, template: str):
    """Return a tokenizer carrying the ``model_info``/``model_meta`` the template needs.

    Raises:
        ValueError: if ``model_dir`` looks multimodal; this shim is text-only.
    """
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    declared = [key for key in _MULTIMODAL_CONFIG_KEYS if getattr(config, key, None) is not None]
    if declared:
        raise ValueError(
            f"{model_dir} looks multimodal (config declares {declared}), which "
            "the vendored Swift SFT processor does not support. Use "
            "swift.model.get_processor, "
            "which resolves AutoProcessor and the image/video pipeline."
        )

    processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    processor.model_info = ShimModelInfo(
        model_type=model_type,
        model_dir=model_dir,
        config=config,
        torch_dtype=_get_torch_dtype(config),
        max_model_len=_get_max_model_len(config),
    )
    processor.model_meta = ShimModelMeta(model_type=model_type, template=template)
    return processor
