# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Vendored from ms-swift (Apache-2.0): the ChatML -> Qwen -> Qwen3 meta chain of
# swift/template/templates/{utils,qwen}.py, copied unchanged.
#
# Upstream qwen.py is ~1770 lines registering the whole Qwen family, most of it
# multimodal (Qwen2-VL/Qwen3-VL/Qwen3.5/Ovis/Omni) plus embedding and reranker
# variants. Only the plain-text Qwen3 metas are carried over; the multimodal
# classes would pull in vision_utils, which this text-only path stubs out.
#
# Qwen3 exists here as a *verification* model, not a training target: the
# DeepSeek-V4 attention variants in Megatron-LM-FL reject packed sequences
# (``csa.py``/``dsa.py`` both assert ``packed_seq_params is None``), so a dsv4 run
# cannot exercise the packed TP/CP paths. Qwen3 uses standard attention and can.

from dataclasses import dataclass, field
from typing import List, Optional

from .constant import LLMTemplateType
from .register import TemplateMeta, register_template
from .utils import Prompt, Word

DEFAULT_SYSTEM = 'You are a helpful assistant.'


@dataclass
class ChatmlTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    auto_add_bos: bool = True


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


@dataclass
class Qwen3MixedTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = None
    non_thinking_prefix: str = '<think>\n\n</think>\n\n'


register_template(Qwen3MixedTemplateMeta(LLMTemplateType.qwen3, is_thinking=True))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking, default_system=None, is_thinking=True, thinking_prefix='<think>\n'))

register_template(QwenTemplateMeta(LLMTemplateType.qwen3_nothinking, default_system=None))
