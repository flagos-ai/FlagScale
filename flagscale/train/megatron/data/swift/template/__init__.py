# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Prompt templates, vendored from ms-swift (Apache-2.0).

The modules here are copied unchanged from ``swift/template/`` apart from their
imports; see each file's header. Two deliberate differences from upstream:

- Only the DeepSeek V2.5 -> V3.1 -> V4 -> V4-Flash chain is registered
  (:mod:`.deepseek`). Upstream registers ~200 model families, which is where most
  of ``swift/template/``'s bulk lives.
- :mod:`.vision_utils` is a set of fail-loud stubs: this path is text-only.

``tests/unit_tests/train/megatron/test_swift_sft_align.py`` encodes real samples
through both this package and ``swift.template`` and requires identical
``input_ids`` / ``labels`` / ``loss_scale``.
"""

from . import deepseek  # noqa: F401  -- registers the DeepSeek template chain
from . import qwen  # noqa: F401  -- registers the Qwen3 chain (verification only)
from .base import MaxLengthError, Template
from .constant import LLMTemplateType, MLLMTemplateType, TemplateType
from .register import TEMPLATE_MAPPING, get_template, get_template_meta, register_template
from .template_inputs import StdTemplateInputs, TemplateInputs
from .template_meta import TemplateMeta
from .utils import (
    Context,
    ContextType,
    History,
    Messages,
    Prompt,
    Tool,
    Word,
    findall,
    get_last_user_round,
    history_to_messages,
    messages_to_history,
    split_str_parts_by,
)

__all__ = [
    "MaxLengthError",
    "Template",
    "LLMTemplateType",
    "MLLMTemplateType",
    "TemplateType",
    "TEMPLATE_MAPPING",
    "get_template",
    "get_template_meta",
    "register_template",
    "StdTemplateInputs",
    "TemplateInputs",
    "TemplateMeta",
    "Context",
    "ContextType",
    "History",
    "Messages",
    "Prompt",
    "Tool",
    "Word",
    "findall",
    "get_last_user_round",
    "history_to_messages",
    "messages_to_history",
    "split_str_parts_by",
]
