# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Vendored from ms-swift (Apache-2.0): the DeepSeek-V2.5 -> V3.1 -> V4 -> V4-Flash
# chain of swift/template/templates/deepseek.py (lines 625-762 upstream), copied
# unchanged. The rest of that file registers DeepSeek's multimodal and OCR
# templates (DeepseekVL/Janus/OCR/UnlimitedOCR/VL2), which this text-only data
# path does not use; omitting them also drops the vision_utils dependency.
#
# V2.5 and R1 are kept because the V4 metas inherit from DeepseekV2_5TemplateMeta;
# registering them costs nothing and keeps the chain identical to upstream.

from dataclasses import dataclass, field
from typing import Optional

from .._utils import get_env_args, get_logger
from .base import Template
from .constant import LLMTemplateType
from .register import TemplateMeta, register_template
from .utils import Prompt

logger = get_logger()



@dataclass
class DeepseekV2_5TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<｜begin▁of▁sentence｜>{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<｜User｜>{{QUERY}}<｜Assistant｜>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<｜end▁of▁sentence｜>'])
    suffix: Prompt = field(default_factory=lambda: ['<｜end▁of▁sentence｜>'])


register_template(DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_v2_5))

register_template(DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_r1, is_thinking=True, thinking_prefix='<think>\n'))


class DeepseekV3_1Template(Template):
    jinja_enable_thinking_key = 'thinking'
    non_thinking_prefix_only_after_user = True


register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v3_1,
        agent_template='deepseek_v3_1',
        is_thinking=True,
        template_cls=DeepseekV3_1Template,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))

# Reasoning-effort prefixes, prepended at the very beginning of the conversation
# (before the system content) when thinking is enabled. Naming follows the prompt text
# rather than the level, because the level each one maps to differs between releases:
# `ABSOLUTE_MAX` is `max` for V4-Flash/V4-Pro (preview) but `high` for V4-Flash-0731.
REASONING_EFFORT_ABSOLUTE_MAX = (
    'Reasoning Effort: Absolute maximum with no shortcuts permitted.\n'
    'You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve '
    'the root cause, rigorously stress-testing your logic against all potential paths, edge cases, '
    'and adversarial scenarios.\n'
    'Explicitly write out your entire deliberation process, documenting every intermediate step, '
    'considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n')

REASONING_EFFORT_BEYOND_MAX = (
    'Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n'
    'You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: '
    'exhaustively decompose the problem into its most fundamental components, trace every causal chain '
    'to its root, and resolve the underlying cause rather than any surface symptom.\n'
    'Do not stop reasoning until you have independently verified the solution from multiple angles and '
    'are certain that no assumption remains unchecked and no error remains undiscovered.\n\n')


class DeepseekV4Template(DeepseekV3_1Template):
    # V4-Flash / V4-Pro (preview) ship two thinking levels; `high` adds no prefix.
    reasoning_effort_prompts = {'high': '', 'max': REASONING_EFFORT_ABSOLUTE_MAX}
    default_reasoning_effort = 'high'

    def init_env_args(self):
        super().init_env_args()
        self.reasoning_effort = self._check_reasoning_effort(get_env_args('reasoning_effort', str, None))
        if self.reasoning_effort is None:
            self.reasoning_effort = self.default_reasoning_effort if self.enable_thinking else None
        self.enable_thinking = self.reasoning_effort in self.reasoning_effort_prompts
        self.chat_template_kwargs['reasoning_effort'] = self.reasoning_effort

    def _check_reasoning_effort(self, reasoning_effort):
        """Drop an unknown level so that it falls back to the default instead of disabling thinking.

        Every accepted level is a thinking level, so an unrecognized value would otherwise be
        indistinguishable from "thinking off" and silently turn reasoning off.
        """
        if reasoning_effort is not None and reasoning_effort not in self.reasoning_effort_prompts:
            logger.warning(f'Ignoring unknown reasoning_effort: {reasoning_effort!r}. '
                           f'Expected one of {list(self.reasoning_effort_prompts)}.')
            return None
        return reasoning_effort

    def _get_reasoning_effort(self, inputs=None):
        reasoning_effort = None if inputs is None else inputs.chat_template_kwargs.get('reasoning_effort')
        reasoning_effort = self._check_reasoning_effort(reasoning_effort)
        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort
        return reasoning_effort

    def _get_enable_thinking(self, inputs=None):
        reasoning_effort = None if inputs is None else inputs.chat_template_kwargs.get('reasoning_effort')
        reasoning_effort = self._check_reasoning_effort(reasoning_effort)
        if reasoning_effort is not None:
            return reasoning_effort in self.reasoning_effort_prompts
        return super()._get_enable_thinking(inputs)

    def _get_system(self, inputs):
        system = super()._get_system(inputs)
        if self._get_enable_thinking(inputs):
            prefix = self.reasoning_effort_prompts.get(self._get_reasoning_effort(inputs)) or ''
            if prefix:
                system = prefix + (system or '')
        return system

    def _remove_history_thinking(self, inputs) -> None:
        # The official encoding disables `drop_thinking` once tools are defined: tool-calling
        # conversations keep the reasoning of every turn so the model can track multi-step
        # reasoning across tool calls.
        if inputs.tools:
            return
        super()._remove_history_thinking(inputs)


register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v4,
        agent_template='deepseek_v4',
        is_thinking=True,
        template_cls=DeepseekV4Template,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))


class DeepseekV4FlashTemplate(DeepseekV4Template):
    # V4-Flash-0731 ships three thinking levels and shifts the prefixes one level down:
    # what `max` meant in the preview release is `high` here, and `max` gets a stronger text.
    # `low` is the default and adds no prefix (it is still a thinking level).
    reasoning_effort_prompts = {
        'low': '',
        'high': REASONING_EFFORT_ABSOLUTE_MAX,
        'max': REASONING_EFFORT_BEYOND_MAX,
    }
    default_reasoning_effort = 'low'


register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v4_flash,
        agent_template='deepseek_v4',
        is_thinking=True,
        template_cls=DeepseekV4FlashTemplate,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))
