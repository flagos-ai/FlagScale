# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Agent (tool-calling) templates, vendored from ms-swift (Apache-2.0).

Only the DeepSeek entries are carried over; upstream ``swift/agent_template/``
registers ~20 families (Qwen, Hermes, ReAct, GLM, Kimi, ...). ``agent_template_map``
keeps the same shape so ``Template.agent_template`` resolves by name as upstream
does, and its error message still lists what is available.

``deepseek_v4`` renders tool calls in DSML (``<｜DSML｜tool_calls>``) and tool
results as ``<tool_result>`` blocks merged into one user turn. Note that a dataset
which hand-writes its tool syntax into message content -- rather than using the
``tool_call`` / ``tool_response`` roles and a ``tools`` field -- never reaches this
code; the markup is then just text.
"""

from .base import BaseAgentTemplate
from .deepseek_v3_1 import DeepSeekV31AgentTemplate
from .deepseek_v4 import DeepSeekV4AgentTemplate

# Add an agent template here, select it with --agent_template xxx
agent_template_map = {
    # ref: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
    'deepseek_v3_1': DeepSeekV31AgentTemplate,
    # ref: https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash
    'deepseek_v4': DeepSeekV4AgentTemplate,
}

__all__ = [
    "BaseAgentTemplate",
    "DeepSeekV31AgentTemplate",
    "DeepSeekV4AgentTemplate",
    "agent_template_map",
]
