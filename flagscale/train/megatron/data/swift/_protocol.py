# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""The one ``swift.infer_engine`` type the agent templates need.

``swift.infer_engine`` is the inference stack (vLLM/LMDeploy/pt engines, OpenAI
protocol dataclasses). The agent templates import a single dataclass from it, to
return parsed tool calls from ``get_toolcall``. Training never calls that path --
it only matters for inference -- but it is part of the class contract, so the type
is reproduced here rather than dragging in the package.

Vendored from ms-swift (Apache-2.0): ``swift/infer_engine/protocol.py``.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ._utils import remove_response


@dataclass
class InferRequest:
    """An inference request, as ``Template.encode`` may be handed one.

    ``encode`` opens with ``isinstance(inputs, InferRequest)`` and converts via
    ``asdict``; the SFT path always passes a plain dict, so this type exists to
    keep that check meaningful rather than to be constructed here. The multimodal
    fields are kept so ``asdict`` yields the same keys as upstream.

    Vendored from ms-swift (Apache-2.0): ``swift/infer_engine/protocol.py``.
    """

    messages: List[Dict[str, Any]]

    images: List[Any] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    tools: Optional[List[Dict[str, Any]]] = None
    objects: Dict[str, Any] = field(default_factory=dict)
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key in ['images', 'audios', 'videos']:
            val = getattr(self, key)
            if isinstance(val, str):
                setattr(self, key, [val])
        assert isinstance(self.messages, list), f'messages: {self.messages}'

    @staticmethod
    def remove_response(messages) -> Optional[str]:
        return remove_response(messages)


@dataclass
class Function:
    name: str
    arguments: Optional[Union[str, Any]]

    def __post_init__(self):
        if not isinstance(self.arguments, str):
            self.arguments = json.dumps(self.arguments, ensure_ascii=False)
        self.name = self.name.strip()
        self.arguments = self.arguments.strip()
