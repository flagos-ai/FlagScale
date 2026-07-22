# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_policy import TrainablePolicy
from .gr00t_n1_5 import Gr00tN15
from .protocols import ActionModel, VLMBackbone
from .registry import (
    build_action_model,
    build_vlm,
    register_action_model,
    register_vlm,
)
from .utils import get_vlm_config

# TODO: (yupu) QwenGr00t and VLM backbones require a newer transformers (Qwen3VLForConditionalGeneration)
# that is not available in the PI0/PI0.5 conda env. Consolidate into a single env and remove this.
try:
    from .qwen_gr00t import QwenGr00t
    from .vlm import Qwen3VLBackbone, Qwen25VLBackbone, QwenVLBackbone
except ImportError:
    pass

try:
    from flagscale.models.pi0.modeling_pi0 import PI0Policy
    from flagscale.models.pi05.modeling_pi05 import PI05Policy
except ImportError:
    pass

__all__ = [
    "TrainablePolicy",
    "VLMBackbone",
    "ActionModel",
    "register_vlm",
    "register_action_model",
    "build_vlm",
    "build_action_model",
    "get_vlm_config",
    "Gr00tN15",
    "QwenGr00t",
    "PI0Policy",
    "PI05Policy",
    "QwenVLBackbone",
    "Qwen25VLBackbone",
    "Qwen3VLBackbone",
]
