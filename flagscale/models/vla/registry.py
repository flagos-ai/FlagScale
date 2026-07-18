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

VLM_REGISTRY: dict[str, type] = {}
ACTION_MODEL_REGISTRY: dict[str, type] = {}


def register_vlm(name: str):
    def decorator(cls):
        VLM_REGISTRY[name] = cls
        return cls

    return decorator


def register_action_model(name: str):
    def decorator(cls):
        ACTION_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_vlm(name: str, **kwargs):
    if name not in VLM_REGISTRY:
        raise ValueError(f"Unknown VLM: {name}. Available: {list(VLM_REGISTRY.keys())}")
    return VLM_REGISTRY[name](**kwargs)


def build_action_model(name: str, **kwargs):
    if name not in ACTION_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown ActionModel: {name}. Available: {list(ACTION_MODEL_REGISTRY.keys())}"
        )
    return ACTION_MODEL_REGISTRY[name](**kwargs)
