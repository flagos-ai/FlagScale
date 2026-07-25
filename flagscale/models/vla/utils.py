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


def get_vlm_config(vlm_config) -> dict:
    """
    Extract common fields from any VLM config, handling structural differences.

    Args:
        vlm_config: HF config object (may have hidden_size directly or via text_config).
    Returns:
        dict with 'hidden_size' and 'num_hidden_layers'.
    """
    return {
        "hidden_size": _get_hidden_size(vlm_config),
        "num_hidden_layers": _get_num_layers(vlm_config),
    }


def _get_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size from config: {type(config)}")


def _get_num_layers(config) -> int:
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config"):
        return config.text_config.num_hidden_layers
    raise ValueError(f"Cannot determine num_hidden_layers from config: {type(config)}")
