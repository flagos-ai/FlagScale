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

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any

from flagscale.models.configs.types import NormalizationMode
from flagscale.models.utils.constants import ACTION
from flagscale.models.vla.action_model.gr00t_action_header import GR00TActionHeadConfig
from flagscale.models.vla.pretrained_config import PreTrainedConfig
from flagscale.models.vla.vlm.qwenvl_backbone import QwenVLConfig

if TYPE_CHECKING:
    from flagscale.train.train_config import TrainConfig

logger = getLogger(__name__)


@dataclass
class QwenGr00tConfig(PreTrainedConfig):
    vlm: QwenVLConfig = field(default_factory=QwenVLConfig)
    action_model: GR00TActionHeadConfig = field(default_factory=GR00TActionHeadConfig)

    prompt_template: str | None = None

    # Chunked cross-entropy for VLM co-training loss.
    # 0 = disabled (use HF model's built-in CE), >0 = chunk size in tokens.
    chunked_ce_tokens: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_model.future_action_window_size + 1))

    def validate_features(self) -> None:
        if not self.output_features:
            raise ValueError("output_features must be set")
        action_ft = self.action_feature
        if action_ft is None:
            raise ValueError(f"output_features must contain '{ACTION}' with type ACTION")

    @classmethod
    def from_train_config(cls, train_config: TrainConfig) -> QwenGr00tConfig:
        model_cfg = train_config.model

        vlm_section = model_cfg.vlm
        vlm = QwenVLConfig(
            type=vlm_section.get("type", "qwen3-vl"),
            base_vlm=vlm_section.get("base_vlm", ""),
            load_pretrained=vlm_section.get("load_pretrained", True),
            attn_implementation=vlm_section.get("attn_implementation", None),
        )

        action_model = GR00TActionHeadConfig.from_omegaconf(model_cfg.action_model)

        prompt_template = getattr(model_cfg, "prompt_template", None)
        chunked_ce_tokens = getattr(model_cfg, "chunked_ce_tokens", 0)

        kwargs = dict(
            vlm=vlm,
            action_model=action_model,
            prompt_template=prompt_template,
            chunked_ce_tokens=chunked_ce_tokens,
        )

        raw_norm = getattr(model_cfg, "normalization_mapping", None)
        if raw_norm is not None:
            kwargs["normalization_mapping"] = {k: NormalizationMode(v) for k, v in raw_norm.items()}

        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> QwenGr00tConfig:
        if "vlm" in data and isinstance(data["vlm"], dict):
            data["vlm"] = QwenVLConfig(**data["vlm"])
        if "action_model" in data and isinstance(data["action_model"], dict):
            data["action_model"] = GR00TActionHeadConfig(**data["action_model"])
        if "normalization_mapping" in data and isinstance(data["normalization_mapping"], dict):
            data["normalization_mapping"] = {
                k: NormalizationMode(v) for k, v in data["normalization_mapping"].items()
            }
        return cls(**data)
