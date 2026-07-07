# Copied from https://github.com/huggingface/lerobot/blob/0db5f66d/src/lerobot/policies/groot/modeling_groot.py
# Below is the original copyright:

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootPolicy.finetune_with_groot_runner` below.
"""

import dataclasses
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor

from flagscale.models.utils.constants import (
    SAFETENSORS_FILE,
    resolve_pretrained_dir,
)
from flagscale.models.vla.base_policy import TrainablePolicy
from flagscale.models.vla.gr00t_n1_5.configuration_gr00t_n1_5 import Gr00tN15Config
from flagscale.models.vla.gr00t_n1_5.gr00t_n1 import GR00TN15, GR00TN15Config

GR00T_CONFIG_DIR = "gr00t_config"


class Gr00tN15(TrainablePolicy):
    """Wrapper around GR00T N1.5 model for FlagScale training integration."""

    def __init__(self, config: Gr00tN15Config):
        super().__init__(config)
        config.validate_features()

        self._handle_flash_attention_compatibility()

        if config.load_pretrained:
            self._groot_model = GR00TN15.from_pretrained(
                pretrained_model_name_or_path=config.base_model_path,
                tune_llm=config.tune_llm,
                tune_visual=config.tune_visual,
                tune_projector=config.tune_projector,
                tune_diffusion_model=config.tune_diffusion_model,
            )
        else:
            gr00t_config = GR00TN15Config.from_pretrained(config.base_model_path)
            self._groot_model = GR00TN15(
                gr00t_config,
                local_model_path=config.base_model_path,
            )
            self._groot_model.backbone.set_trainable_parameters(
                tune_visual=config.tune_visual,
                tune_llm=config.tune_llm,
            )
            self._groot_model.action_head.set_trainable_parameters(
                tune_projector=config.tune_projector,
                tune_diffusion_model=config.tune_diffusion_model,
            )

        self._groot_model.compute_dtype = config.compute_dtype
        self._groot_model.config.compute_dtype = config.compute_dtype

        if config.input_features:
            self.input_features = config.input_features
        if config.output_features:
            self.output_features = config.output_features

    def _save_pretrained(self, save_directory: Path, state_dict=None) -> None:
        """Save a self-contained GR00T N1.5 checkpoint."""
        save_directory = Path(save_directory)

        gr00t_config_dir = save_directory / GR00T_CONFIG_DIR
        gr00t_config_dir.mkdir(parents=True, exist_ok=True)
        self._groot_model.config.save_pretrained(gr00t_config_dir)

        save_config = dataclasses.replace(
            self.config,
            base_model_path=GR00T_CONFIG_DIR,
            load_pretrained=False,
        )
        save_config._save_pretrained(save_directory)

        if state_dict is not None:
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        else:
            state_dict = {k: v.clone().contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, str(save_directory / SAFETENSORS_FILE))

    @classmethod
    def from_pretrained(cls, pretrained_path, device="cpu", *, config=None):
        """Load a fine-tuned GR00T N1.5 checkpoint without reloading the base model."""
        path = resolve_pretrained_dir(Path(pretrained_path), SAFETENSORS_FILE)
        if config is None:
            config = Gr00tN15Config.from_pretrained(path)

        base_model_path = Path(config.base_model_path)
        if not base_model_path.is_absolute():
            base_model_path = path / base_model_path

        config = dataclasses.replace(
            config,
            base_model_path=str(base_model_path),
            load_pretrained=False,
        )

        return super().from_pretrained(pretrained_path, device=device, config=config)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_"))
            and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        use_bf16 = self.config.compute_dtype == "bfloat16"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict actions for inference by delegating to Isaac-GR00T.

        Returns a dict with 'action' key containing tensor of shape (B, T, action_dim).
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_"))
            and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        use_bf16 = self.config.compute_dtype == "bfloat16"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")
        return {"action": actions}

    def fsdp_units(self):
        """Return FSDP-shardable units from both backbone and action head."""
        units = []
        units.extend(self._groot_model.backbone.fsdp_units())
        units.extend(self._groot_model.action_head.fsdp_units())
        return units

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
