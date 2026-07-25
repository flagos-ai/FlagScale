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

import argparse

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from flagscale.logger import logger
from flagscale.models.utils.constants import OBS_STATE
from flagscale.models.vla import TrainablePolicy
from flagscale.platforms import get_platform  # noqa: F401 must be before model imports
from flagscale.train.processor import PolicyProcessorPipeline
from flagscale.train.processor.pipeline import get_device_override


def load_image(image_path: str, size: tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    # uint8 HWC, matching the training pipeline
    return torch.from_numpy(np.array(img)).unsqueeze(0)


def load_state_from_file(state_path: str) -> torch.Tensor:
    # (1, state_dim)
    state = torch.load(state_path, map_location="cpu")
    return state


def run_inference(config_path: str):
    logger.info(f"Loading config from {config_path}...")
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    engine_cfg = cfg.engine
    generate_cfg = cfg.generate

    pretrained_dir = engine_cfg.model
    runtime_device = getattr(engine_cfg, "device", None) or "cpu"
    model = TrainablePolicy.from_pretrained(pretrained_dir, device=runtime_device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_dir,
        config_filename="policy_preprocessor.json",
        overrides=get_device_override(runtime_device),
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_dir,
        config_filename="policy_postprocessor.json",
    )

    # TODO: (yupu): model.to(dtype)?

    images = generate_cfg.images
    state_path = generate_cfg.get("state_path")
    task_path = generate_cfg.get("task_path")

    image_keys = list(images.keys())
    logger.info(f"Loading {len(image_keys)} images...")
    loaded_images = {}
    for img_key, img_path in images.items():
        img = load_image(img_path, size=(224, 224))
        loaded_images[img_key] = img
        logger.info(f"Loaded image: {img_key} from {img_path} with shape {img.shape}")

    logger.info(f"Loading state from {state_path}...")
    state = load_state_from_file(state_path)
    logger.info(f"Loaded state with shape: {state.shape}")

    logger.info(f"Loading task from {task_path}...")
    with open(task_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    logger.info(f"Loaded task prompt: '{prompt}'")

    batch = {}
    for img_key, img in loaded_images.items():
        batch[img_key] = img
    batch[OBS_STATE] = state
    batch["task"] = [prompt]

    logger.info("Preprocessing batch...")
    batch = preprocessor(batch)

    logger.info("Running inference...")
    with torch.no_grad():
        action = model.predict_action(batch)
        logger.info(f"action before postprocessor: {action}")

    logger.info("Applying postprocessor...")
    action = postprocessor(action)
    logger.info(f"action after postprocessor: {action}")

    logger.info(f"Final action: {action}")
    logger.info("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to config YAML file")

    args = parser.parse_args()
    run_inference(config_path=args.config_path)


if __name__ == "__main__":
    main()
