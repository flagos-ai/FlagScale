import argparse
import importlib

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from flagscale.logger import logger
from flagscale.models.utils.constants import OBS_STATE
from flagscale.train.utils.train_utils import load_checkpoint


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

    model_variant = engine_cfg.model_variant
    policy = getattr(importlib.import_module("flagscale.models.vla"), model_variant)
    model, preprocessor, postprocessor = load_checkpoint(
        engine_cfg.model, policy, engine_cfg.device
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
