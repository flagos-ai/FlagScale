import argparse

from typing import Union

import torch

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.inference.inference_engine import InferenceEngine
from flagscale.inference.utils import parse_torch_dtype
from flagscale.runner.utils import logger


def parse_config() -> Union[DictConfig, ListConfig]:
    """Parse the configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    return config


def inference(cfg: DictConfig) -> None:
    """Run the model inference

    Args:
        cfg: The parsed configuration
    """
    generate_cfg = cfg.get("generate", {})
    dtype_config = cfg.get("engine", {}).get("torch_dtype")
    dtype = parse_torch_dtype(dtype_config) if dtype_config else torch.float32
    batch = build_input(generate_cfg, dtype)

    engine = InferenceEngine(cfg.get("engine", {}))

    images, img_masks = engine.model_or_pipeline.prepare_images(batch)
    state = engine.model_or_pipeline.prepare_state(batch)
    lang_tokens, lang_masks = engine.model_or_pipeline.prepare_language(batch)
    images = [i.to(dtype=dtype) for i in images]
    state = state.to(dtype=dtype)

    with torch.no_grad():
        actions = engine.model_or_pipeline.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=None
        )
        logger.info(f"actions: {actions.shape}")
    actions_trunked = actions[:, : generate_cfg.action_horizon, : generate_cfg.action_dim]
    logger.info(f"actions_trunked: {actions_trunked}")


def build_input(generate_cfg, dtype):
    batch = {}
    batch_size = generate_cfg.batch_size
    for k in generate_cfg.images_keys:
        batch[k] = torch.randn(batch_size, *generate_cfg.images_shape, dtype=dtype).cuda()
    batch[generate_cfg.state_key] = torch.randn(
        batch_size, generate_cfg.action_dim, dtype=dtype
    ).cuda()
    batch.update(generate_cfg.instruction)
    return batch


if __name__ == "__main__":
    parsed_cfg = parse_config()
    assert isinstance(parsed_cfg, DictConfig)  # To make pyright happy
    inference(parsed_cfg)
    logger.info("Inference model inference completed")
