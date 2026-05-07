# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/deployment/model_server/server_policy.py

import argparse
import time

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import flagscale.serve.processor  # noqa: F401 — registers serve-specific processor steps
from flagscale.logger import logger
from flagscale.models.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from flagscale.models.vla import TrainablePolicy
from flagscale.serve.processor.image_layout_processor import ImageLayoutProcessorStep
from flagscale.serve.processor.image_resize_processor import ImageResizeProcessorStep
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer
from flagscale.train.processor import PolicyProcessorPipeline, ProcessorStepRegistry

# TODO: (yupu) to constant.py?
TASK_KEY = "task"


def _default_serve_preprocessor(model_variant: str) -> PolicyProcessorPipeline | None:
    """Return the default serve preprocessor for a given model variant, or None if not needed."""
    variant = model_variant.lower()
    if variant == "pi0.5":
        return _default_pi05_serve_preprocessor()
    if variant == "qwengr00t":
        return _default_qwen_gr00t_serve_preprocessor()
    return None


def _default_pi05_serve_preprocessor() -> PolicyProcessorPipeline:
    """Default serve preprocessor for pi0.5.

    pi0.5's ``_preprocess_images`` expects images as [B, C, H, W] float32 [0, 1].
    Client observations arrive as HWC uint8 numpy arrays, so this pipeline converts them.
    """
    return PolicyProcessorPipeline(
        steps=[
            ImageLayoutProcessorStep(
                src_layout="hwc",
                dst_layout="chw",
                add_batch_dim=True,
                to_float=True,
            )
        ]
    )


def _default_qwen_gr00t_serve_preprocessor() -> PolicyProcessorPipeline:
    """Default serve preprocessor for QwenGr00t.

    Resizes images to the training resolution (224x224). The Qwen VL processor
    handles further preprocessing (normalization, patching) internally.
    """
    return PolicyProcessorPipeline(steps=[ImageResizeProcessorStep(image_size=[224, 224])])


def validate_batch(batch: dict) -> list[str]:
    """Validate a batch against the internal data contract.

    Expected canonical keys (after rename_map has been applied)::

        - ``observation.images.*``: ``np.ndarray``, HWC uint8, ndim == 3
        - ``observation.state``: ``np.ndarray`` or ``list``
        - ``task``: ``str``

    Returns a list of warning/error messages. Empty list means the batch is valid.
    """
    errors: list[str] = []

    if TASK_KEY not in batch:
        errors.append(f"Missing required key '{TASK_KEY}'")
    elif not isinstance(batch[TASK_KEY], str):
        errors.append(f"'{TASK_KEY}' must be str, got {type(batch[TASK_KEY]).__name__}")

    if OBS_STATE not in batch:
        errors.append(f"Missing required key '{OBS_STATE}'")
    elif not isinstance(batch[OBS_STATE], (np.ndarray, list)):
        errors.append(
            f"'{OBS_STATE}' must be np.ndarray or list, got {type(batch[OBS_STATE]).__name__}"
        )

    image_keys = [k for k in batch if k.startswith(OBS_IMAGES)]
    if not image_keys:
        errors.append(f"No image keys found (expected keys starting with '{OBS_IMAGES}')")

    for key in image_keys:
        img = batch[key]
        if not isinstance(img, np.ndarray):
            errors.append(f"'{key}' must be np.ndarray, got {type(img).__name__}")
            continue
        if img.ndim != 3:
            errors.append(f"'{key}' must be HWC (ndim=3), got ndim={img.ndim} shape={img.shape}")
        if img.dtype != np.uint8:
            errors.append(f"'{key}' expected dtype=uint8, got {img.dtype}")

    return errors


def debug_print(batch):
    for k, v in batch.items():
        if hasattr(v, "shape"):
            logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            logger.info(f"  {k}: type={type(v).__name__} value={repr(v)[:120]}")


class Policy:
    """VLA policy server wrapping a TrainablePolicy with pre/post-processing.

    Loads a pretrained policy model and its processor pipelines from a checkpoint.
    Optionally applies serve-time preprocessing (e.g., image resize) and observation
    key renaming before the saved model preprocessor.

    Args:
        config: Serve config containing an ``engine_args`` section.
    """

    def __init__(self, config: DictConfig | ListConfig) -> None:
        self.config_engine = config["engine_args"]

        self.host: str = self.config_engine.get("host", "0.0.0.0")
        self.port: int = self.config_engine.get("port", 5000)
        self.model: TrainablePolicy | None = None
        self.preprocessor: PolicyProcessorPipeline | None = None
        self.postprocessor: PolicyProcessorPipeline | None = None
        self.serve_preprocessor: PolicyProcessorPipeline | None = None
        self.rename_map: dict[str, str] | None = None

        self.load_policy()

    def load_policy(self) -> None:
        """Load the policy model and all processor pipelines from the checkpoint."""
        t_s = time.perf_counter()
        pretrained_dir: str = self.config_engine.model
        self.model = TrainablePolicy.from_pretrained(
            pretrained_dir, device=self.config_engine.device
        )
        self._load_processors(pretrained_dir)
        # TODO: (yupu): model.to(dtype)?
        logger.info(f"Policy model loading latency: {time.perf_counter() - t_s:.2f}s")

    def _load_processors(self, pretrained_dir: str) -> None:
        """Load pre/post-processors from the checkpoint and build the serve preprocessor."""
        self.rename_map = self.config_engine.get("rename_map")

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_preprocessor.json",
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_postprocessor.json",
        )
        self.serve_preprocessor = self._build_serve_preprocessor()

    def _build_serve_preprocessor(self) -> PolicyProcessorPipeline | None:
        """Build a serve-time preprocessor pipeline from the ``serve_preprocessor`` config.

        Returns ``None`` if no serve preprocessor is configured. Steps are resolved
        from the ``ProcessorStepRegistry`` by their ``registry_name``.
        """
        serve_preproc_cfg = self.config_engine.get("serve_preprocessor")
        if not serve_preproc_cfg or not serve_preproc_cfg.get("steps"):
            # TODO: (yupu) No way to explicitly opt out of the default — need a flag like
            # ``serve_preprocessor: {disabled: true}`` for cases where no steps are wanted.
            return _default_serve_preprocessor(self.config_engine.get("model_variant", ""))
        steps = []
        for step_cfg in serve_preproc_cfg.steps:
            step_cls = ProcessorStepRegistry.get(step_cfg.registry_name)
            config = OmegaConf.to_container(step_cfg.get("config", {}), resolve=True)
            steps.append(step_cls(**config))
        return PolicyProcessorPipeline(steps=steps)

    def inference(self, batch: dict) -> dict:
        """Run inference on a single batch.


        Args:
            batch: Raw observation dict from the client. Image values are typically
                ``np.ndarray`` in HWC uint8 format.

        Returns:
            Dict with ``action`` key containing a numpy array of shape
            ``[B, T, action_dim]``.
        """
        logger.info("Start to inference")
        logger.info(f"Raw batch keys: {list(batch.keys())}")

        if self.rename_map:
            batch = {self.rename_map.get(k, k): v for k, v in batch.items()}
            logger.info(f"After rename keys: {list(batch.keys())}")

        errors = validate_batch(batch)
        if errors:
            for err in errors:
                # TODO: (yupu) Response with error status?
                logger.warning(f"Batch validation: {err}")

        debug_print(batch)
        # for k, v in batch.items():
        #     if hasattr(v, "shape"):
        #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")
        #     else:
        #         logger.info(f"  {k}: type={type(v).__name__} value={repr(v)[:120]}")

        if self.serve_preprocessor:
            batch = self.serve_preprocessor(batch)
            logger.info("After serve_preprocessor:")
            debug_print(batch)
            # for k, v in batch.items():
            #     if hasattr(v, "shape"):
            #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")

        batch = self.preprocessor(batch)
        logger.info("After preprocessor:")
        debug_print(batch)
        # for k, v in batch.items():
        #     if hasattr(v, "shape"):
        #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")

        with torch.no_grad():
            action = self.model.predict_action(batch)

        logger.info(f"Raw action keys: {list(action.keys())}")
        debug_print(action)
        # for k, v in action.items():
        #     if hasattr(v, "shape"):
        #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype} first_step={v[0,0,:7]}")

        action = self.postprocessor(action)

        # Convert to numpy for msgpack serialization; squeeze batch dim [1,T,D] → [T,D]
        action[ACTION] = action[ACTION].squeeze(0).detach().cpu().numpy()
        # TODO: (yupu): rename_map for output key
        action["actions"] = action[ACTION]
        logger.info(f"Final action shape: {action[ACTION].shape}, first_step={action[ACTION][0]}")

        return action


def parse_config() -> DictConfig | ListConfig:
    """Parse the configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the log")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    return config


def main(config: DictConfig | ListConfig) -> None:
    """Start the websocket policy server."""
    policy = Policy(config)
    # start websocket server
    server = WebsocketPolicyServer(
        policy=policy,
        host=policy.host,
        port=policy.port,
        metadata={"env": "simpler_env"},
    )
    logger.info(f"Server running at {policy.host}:{policy.port}...")
    server.serve_forever()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    if isinstance(parsed_cfg, ListConfig):
        main(parsed_cfg[0])
    else:
        main(parsed_cfg["serve"][0])
