import argparse
import base64
import io
import json
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from flagscale.models.configs.types import FeatureType, NormalizationMode, PolicyFeature
from flagscale.models.pi0.configuration_pi0 import PI0Config
from flagscale.models.pi0.modeling_pi0 import PI0Policy
from flagscale.models.pi05.configuration_pi05 import PI05Config
from flagscale.models.pi05.modeling_pi05 import PI05Policy
from flagscale.models.utils.constants import ACTION, OBS_STATE
from flagscale.runner.utils import logger
from flagscale.train.train_pi import make_pre_post_processors

app = Flask(__name__)
CORS(app)


class PI0Server:
    def __init__(self, config):
        self.config_engine = config["engine_args"]

        self.host = self.config_engine.get("host", "0.0.0.0")
        self.port = self.config_engine.get("port", 5000)

        self.load_model()
        self.warmup()

    def warmup(self):
        # Build a dummy batch for warmup
        batch = {}
        batch_size = 1
        # Use image keys from config if available, otherwise use default keys
        image_keys = self.config_engine.get(
            "images_keys",
            [
                "observation.images.camera0",
                "observation.images.camera1",
                "observation.images.camera2",
            ],
        )
        images_shape = self.config_engine.get("images_shape", [3, 480, 640])
        for k in image_keys:
            batch[k] = torch.randn(batch_size, *images_shape, dtype=torch.float32).to(
                self.config_engine.device
            )
        state_key = self.config_engine.get("state_key", OBS_STATE)
        action_dim = self.config_engine.get("action_dim", 14)
        batch[state_key] = torch.randn(batch_size, action_dim, dtype=torch.float32).to(
            self.config_engine.device
        )
        batch["task"] = ["warmup task"]

        # Preprocess and run inference
        batch = self.preprocessor(batch)
        with torch.no_grad():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        logger.info("Warmup completed")

    def load_model(self):
        t_s = time.time()
        pretrained_path = self.config_engine.model

        # Get model variant from config (defaults to "pi0")
        model_variant = self.config_engine.get("model_variant", "pi0").lower()
        if model_variant not in ["pi0", "pi0.5"]:
            raise ValueError(f"Invalid model_variant: {model_variant}. Must be 'pi0' or 'pi0.5'")

        logger.info(f"Loading {model_variant} model from {pretrained_path}...")

        # Select config and policy classes based on model variant
        if model_variant == "pi0.5":
            policy_config = PI05Config.from_pretrained(pretrained_path)
            policy_cls = PI05Policy
        else:
            policy_config = PI0Config.from_pretrained(pretrained_path)
            policy_cls = PI0Policy

        policy_config.pretrained_path = pretrained_path
        policy_config.device = self.config_engine.device

        self.policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)
        self.policy = self.policy.to(device=self.config_engine.device)
        self.policy.eval()
        logger.info(f"{model_variant} model loaded successfully")

        # Set normalization mapping for pi0.5 (when not using quantiles)
        use_quantiles = self.config_engine.get("use_quantiles", False)
        if not use_quantiles and model_variant == "pi0.5":
            self.policy.config.normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            }
            logger.info("Set normalization_mapping for pi0.5 model")

        # Load stats and set output_features
        logger.info(f"Loading dataset stats from {self.config_engine.stat_path}...")
        with open(self.config_engine.stat_path, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
        dataset_stats = {}
        for key, sub_dict in stats_dict.items():
            dataset_stats[key] = {
                k: torch.tensor(v).to(self.config_engine.device) for k, v in sub_dict.items()
            }

        # Set output_features from stats to get the actual action dimension
        if ACTION in dataset_stats:
            actual_action_dim = dataset_stats[ACTION]["mean"].shape[-1]
            policy_config.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(actual_action_dim,)
            )
            logger.info(f"Set output_features[ACTION] to actual dimension: {actual_action_dim}")

        # Get rename_map from config
        rename_map = self.config_engine.get("rename_map")

        # Create preprocessor and postprocessor
        processor_kwargs = {}
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": self.config_engine.device},
            "normalizer_processor": {
                "stats": dataset_stats,
                "features": {**policy_config.input_features},
                "norm_map": self.policy.config.normalization_mapping,
            },
            "tokenizer_processor": {"tokenizer_name": self.config_engine.tokenizer},
        }

        if rename_map:
            processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
                "rename_map": rename_map
            }

        postprocessor_kwargs = {}
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset_stats,
                "features": self.policy.config.output_features,
                "norm_map": self.policy.config.normalization_mapping,
            }
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            pretrained_path=pretrained_path, **processor_kwargs, **postprocessor_kwargs
        )

        logger.info(f"PI0 loaded latency: {time.time() - t_s:.2f}s")

    def infer(self, batch):
        """Run inference on a batch.

        Args:
            batch: Dictionary with images, state, and task (before preprocessing)

        Returns:
            Action tensor after postprocessing
        """
        t_s = time.time()

        # Move batch to device
        batch = {
            k: (
                v.to(self.config_engine.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        # Preprocess batch
        batch = self.preprocessor(batch)

        # Run inference
        with torch.no_grad():
            action = self.policy.predict_action_chunk(batch)

        # Postprocess action
        action = self.postprocessor(action)

        logger.info(f"PI0 infer latency: {time.time() - t_s:.2f}s")
        logger.info(f"action shape: {action.shape}")
        return action

    def serve(self):
        logger.info(f"Serve URL: http://{self.host}:{self.port}")
        logger.info("Available API:")
        logger.info("  - POST /infer   - inference api")
        app.run(host=self.host, port=self.port, debug=False, threaded=True)


PI0_SERVER: PI0Server = None


def decode_image_base64(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        # shape to: [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise ValueError(f"Image decode error: {e}")


def process_images(images_json):
    # images_json: List[Dict[str, base64]]
    processed = []
    for i, sample in enumerate(images_json):
        try:
            sample_dict = {}
            for k, v in sample.items():
                sample_dict[k] = decode_image_base64(v)
            processed.append(sample_dict)
        except Exception as e:
            logger.error(f"Image[{i}] decode error: {e}")
            raise ValueError(f"Image[{i}] decode error: {e}")
    return processed


@app.route("/infer", methods=["POST"])
def infer_api():
    if PI0_SERVER is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request format error"}), 400
    if "state" not in data:
        return jsonify({"success": False, "error": "Request requires: state"}), 400
    try:
        state = torch.tensor(data["state"]).cuda()
        instruction = data.get("instruction")
        images = data.get("images")
    except Exception as e:
        return (
            jsonify({"success": False, "error": f"State parameters processing error: {e}"}),
            400,
        )
    if instruction is None:
        return jsonify({"success": False, "error": "Request requires instruction"}), 400
    images_tensor = None
    if images is not None:
        try:
            images_tensor = process_images(images)
            if torch.cuda.is_available():
                for sample in images_tensor:
                    for key in sample:
                        sample[key] = sample[key].cuda()
        except Exception as e:
            return jsonify({"success": False, "error": f"image process failed: {e}"}), 400

    if images_tensor is None or len(images_tensor) == 0:
        return jsonify({"success": False, "error": "Request requires images"}), 400

    sample = images_tensor[0]  # Use first sample

    image_keys = PI0_SERVER.config_engine.get("images_keys", [])
    if not image_keys:
        return jsonify({"success": False, "error": "Config missing images_keys"}), 400

    batch = {}
    for config_key in image_keys:
        if config_key in sample:
            img = sample[config_key]
            # Add batch dimension if needed
            if img.dim() == 3:
                img = img.unsqueeze(0)
            batch[config_key] = img
        # If image is missing, the model will handle it by padding (see _preprocess_images)

    # Ensure at least one image is provided (model requirement)
    if len(batch) == 0:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"No images provided. At least one image is required from: {image_keys}",
                }
            ),
            400,
        )

    state_key = PI0_SERVER.config_engine.get("state_key", OBS_STATE)
    if state.dim() == 1:
        state = state.unsqueeze(0)
    batch[state_key] = state

    batch["task"] = [instruction]

    actions = PI0_SERVER.infer(batch)

    return jsonify({"success": True, "actions": actions.cpu().tolist()})


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


def main(config):
    global PI0_SERVER
    PI0_SERVER = PI0Server(config)
    PI0_SERVER.serve()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"][0])
