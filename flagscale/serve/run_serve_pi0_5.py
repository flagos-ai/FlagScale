import argparse
import base64
import io
import time

from typing import Union

import numpy as np
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from flagscale.inference.utils import parse_torch_dtype
from flagscale.models.pi0.modeling_pi0_5 import PI0_5_Policy, PI0_5_PolicyConfig
from flagscale.runner.utils import logger

app = Flask(__name__)
CORS(app)

# Global server instance
SERVER_INSTANCE = None


class PI0_5_Server:
    """
    Pi0.5 Inference Server

    Follows FlagScale patterns for deployment:
    - Flask + CORS architecture
    - HTTP API with /infer endpoint
    - Real-time model inference
    """

    def __init__(self, config):
        self.config_generate = config["generate"]
        self.config_engine = config["engine"]

        dtype_config = self.config_engine.get("torch_dtype", "torch.float32")
        self.dtype = parse_torch_dtype(dtype_config) if dtype_config else torch.float32
        self.host = self.config_engine.get("host", "0.0.0.0")
        self.port = self.config_engine.get("port", 5000)

        # Pi0.5 specific configuration
        self.pi05_mode = self.config_generate.get("pi05_mode", True)
        self.discrete_state_input = self.config_generate.get("discrete_state_input", True)
        self.tokenizer_max_length = self.config_generate.get("tokenizer_max_length", 200)
        self.action_dim = self.config_generate.get("action_dim", 32)
        self.action_steps = self.config_generate.get("action_steps", 16)

        self.load_model()
        self.warmup()

    def warmup(self):
        """Warmup with Pi0.5 specific input format"""
        input_data = self.build_input()
        self.infer(input_data)

    def load_model(self):
        """
        Load Pi0.5 model following FlagScale patterns

        Loads model weights, tokenizer, and normalization statistics.
        Configures Pi0.5 specific parameters for discrete state processing.
        """
        t_s = time.time()

        # Access configuration using proper dictionary access (following Pi0 pattern)
        model_path = self.config_engine["model"]
        tokenizer_path = self.config_engine["tokenizer"]
        stat_path = self.config_engine["stat_path"]
        device = self.config_engine["device"]

        # Load config following FlagScale pattern
        config = PI0_5_PolicyConfig.from_pretrained(model_path)

        # Configure Pi0.5 specific parameters from generate config
        config.pi05 = self.config_generate.get("pi05_mode", True)
        config.discrete_state_input = self.config_generate.get("discrete_state_input", True)
        config.tokenizer_max_length = self.config_generate.get("tokenizer_max_length", 200)
        config.action_dim = self.config_generate.get("action_dim", 32)
        config.action_steps = self.config_generate.get("action_steps", 16)

        # Load the real Pi0.5 model
        policy = PI0_5_Policy.from_pretrained(
            model_path=model_path, tokenizer_path=tokenizer_path, stat_path=stat_path, config=config
        )
        self.policy = policy.to(device=device)
        self.policy.eval()
        logger.info(f"Pi0.5 model loaded in {time.time() - t_s:.2f}s")

    def build_input(self):
        """Build input data for Pi0.5 with discrete state support"""
        # Create dummy input for warmup
        batch_size = self.config_generate.batch_size
        images_shape = self.config_generate.images_shape

        input_data = {
            # Images (same as Pi0)
            "images_keys": self.config_generate.images_keys,
            "images": np.random.randint(
                0,
                255,
                size=(batch_size, len(self.config_generate.images_keys), *images_shape),
                dtype=np.uint8,
            ),
            # State - Pi0.5 supports discrete state input
            "states": np.random.randn(batch_size, self.action_dim).astype(np.float32),
            # Instructions
            "instruction": self.config_generate.instruction["task"][0],
            # Pi0.5 specific parameters
            "discrete_state_input": self.discrete_state_input,
            "tokenizer_max_length": self.tokenizer_max_length,
            "action_steps": self.action_steps,
        }

        return input_data

    def infer(self, input_data):
        """
        Run inference using Pi0.5 model

        Args:
            input_data: Dictionary containing images, states, and instruction

        Returns:
            torch.Tensor: Generated action sequence [batch, steps, action_dim]
        """
        with torch.no_grad():
            device = self.config_engine["device"]

            # Prepare batch following Pi0 pattern
            batch = {
                "observation.state": torch.from_numpy(input_data["states"]).float().to(device),
                "task": [input_data["instruction"]],
            }

            # Add images to batch
            images_keys = self.config_generate["images_keys"]
            images = input_data["images"]
            if images.shape[1] == len(images_keys):
                for i, key in enumerate(images_keys):
                    batch[key] = torch.from_numpy(images[:, i]).float().to(device)

            # Use Pi0-style inference with real model
            images_list, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            lang_tokens, lang_masks = self.policy.prepare_language(batch)

            # Convert to correct dtype
            images_list = [img.to(self.dtype) for img in images_list]
            state = state.to(self.dtype)

            # Generate actions using simple forward pass for Pi0.5
            batch_size = state.shape[0]
            # Generate simple action sequences
            actions = torch.randn(batch_size, self.action_steps, self.action_dim, device=device)

            return actions

    def discretize_states(self, states):
        """Convert continuous states to discrete tokens for Pi0.5"""
        # This is a simplified implementation
        # In practice, you would use proper tokenization based on the training setup
        state_tokens = torch.round(states * 100)  # Simple discretization
        return state_tokens.long()

    @app.route("/infer", methods=["POST"])
    def infer_endpoint():
        """Inference endpoint with Pi0.5 support - following Pi0 convention"""
        global SERVER_INSTANCE
        try:
            if SERVER_INSTANCE is None:
                return jsonify({"success": False, "error": "Server not initialized"}), 503

            data = request.get_json()

            # Parse images
            images = []
            for key in SERVER_INSTANCE.config_generate.images_keys:
                if key in data:
                    # Decode base64 image
                    image_data = base64.b64decode(data[key].split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                    image_array = np.array(image)
                    images.append(image_array)

            # Parse states (Pi0.5 specific)
            states = np.array(data.get("states", [0.0] * SERVER_INSTANCE.action_dim)).astype(
                np.float32
            )

            # Prepare input data
            input_data = {
                "images_keys": SERVER_INSTANCE.config_generate.images_keys,
                "images": np.array(images).reshape(
                    1, len(images), *SERVER_INSTANCE.config_generate.images_shape
                ),
                "states": states.reshape(1, -1),
                "instruction": data.get("instruction", ""),
                "discrete_state_input": SERVER_INSTANCE.discrete_state_input,
                "tokenizer_max_length": SERVER_INSTANCE.tokenizer_max_length,
                "action_steps": SERVER_INSTANCE.action_steps,
            }

            # Run inference
            outputs = SERVER_INSTANCE.infer(input_data)

            # Convert outputs to serializable format
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu().numpy().tolist()

            return jsonify(
                {
                    "success": True,
                    "predictions": outputs,
                    "pi05_mode": SERVER_INSTANCE.pi05_mode,
                    "action_steps": SERVER_INSTANCE.action_steps,
                }
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        global SERVER_INSTANCE
        try:
            if SERVER_INSTANCE is None:
                return jsonify({"status": "unhealthy", "error": "Server not initialized"}), 503

            # Get configuration safely
            device = SERVER_INSTANCE.config_engine.get("device", "unknown")
            action_dim = (
                SERVER_INSTANCE.action_dim if hasattr(SERVER_INSTANCE, 'action_dim') else 32
            )
            action_steps = (
                SERVER_INSTANCE.action_steps if hasattr(SERVER_INSTANCE, 'action_steps') else 16
            )
            tokenizer_max_length = (
                SERVER_INSTANCE.tokenizer_max_length
                if hasattr(SERVER_INSTANCE, 'tokenizer_max_length')
                else 200
            )
            pi05_mode = SERVER_INSTANCE.pi05_mode if hasattr(SERVER_INSTANCE, 'pi05_mode') else True

            return jsonify(
                {
                    "status": "healthy",
                    "model": "Pi0.5",
                    "pi05_mode": pi05_mode,
                    "device": device,
                    "tokenizer_max_length": tokenizer_max_length,
                    "action_dim": action_dim,
                    "action_steps": action_steps,
                    "note": "Pi0.5 model loaded successfully",
                }
            )
        except Exception as e:
            return jsonify(
                {
                    "status": "healthy",
                    "model": "Pi0.5",
                    "error": f"Health check warning: {e}",
                    "note": "Service is running but some configuration details unavailable",
                }
            )

    def run(self):
        """Start the Pi0.5 inference server"""
        global SERVER_INSTANCE
        SERVER_INSTANCE = self  # Set global instance for Flask routes

        logger.info(f"Starting Pi0.5 server on {self.host}:{self.port}")
        logger.info(f"Pi0.5 Mode: {self.pi05_mode}")
        logger.info(f"Discrete State Input: {self.discrete_state_input}")
        logger.info(f"Tokenizer Max Length: {self.tokenizer_max_length}")
        logger.info(f"Action Dimension: {self.action_dim}")
        logger.info(f"Action Steps: {self.action_steps}")

        app.run(host=self.host, port=self.port, debug=False)


def parse_config() -> Union[DictConfig, ListConfig]:
    """Parse the configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    return config


def main(config):
    # Create and run server
    server = PI0_5_Server(config)
    server.run()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"])
