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
from flagscale.models.pi0.modeling_pi0 import PI0Policy, PI0PolicyConfig
from flagscale.runner.utils import logger

app = Flask(__name__)
CORS(app)


class PI0Server:
    def __init__(self, config):
        self.config_generate = config["generate"]
        self.config_engine = config["engine"]

        dtype_config = self.config_engine.get("torch_dtype", "torch.float32")
        self.dtype = parse_torch_dtype(dtype_config) if dtype_config else torch.float32
        self.host = self.config_engine.get("host", "0.0.0.0")
        self.port = self.config_engine.get("port", 5000)

        self.load_model()
        self.warmup()

    def warmup(self):
        self.infer(self.build_input())

    def load_model(self):
        t_s = time.time()
        config = PI0PolicyConfig.from_pretrained(self.config_engine.model)
        policy = PI0Policy.from_pretrained(
            model_path=self.config_engine.model,
            tokenizer_path=self.config_engine.tokenizer,
            stat_path=self.config_engine.stat_path,
            config=config,
        )
        self.policy = policy.to(device=self.config_engine.device)
        self.policy.eval()
        logger.info(f"PI0 loaded latency: {time.time() - t_s:.2f}s")

    def build_input(self):
        batch = {}
        batch_size = self.config_generate["batch_size"]
        for k in self.config_generate["images_keys"]:
            batch[k] = torch.randn(
                batch_size, *self.config_generate["images_shape"], dtype=self.dtype
            ).cuda()
        batch[self.config_generate["state_key"]] = torch.randn(
            batch_size, self.config_generate["action_dim"], dtype=self.dtype
        ).cuda()
        batch.update(self.config_generate["instruction"])
        return batch

    def infer(self, batch):
        t_s = time.time()
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens, lang_masks = self.policy.prepare_language(batch)
        images = [i.to(dtype=self.dtype) for i in images]
        state = state.to(dtype=self.dtype)
        with torch.no_grad():
            actions = self.policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=None
            )
            logger.info(f"actions: {actions.shape}")
        actions_trunked = actions[
            :, : self.config_generate["action_horizon"], : self.config_generate["action_dim"]
        ]
        logger.info(f"PI0 infer latency: {time.time() - t_s:.2f}s")
        logger.info(f"actions_trunked: {actions_trunked}")
        return actions_trunked

    def serve(self):
        logger.info(f"Serve URL: http://{self.host}:{self.port}")
        logger.info(f"Available API:")
        logger.info(f"  - POST /infer   - inference api")
        app.run(host=self.host, port=self.port, debug=False, threaded=True)


PI0_SERVER: PI0Server = None


def decode_image_base64(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
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


@app.route('/infer', methods=['POST'])
def infer_api():
    if PI0_SERVER is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request format error"}), 400
    if 'qpos' not in data:
        return jsonify({"success": False, "error": "Request requires: qpos"}), 400
    if 'eef_pose' not in data:
        return jsonify({"success": False, "error": "Request requires: eef_pose"}), 400
    try:
        qpos = torch.tensor(data['qpos']).cuda()
        eef_pose = torch.tensor(data['eef_pose']).cuda()
        instruction = data.get('instruction')
        images = data.get('images')
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

    with torch.no_grad():
        batch = {
            # images:[1,3,480,640] state&action:[1,14] task:[str]
            "observation.images.camera0": sample["cam_high"][None,],
            "observation.images.camera1": sample["cam_left_wrist"][None,],
            "observation.images.camera2": sample["cam_right_wrist"][None,],
            "observation.state": eef_pose,
            "action": qpos,
            "task": [instruction],
        }
        actions = PI0_SERVER.infer(batch)
    return jsonify({"success": True, "qpos": actions.tolist()})


def parse_config() -> Union[DictConfig, ListConfig]:
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
    main(parsed_cfg["serve"])
