# Adopted from FlagOpen/RoboBrain-X0 (https://github.com/FlagOpen/RoboBrain-X0/blob/main/agilex/server_agilex.py)

import base64
import io
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from flagscale import serve
from flagscale.inference.utils import parse_torch_dtype
from flagscale.models.robobrain_x.qwen_groot import Qwen_GR00T
from flagscale.runner.utils import logger

app = Flask(__name__)
CORS(app)

serve.load_args()

TASK_CONFIG = serve.task_config
ENGINE_CONFIG = TASK_CONFIG.serve[0].engine_args
MODEL_PATH = ENGINE_CONFIG["model"]
SERVICE_CONFIG = {
    "host": ENGINE_CONFIG["host"],
    "port": ENGINE_CONFIG["port"],
    "debug": ENGINE_CONFIG["debug"],
    "threaded": ENGINE_CONFIG["threaded"],
    "max_content_length": 16 * 1024 * 1024,
}


MODEL: Qwen_GR00T = None


class RobobrainX05Server:
    def __init__(self, config):
        self.config = config
        dtype_config = self.config.get("torch_dtype", "torch.float32")
        self.dtype = parse_torch_dtype(dtype_config) if dtype_config else torch.float32
        self.device = torch.device(self.config.get("device", "cuda"))
        self.host = self.config.get("host", "0.0.0.0")
        self.port = self.config.get("port", 5000)

        self.load_model()
        self.warmup()

    def warmup(self):
        for i in range(3):
            logger.info(f"Warming up RobobrainX05Server, step {i + 1}/3")
            self.infer(self.build_input())

    def load_model(self):
        t_s = time.time()
        self.model = Qwen_GR00T.from_pretrained(self.config["model"], custom_config=self.config)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded, latency: {time.time() - t_s:.2f}s")

    def build_input(self):
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        sample = {
            "action": np.random.uniform(-1, 1, size=(16, 7)).astype(
                np.float16
            ),  # action_chunk, action_dim
            "image": [image, image],  # two views
            "lang": "This is a fake for testing.",
            "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
        }
        batch = [sample, sample]  # batch size 2
        return batch

    def infer(self, batch):
        print(f"{batch[0]['image']=}")
        t_s = time.time()
        with torch.no_grad():
            predict_output = self.model.predict_action(
                batch_images=[batch[0]["image"]],
                instructions=[batch[0]["lang"]],
                state=[batch[0]["state"]],
            )
            normalized_actions = predict_output["normalized_actions"]
            logger.info(f"Unnormalized Action: {normalized_actions.shape}")
            logger.info(f"{normalized_actions[0,0,:]=}")

        logger.info(f"Infer latency: {time.time() - t_s:.2f}s")
        return normalized_actions

    def serve(self):
        logger.info(f"Serve URL: http://{self.host}:{self.port}")
        logger.info("Available API:")
        logger.info("  - POST /infer   - inference api")
        app.run(host=self.host, port=self.port, debug=False, threaded=True)


def decode_image_base64(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # image = np.array(image).astype(np.float32) / 255.0
        # shape to: [C, H, W]
        # image = torch.from_numpy(image).permute(2, 0, 1)
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
    if SERVER is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request format error"}), 400
    if "qpos" not in data:
        return jsonify({"success": False, "error": "Request requires: qpos"}), 400
    if "eef_pose" not in data:
        return jsonify({"success": False, "error": "Request requires: eef_pose"}), 400
    return infer(data)


def infer(data):
    try:
        qpos = data["qpos"]
        eef_pose = data["eef_pose"]
        instruction = data.get("instruction")
        images = data.get("images")
    except Exception as e:
        return (
            jsonify({"success": False, "error": f"State parameters processing error: {e}"}),
            400,
        )
    if instruction is None:
        return jsonify({"success": False, "error": "Request requires instruction"}), 400
    if images is not None:
        try:
            pil_images = process_images(images)
        except Exception as e:
            return jsonify({"success": False, "error": f"image process failed: {e}"}), 400
        batch = [
            {
                # images:[pil.image, ...] state&action:[1,7] task:str
                "image": [pil_images[0]["cam_high"], pil_images[0]["cam_left_wrist"]],
                "state": eef_pose,
                "action": qpos,
                "lang": instruction,
            }
        ]
    with torch.no_grad():
        actions = SERVER.infer(batch)
    return jsonify({"success": True, "qpos": actions.tolist()})


if __name__ == "__main__":
    global SERVER
    SERVER = RobobrainX05Server(ENGINE_CONFIG)
    SERVER.serve()
