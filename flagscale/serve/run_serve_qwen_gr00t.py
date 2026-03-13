# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/deployment/model_server/server_policy.py

import argparse
import time

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.logger import logger
from flagscale.models.utils.constants import ACTION
from flagscale.models.vla import TrainablePolicy
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer
from flagscale.train.processor import PolicyProcessorPipeline


class Policy:
    def __init__(self, config: DictConfig | ListConfig):
        self.config_engine = config["engine_args"]

        self.host = self.config_engine.get("host", "0.0.0.0")
        self.port = self.config_engine.get("port", 5000)
        self.model = None
        self.preprocessor = None
        self.postprocessor = None

        self.load_model()

    def load_model(self):
        t_s = time.perf_counter()
        pretrained_dir = self.config_engine.model
        self.model = TrainablePolicy.from_pretrained(
            pretrained_dir, device=self.config_engine.device
        )
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_preprocessor.json",
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_postprocessor.json",
        )
        # TODO: (yupu): model.to(dtype)?
        logger.info(f"Policy model loading latency: {time.perf_counter() - t_s:.2f}s")

    def inference(self, batch):
        logger.info("Start to inference")
        # {
        #     "observation.images.image": np.ndarray, shape=(224, 224, 3), dtype=uint8,
        #     "observation.images.wrist_image": np.ndarray, shape=(224, 224, 3), dtype=uint8,
        #     "task": str,
        # }
        # NOTE: Images must be 224x224 resolution (uint8 HWC format).
        batch = self.preprocessor(batch)

        with torch.no_grad():
            action = self.model.predict_action(batch)
            logger.info(f"action before postprocessor: {action}")

        logger.info("Applying postprocessor...")
        action = self.postprocessor(action)

        # Convert to numpy for msgpack serialization
        action[ACTION] = action[ACTION].detach().cpu().numpy()

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


def main(config):
    policy = Policy(config)
    logger.info("Done")
    # start websocket server
    server = WebsocketPolicyServer(
        policy=policy,
        host=policy.host,
        port=policy.port,
        metadata={"env": "simpler_env"},
    )
    logger.info("Server running ...")
    server.serve_forever()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"][0])
