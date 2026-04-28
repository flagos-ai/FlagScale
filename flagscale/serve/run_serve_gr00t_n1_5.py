# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/deployment/model_server/server_policy.py

import argparse
import importlib
import time

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import flagscale.models.vla.gr00t_n1_5.processor_gr00t  # noqa: F401  register GR00T processor steps
from flagscale.logger import logger
from flagscale.models.utils.constants import ACTION
from flagscale.serve.websocket_policy_server import WebsocketPolicyServer
from flagscale.train.utils.train_utils import load_checkpoint


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
        model_variant = self.config_engine.model_variant
        policy = getattr(importlib.import_module("flagscale.models.vla"), model_variant)
        self.model, self.preprocessor, self.postprocessor = load_checkpoint(
            self.config_engine.model, policy, self.config_engine.device
        )
        logger.info(f"Policy model loading latency: {time.perf_counter() - t_s:.2f}s")

    def inference(self, batch):
        logger.info("Start to inference")
        # {
        #     "observation.images.image": np.ndarray, shape=(224, 224, 3), dtype=uint8,
        #     "observation.images.wrist_image": np.ndarray, shape=(224, 224, 3), dtype=uint8,
        #     "observation.state": np.ndarray, shape=(N,), dtype=float32,
        #     "task": str,
        # }
        # NOTE: Images must be 224x224 resolution (uint8 HWC format).
        # TODO: (yupu) Add explicit numpy-to-tensor conversion here before preprocessing,
        # instead of relying on ad-hoc conversions inside each processor step.

        # Debug: log incoming keys and state info
        logger.info(f"incoming keys: {list(batch.keys())}")
        if "observation.state" in batch:
            s = batch["observation.state"]
            logger.info(
                f"observation.state: type={type(s).__name__}, shape={s.shape if hasattr(s, 'shape') else 'N/A'}, values={s}"
            )

        batch = self.preprocessor(batch)

        with torch.no_grad():
            action = self.model.predict_action(batch)
            a_raw = action[ACTION]
            logger.info(
                f"action before postprocessor: shape={a_raw.shape}, first_step_7={a_raw[0, 0, :7]}"
            )

        logger.info("Applying postprocessor...")
        action = self.postprocessor(action)

        # Convert to numpy for msgpack serialization
        action[ACTION] = action[ACTION].detach().cpu().numpy()

        # Debug: log action shape and first-timestep values after postprocessing
        a = action[ACTION]
        logger.info(f"action after postprocessor: shape={a.shape}, first_step={a[0, 0]}")

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
