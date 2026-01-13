import argparse

import yaml
from omegaconf import OmegaConf

_g_ignore_fields = ["experiment", "action"]


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument("--log-dir", type=str, default="outputs", help="Path to the model")
    args = parser.parse_args()

    # Open the YAML file and convert it into a dictionary
    with open(args.config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    # Convert the dictionary into a DictConfig
    config = OmegaConf.create(yaml_dict)
    return config
