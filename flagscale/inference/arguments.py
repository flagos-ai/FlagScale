import argparse

from omegaconf import OmegaConf


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    assert config.get("llm", None) and config.get("generate", None)
    return config
