import argparse
import threading
from functools import wraps

from omegaconf import OmegaConf

task_config = OmegaConf.create()


def load_once(func):
    loaded = False
    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal loaded
        with lock:
            if not loaded:
                func(*args, **kwargs)
                loaded = True

    return wrapper


@load_once
def load_args() -> None:
    """Load configuration for cluster init"""
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument("--config-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--log-dir", type=str, default="outputs", help="Path to the model")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    global task_config
    task_config.update(config)
    task_config.update({"log_dir": args.log_dir})

    return
