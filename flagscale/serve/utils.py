# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
