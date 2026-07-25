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

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.inference.core.inference_engine import InferenceEngine
from flagscale.runner.utils import logger


def parse_config() -> DictConfig | ListConfig:
    """Parse the configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # TODO(yupu): Any checks?

    return config


def inference(cfg: DictConfig) -> None:
    """Run the model inference

    Args:
        cfg: The parsed configuration
    """

    engine = InferenceEngine(cfg.get("engine", {}))

    generate_cfg = cfg.get("generate", {})

    def _to_plain_dict(dc):
        if isinstance(dc, DictConfig):
            return {k: dc.get(k) for k in dc}
        return dc

    def _normalize_runs(gen_cfg):
        # List of full specs
        if isinstance(gen_cfg, (list, ListConfig)):
            return [(_to_plain_dict(spec) or {}) for spec in gen_cfg]

        # Dict-like specs
        if isinstance(gen_cfg, DictConfig):
            # prompts array with shared kwargs
            prompts = gen_cfg.get("prompts", None)
            if prompts is not None and len(prompts) > 0:
                base = {k: gen_cfg.get(k) for k in gen_cfg if k != "prompts"}
                return [dict(base, prompt=p) for p in prompts]

            # single run
            return [{k: gen_cfg.get(k) for k in gen_cfg}]

        # Fallback if plain dict provided
        return [gen_cfg]

    runs = _normalize_runs(generate_cfg)

    for idx, run_cfg in enumerate(runs):
        single_cfg = dict(run_cfg)
        name_prefix = single_cfg.pop("name", None) or f"sample_{idx}"
        outputs = engine.generate(**single_cfg)
        engine.save(outputs, name_prefix=name_prefix)


if __name__ == "__main__":
    parsed_cfg = parse_config()
    assert isinstance(parsed_cfg, DictConfig)  # To make pyright happy
    inference(parsed_cfg)
    logger.info("Inference model inference completed")
