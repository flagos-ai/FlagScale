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
import json
import statistics
import time
from pathlib import Path

import torch
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
    benchmark_records = []

    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for idx, run_cfg in enumerate(runs):
        single_cfg = dict(run_cfg)
        benchmark = bool(single_cfg.pop("benchmark", False))
        warmup = int(single_cfg.pop("warmup", 0))
        measured_runs = int(single_cfg.pop("runs", 1))
        save_output = bool(single_cfg.pop("save", True))
        duration_seconds = single_cfg.pop("duration_seconds", None)
        name_prefix = single_cfg.pop("name", None) or "sample"
        if warmup < 0 or measured_runs < 1:
            raise ValueError("generate.warmup must be >= 0 and generate.runs must be positive")

        def generate_once(prefix: str | None, *, measure: bool) -> float:
            synchronize()
            started = time.perf_counter()
            outputs = engine.generate(**single_cfg)
            if duration_seconds is not None:
                target = int(duration_seconds * float(cfg.engine.output_fps))
                if target <= 0:
                    raise ValueError("generate.duration_seconds must be positive")
                is_horizonkv = "HorizonKVPipeline" in str(cfg.engine.pipeline.get("class", ""))
                first = 3 if is_horizonkv else 0
                outputs.frames = [frames[first : first + target] for frames in outputs.frames]
            if save_output and prefix is not None:
                engine.save(outputs, name_prefix=prefix)
            synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if measure:
                benchmark_records.append({"run": len(benchmark_records), "end_to_end_ms": elapsed})
            return elapsed

        if benchmark:
            for warmup_idx in range(warmup):
                generate_once(f"warmup_{warmup_idx:02d}", measure=False)
            for run_idx in range(measured_runs):
                generate_once(f"{name_prefix}_{run_idx:02d}", measure=True)
        else:
            generate_once(name_prefix, measure=False)

    if benchmark_records:
        times = [record["end_to_end_ms"] for record in benchmark_records]
        summary = {
            "warmup": warmup,
            "runs": len(times),
            "end_to_end_ms": times,
            "median_end_to_end_ms": statistics.median(times),
        }
        path = Path(str(cfg.engine.results_path))
        path.mkdir(parents=True, exist_ok=True)
        (path / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"[inference] benchmark runs={len(times)} "
            f"median_e2e={summary['median_end_to_end_ms']:.0f}ms",
            flush=True,
        )


if __name__ == "__main__":
    parsed_cfg = parse_config()
    assert isinstance(parsed_cfg, DictConfig)  # To make pyright happy
    inference(parsed_cfg)
    logger.info("Inference model inference completed")
