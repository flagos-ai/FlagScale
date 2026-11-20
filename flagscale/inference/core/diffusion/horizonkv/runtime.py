# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagScale-owned HorizonKV configuration and scoped native execution state."""
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

_NATIVE_LOCK = threading.RLock()
_ENV_PREFIXES = ("HORIZONKV_", "KV_SPARSE_", "SELF_FORCING_", "TIME_BREAKDOWN_")
PACKAGE_ROOT = Path(__file__).resolve().parent


def check_environment(require_cuda=False):
    """Check the supported native ABI before allocating model or host KV memory."""
    import torch

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if sys.version_info[:2] != (3, 10) or torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(
            f"HorizonKV currently requires a Python 3.10 / PyTorch 2.5.1 worker. Found Python {report['python']} / torch {report['torch']}. Use examples/horizonkv/setup_env.sh and experiment.cmds.before_start; do not install its CUDA extensions into the default FlagScale environment."
        )
    for name, version in (("triton", "3.1.0"), ("flash-attn", "2.8.3"), ("diffusers", "0.31.0")):
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing HorizonKV dependency: {name}=={version}") from exc
        if installed.split("+")[0] != version:
            raise RuntimeError(f"HorizonKV requires {name}=={version}, found {installed}")
        report[name] = installed
    for name in ("accelerate", "safetensors"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing model-loading dependency: {name}; run examples/horizonkv/setup_env.sh"
            ) from exc
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("HorizonKV requires an accessible NVIDIA CUDA device")
        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("HorizonKV requires NVIDIA compute capability >= 8.0")
        from flagscale.inference.core.diffusion.horizonkv.algorithm.sparse_backend import (
            rows_blocksparse_backend_is_available,
        )
        from flagscale.inference.core.diffusion.horizonkv.algorithm.transfer import _load_module

        if not rows_blocksparse_backend_is_available():
            raise RuntimeError(
                "Build HorizonKV's bundled Block-Sparse-Attention backend; the stock wheel is insufficient"
            )
        extension = _load_module()
        if extension is None or getattr(extension, "slot_plan_version", None) != 3:
            raise RuntimeError(
                "HorizonKV transfer extension is missing/outdated; enable HORIZONKV_FUSED_ONLOAD_VERBOSE=1 to diagnose"
            )
        report["gpu"] = torch.cuda.get_device_name()
    return report


class HorizonKVRuntime:
    """Resolve inference settings without an external checkout or experiment CLI."""

    def __init__(
        self,
        *,
        assets_root,
        checkpoint,
        method="horizonkv",
        cache=15,
        window=180,
        sink=1,
        topk=7,
        sparsity=0.5,
        min_frames=5,
        fixed=False,
        no_overlap=False,
        rasa=None,
        split_min_misses=None,
    ):
        from .algorithm.rasa_policy import load_policy, validate_policy

        if method != "horizonkv":
            raise ValueError(
                "Use HorizonKV for sparse inference or ARDiffusionPipeline for the baseline"
            )
        if any((type(n) is not int for n in (cache, window, sink, topk, min_frames))):
            raise ValueError("cache, window, sink, topk and min_frames must be integers")
        if not 0 <= sparsity < 1 or not math.isfinite(sparsity):
            raise ValueError("sparsity must be finite and in [0, 1)")
        if not 0 <= sink < cache or not sink + 4 <= min_frames <= topk <= cache:
            raise ValueError("Require sink + 4 <= min_frames <= topk <= cache")
        if window < topk:
            raise ValueError("window must be at least topk")
        policy = load_policy(rasa or PACKAGE_ROOT / "configs/rasa.json")
        if (
            policy["best_config"] != dict(frames=topk, block_sparsity=sparsity)
            or policy["min_config"]["frames"] != min_frames
        ):
            raise ValueError(
                "RASA policy must match topk, sparsity and min_frames; recalibrate for new settings"
            )
        threshold = (
            policy["overlap_min_miss_items"] if split_min_misses is None else split_min_misses
        )
        if type(threshold) is not int or threshold < 1:
            raise ValueError("split_min_misses must be a positive integer")
        if threshold > policy["switch_miss_items"]:
            raise ValueError("OAE overlap gate must not exceed the RASA switch gate")
        policy["overlap_min_miss_items"] = threshold
        validate_policy(policy)
        config = json.loads((PACKAGE_ROOT / "configs/horizonkv.json").read_text())
        config.update(
            max_stage1_kv_frames=topk,
            gpu_cache_frames=cache,
            block_target_density=1 - sparsity,
            overlap_block_sparsity=sparsity,
        )
        self._directory = tempfile.TemporaryDirectory(prefix="flagscale-horizonkv-")
        directory = Path(self._directory.name)
        config_path, policy_path = (directory / "kv_config.json", directory / "rasa.json")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        policy_path.write_text(json.dumps(policy, indent=2) + "\n")
        values = dict(
            OVERLAP_ATTN=int(not no_overlap),
            OVERLAP_FLASH_SPLIT_ATTN=int(not no_overlap),
            OVERLAP_SPLIT_MIN_MISS_ITEMS=threshold,
            ADAPTIVE_CONFIG=str(policy_path) if not fixed else "",
        )
        self.env = {"HORIZONKV_" + key: str(value) for key, value in values.items()}
        self.env.update(
            ATTN_DUMP="0",
            HORIZONKV_ROOT=str(Path(assets_root).expanduser().resolve()),
            BLOCK_SPARSE_ATTN_64_WARPS="4",
        )
        for key in ("HORIZONKV_FUSED_ONLOAD_BUILD_DIR", "HORIZONKV_FUSED_ONLOAD_VERBOSE"):
            if key in os.environ:
                self.env[key] = os.environ[key]
        self.plan = dict(
            schema_version=1,
            method=method,
            topk=topk,
            block_sparsity=sparsity,
            min_frames=min_frames,
            cache=cache,
            window=window,
            sink=sink,
            local_attn_size=-1,
            checkpoint=str(Path(checkpoint).expanduser().resolve()),
            kv_config=str(config_path),
            adaptive=not fixed,
            overlap=not no_overlap,
            long_video_gen=True,
        )
        self.installed = False

    @contextmanager
    def activated(self):
        with _NATIVE_LOCK:
            keys = {k for k in os.environ if k.startswith(_ENV_PREFIXES)} | set(self.env)
            previous = {key: os.environ.get(key) for key in keys}
            try:
                for key in keys:
                    os.environ.pop(key, None)
                os.environ.update(self.env)
                yield
            finally:
                for key in keys:
                    if previous[key] is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous[key]

    def snapshot(self):
        result = dict(self.plan)
        result["implementation"] = "flagscale.inference.core.diffusion.horizonkv"
        result["kv_config"] = (
            json.loads(Path(self.plan["kv_config"]).read_text()) if self.plan["kv_config"] else None
        )
        result["rasa_policy"] = json.loads((Path(self._directory.name) / "rasa.json").read_text())
        result["env"] = dict(self.env)
        return result

    def close(self):
        self._directory.cleanup()
