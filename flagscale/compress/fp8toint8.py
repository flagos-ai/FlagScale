"""
FP8 → W8A8 generic quantization tool.

Architecture:
  [IO layer]        ShardedCheckpoint  — shard read/write, single-file/multi-shard compatible
  [Strategy ABC]    DequantStrategy    — abstract base: should_dequant / dequant / is_auxiliary
  [Strategy impl]   Fp8LinearDequantStrategy — FP8 Linear dequant, block_size from config,
                                               scale names auto-probed
  [Main flow]       main()             — read config → build strategy → convert BF16 → W8A8
"""

import abc
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from llmcompressor import model_free_ptq

#Global defaults
MODEL_ID = os.environ.get("MODEL_ID", "./models")
SAVE_DIR = os.environ.get("SAVE_DIR", "./output")
_device_env = os.environ.get("DEVICE", "auto")
if _device_env == "auto":
    DEVICE = None
elif "," in _device_env:
    DEVICE = [d.strip() for d in _device_env.split(",")]
else:
    DEVICE = _device_env

IGNORE = [
    "lm_head",
    "re:.*mlp.gate$",
    "re:.*mlp.shared_expert_gate.*",
    "re:.*norm.*",
    "re:.*embed_tokens.*",
    "re:.*visual.*",
    "re:.*conv1d.*",
]

# Files to copy verbatim from the source checkpoint
_CONFIG_FILES = [
    "config.json", "configuration.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json", "merges.txt", "vocab.json",
    "chat_template.jinja", "README.md", "LICENSE",
]

# Auxiliary-tensor detection keywords
_SCALE_KEYWORDS = [
    ".weight_scale", ".weight_scales", ".scale", ".scales",
    "_scale", "_scales",
]


#  IO Tool Layer — ShardedCheckpoint
class ShardedCheckpoint:
    """Read / write sharded safetensors checkpoints.

    Supports both multi-shard (with ``model.safetensors.index.json``) and
    single-file layouts.  Model-agnostic.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._index: Optional[Dict] = None
        self._shard_to_tensors: Optional[Dict[str, List[str]]] = None

    @staticmethod
    def load_json(path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(obj: Dict, path: str):
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def load_config(self) -> Dict:
        return self.load_json(os.path.join(self.model_dir, "config.json"))

    def _load_index(self) -> Dict:
        if self._index is None:
            index_path = os.path.join(
                self.model_dir, "model.safetensors.index.json"
            )
            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"model.safetensors.index.json not found in {self.model_dir}. "
                    "Single-file models are not supported by this multi-shard path."
                )
            self._index = self.load_json(index_path)
        return self._index

    def shard_files(self) -> List[str]:
        """Return sorted list of shard filenames."""
        index_path = os.path.join(
            self.model_dir, "model.safetensors.index.json"
        )
        if os.path.exists(index_path):
            index = self._load_index()
            weight_map: Dict[str, str] = index["weight_map"]
            shard_to_tensors: Dict[str, List[str]] = {}
            for tensor_name, shard_file in weight_map.items():
                shard_to_tensors.setdefault(shard_file, []).append(tensor_name)
            self._shard_to_tensors = shard_to_tensors
            return sorted(shard_to_tensors.keys())
        # Fallback: single-file model.safetensors
        single = os.path.join(self.model_dir, "model.safetensors")
        if os.path.exists(single):
            print("[INFO] single-file checkpoint detected: model.safetensors")
            return ["model.safetensors"]
        raise FileNotFoundError(
            f"No model.safetensors.index.json or model.safetensors "
            f"found in {self.model_dir}"
        )

    def load_shard(self, shard_file: str) -> Dict[str, torch.Tensor]:
        shard_path = os.path.join(self.model_dir, shard_file)
        return load_file(shard_path, device="cpu")

    @staticmethod
    def write_shard(
        out_dir: str,
        shard_file: str,
        tensors: Dict[str, torch.Tensor],
    ) -> int:
        """Save *tensors* into *out_dir/shard_file*, return total bytes."""
        os.makedirs(out_dir, exist_ok=True)
        save_file(tensors, os.path.join(out_dir, shard_file))
        return sum(t.nbytes for t in tensors.values())

    @staticmethod
    def write_index(out_dir: str, weight_map: Dict[str, str], total_size: int):
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        ShardedCheckpoint.save_json(
            index, os.path.join(out_dir, "model.safetensors.index.json")
        )

    def copy_config_files(self, dst_dir: str):
        """Copy auxiliary config / tokenizer files to *dst_dir*."""
        for name in _CONFIG_FILES:
            src = os.path.join(self.model_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, name))


#  Strategy Abstraction Layer — DequantStrategy
class DequantStrategy(abc.ABC):
    """Defines the interface every dequantization strategy must implement."""

    @abc.abstractmethod
    def should_dequant(
        self,
        name: str,
        tensor: torch.Tensor,
        shard_state: Dict[str, torch.Tensor],
    ) -> bool:
        """Return True if *name* / *tensor* should be dequantized."""

    @abc.abstractmethod
    def dequant(
        self,
        name: str,
        tensor: torch.Tensor,
        shard_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the dequantized (BF16) tensor."""

    @abc.abstractmethod
    def is_auxiliary(self, name: str) -> bool:
        """Return True if *name* is a helper tensor (e.g. scale) to drop."""


#  Strategy Implementation — Fp8LinearDequantStrategy
class Fp8LinearDequantStrategy(DequantStrategy):
    """FP8 per-block dequantization for Linear layers.

    * ``block_size`` is read from ``quantization_config.weight_block_size``.
    * Scale tensor names are auto-probed via a candidate list — no
      model-specific naming is hard-coded.
    """

    # Candidate suffixes for locating the scale tensor of a given weight.
    _SCALE_CANDIDATES = (
        ".weight_scale",
        ".weight_scales",
        ".scale",
        ".scales",
        ".weight_scale_inv",
    )
    _SCALE_SUFFIX_FULL = ("_scale", "_scales")

    def __init__(self, block_h: int, block_w: int, skip_modules: List[str]):
        self.block_h = block_h
        self.block_w = block_w
        self.skip_modules = skip_modules

    @staticmethod
    def _candidate_scale_names(weight_name: str) -> List[str]:
        base = (
            weight_name[: -len(".weight")]
            if weight_name.endswith(".weight")
            else weight_name
        )
        return [
            f"{base}.weight_scale",
            f"{base}.weight_scales",
            f"{base}.scale",
            f"{base}.scales",
            f"{base}.weight_scale_inv",
            f"{weight_name}_scale",
            f"{weight_name}_scales",
        ]

    @staticmethod
    def _find_scale_name(
        shard_state: Dict[str, torch.Tensor], weight_name: str
    ) -> Optional[str]:
        for cand in Fp8LinearDequantStrategy._candidate_scale_names(weight_name):
            if cand in shard_state:
                return cand
        return None

    @staticmethod
    def _should_skip(weight_name: str, skip_modules: List[str]) -> bool:
        for mod in skip_modules:
            if weight_name == mod or weight_name.startswith(mod + "."):
                return True
        return False

    def should_dequant(
        self,
        name: str,
        tensor: torch.Tensor,
        shard_state: Dict[str, torch.Tensor],
    ) -> bool:
        if not name.endswith(".weight"):
            return False
        if self._should_skip(name, self.skip_modules):
            return False
        if tensor.ndim != 2:
            return False
        return self._find_scale_name(shard_state, name) is not None

    def dequant(
        self,
        name: str,
        tensor: torch.Tensor,
        shard_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        scale_name = self._find_scale_name(shard_state, name)
        if scale_name is None:
            raise RuntimeError(f"No scale tensor found for {name}")
        return self._block_dequant(
            qweight=tensor,
            scales=shard_state[scale_name],
            block_h=self.block_h,
            block_w=self.block_w,
        )

    def is_auxiliary(self, name: str) -> bool:
        return any(k in name for k in _SCALE_KEYWORDS)

    @staticmethod
    def _block_dequant(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        block_h: int,
        block_w: int,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if qweight.ndim != 2:
            raise ValueError(
                f"Only support 2D weight, got shape={tuple(qweight.shape)}"
            )
        H, W = qweight.shape
        nbh = math.ceil(H / block_h)
        nbw = math.ceil(W / block_w)
        if scales.ndim != 2:
            raise ValueError(
                f"Expected 2D scales, got shape={tuple(scales.shape)}"
            )
        if tuple(scales.shape) != (nbh, nbw):
            raise ValueError(
                f"Scale shape mismatch for {tuple(qweight.shape)}: "
                f"expected ({nbh}, {nbw}), got {tuple(scales.shape)}"
            )
        qweight = qweight.to(torch.bfloat16)
        scales = scales.to(torch.bfloat16)
        out = torch.empty((H, W), dtype=out_dtype, device=qweight.device)
        for bi in range(nbh):
            for bj in range(nbw):
                h0, h1 = bi * block_h, min((bi + 1) * block_h, H)
                w0, w1 = bj * block_w, min((bj + 1) * block_w, W)
                out[h0:h1, w0:w1] = qweight[h0:h1, w0:w1] * scales[bi, bj]
        return out


#  Strategy Implementation — Fp8ChannelDequantStrategy
class Fp8ChannelDequantStrategy(DequantStrategy):
    """FP8 per-channel dequantization for compressed-tensors models.

    Handles models where ``quant_method == "compressed-tensors"`` and weights
    are stored in ``float8_e4m3fn`` with per-channel (per-row) scales of shape
    ``(out_features, 1)``.  Dequant is simply ``weight.to(bf16) * scale``.

    Re-uses the same scale-probing logic as ``Fp8LinearDequantStrategy``.
    """

    _SCALE_CANDIDATES = Fp8LinearDequantStrategy._SCALE_CANDIDATES
    _SCALE_SUFFIX_FULL = Fp8LinearDequantStrategy._SCALE_SUFFIX_FULL

    def __init__(self, skip_modules: List[str]):
        self.skip_modules = skip_modules

    def should_dequant(
        self,
        name: str,
        tensor: torch.Tensor,
        shard_state: Dict[str, torch.Tensor],
    ) -> bool:
        if not name.endswith(".weight"):
            return False
        if Fp8LinearDequantStrategy._should_skip(name, self.skip_modules):
            return False
        if tensor.ndim != 2:
            return False
        if not tensor.dtype.is_floating_point or tensor.element_size() != 1:
            # float8 types have element_size() == 1
            return False
        return Fp8LinearDequantStrategy._find_scale_name(shard_state, name) is not None

    def dequant(
        self,
        name: str,
        tensor: torch.Tensor,
        shard_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        scale_name = Fp8LinearDequantStrategy._find_scale_name(shard_state, name)
        if scale_name is None:
            raise RuntimeError(f"No scale tensor found for {name}")
        scales = shard_state[scale_name]
        return tensor.to(torch.bfloat16) * scales.to(torch.bfloat16)

    def is_auxiliary(self, name: str) -> bool:
        return any(k in name for k in _SCALE_KEYWORDS)


#  Builder: create the right strategy from a config dict
def _get_quant_config(cfg: Dict) -> Dict:
    return cfg.get("quantization_config", {})


def _is_fp8_model(cfg: Dict) -> bool:
    return _get_quant_config(cfg).get("quant_method", "").lower() == "fp8"


def _is_compressed_tensors_fp8(cfg: Dict) -> bool:
    """Detect compressed-tensors FP8 models (e.g. neuralmagic FP8-dynamic).

    Returns True when quant_method is "compressed-tensors" and at least one
    config group has float-type 8-bit weights.
    """
    qc = _get_quant_config(cfg)
    if qc.get("quant_method", "").lower() != "compressed-tensors":
        return False
    for group in qc.get("config_groups", {}).values():
        w = group.get("weights", {})
        if w.get("type") == "float" and w.get("num_bits") == 8:
            return True
    return False


def _get_weight_block_size(cfg: Dict) -> Tuple[int, int]:
    block = _get_quant_config(cfg).get("weight_block_size", [128, 128])
    if not isinstance(block, list) or len(block) != 2:
        raise ValueError(f"Unexpected weight_block_size: {block}")
    return int(block[0]), int(block[1])


def _get_skip_modules(cfg: Dict) -> List[str]:
    return list(_get_quant_config(cfg).get("modules_to_not_convert", []))


def build_strategy(cfg: Dict) -> Optional[DequantStrategy]:
    """Return the appropriate strategy, or *None* for non-FP8 models."""
    if _is_fp8_model(cfg):
        qc = _get_quant_config(cfg)
        wbs = qc.get("weight_block_size")
        if isinstance(wbs, list) and len(wbs) == 2:
            # Block-wise FP8 (llm-compressor FP8_BLOCK)
            block_h, block_w = int(wbs[0]), int(wbs[1])
            skip_modules = _get_skip_modules(cfg)
            print(f"[INFO] FP8 block size = ({block_h}, {block_w})")
            print(f"[INFO] skip modules count = {len(skip_modules)}")
            return Fp8LinearDequantStrategy(block_h, block_w, skip_modules)
        else:
            # Per-tensor FP8 (legacy AutoFP8, no block_size)
            skip_modules = list(qc.get("ignored_layers", []))
            print(f"[INFO] FP8 per-tensor detected (no weight_block_size)")
            print(f"[INFO] skip modules count = {len(skip_modules)}")
            return Fp8ChannelDequantStrategy(skip_modules)
    if _is_compressed_tensors_fp8(cfg):
        skip_modules = list(_get_quant_config(cfg).get("ignore", []))
        print(f"[INFO] compressed-tensors FP8 detected, per-channel dequant")
        print(f"[INFO] skip modules count = {len(skip_modules)}")
        return Fp8ChannelDequantStrategy(skip_modules)
    return None


#  Conversion pipeline
def _to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Cast floating-point tensors to BF16; leave integer tensors alone."""
    return tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor


def build_temp_bf16(model_dir: str) -> str:
    """Convert a (possibly FP8) checkpoint to a temporary BF16 directory.

    * FP8 models  — dequantize weights, drop scales, strip quantization_config.
    * Non-FP8     — copy as-is (cast floats to BF16), no error.
    """
    ckpt = ShardedCheckpoint(model_dir)
    cfg = ckpt.load_config()
    strategy = build_strategy(cfg)

    shard_files = ckpt.shard_files()
    print(f"[INFO] source shards = {len(shard_files)}")

    tmp_dir = tempfile.mkdtemp(prefix="fp8_to_bf16_")
    print(f"[INFO] temporary bf16 dir: {tmp_dir}")

    # Copy auxiliary files
    ckpt.copy_config_files(tmp_dir)

    # Strip quantization_config so llm-compressor treats it as plain BF16
    tmp_cfg = json.loads(json.dumps(cfg))
    tmp_cfg.pop("quantization_config", None)
    ShardedCheckpoint.save_json(tmp_cfg, os.path.join(tmp_dir, "config.json"))

    # Counters
    restored = kept_original = skipped_scale_tensors = failed = 0
    new_weight_map: Dict[str, str] = {}
    total_size = 0

    for shard_file in shard_files:
        shard_state = ckpt.load_shard(shard_file)
        out_shard: Dict[str, torch.Tensor] = {}

        for name, tensor in shard_state.items():
            # Drop auxiliary scale tensors when a strategy is active
            if strategy is not None and strategy.is_auxiliary(name):
                skipped_scale_tensors += 1
                continue

            # Attempt dequantization
            if strategy is not None and strategy.should_dequant(
                name, tensor, shard_state
            ):
                try:
                    out_shard[name] = strategy.dequant(name, tensor, shard_state)
                    restored += 1
                except Exception as e:
                    print(f"[WARN] dequant failed for {name}: {e}")
                    out_shard[name] = _to_bf16(tensor)
                    kept_original += 1
                    failed += 1
                continue

            # Passthrough (non-target tensor or no strategy)
            out_shard[name] = _to_bf16(tensor)
            kept_original += 1

        shard_bytes = ShardedCheckpoint.write_shard(tmp_dir, shard_file, out_shard)
        for key in out_shard:
            new_weight_map[key] = shard_file
        total_size += shard_bytes
        print(f"[SHARD] {shard_file}: {len(out_shard)} tensors written")

    ShardedCheckpoint.write_index(tmp_dir, new_weight_map, total_size)

    print(
        f"[SUMMARY] restored={restored}, kept_original={kept_original}, "
        f"skipped_scale_tensors={skipped_scale_tensors}, failed={failed}"
    )
    return tmp_dir


def main():
    _num_devices = len(DEVICE) if isinstance(DEVICE, list) else (
        torch.cuda.device_count() if DEVICE is None else 1
    )
    _max_workers = int(os.environ.get("MAX_WORKERS", str(_num_devices)))
    print(f"[INFO] MODEL_ID={MODEL_ID}")
    print(f"[INFO] SAVE_DIR={SAVE_DIR}")
    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] max_workers={_max_workers}, num_devices={_num_devices}")

    tmp_bf16_dir = build_temp_bf16(MODEL_ID)

    try:
        model_free_ptq(
            model_stub=tmp_bf16_dir,
            save_directory=SAVE_DIR,
            scheme="W8A8",
            ignore=IGNORE,
            max_workers=_max_workers,
            device=DEVICE,
        )
        print(f"[DONE] W8A8 model saved to: {SAVE_DIR}")
    finally:
        if os.path.exists(tmp_bf16_dir):
            shutil.rmtree(tmp_bf16_dir, ignore_errors=True)
            print(f"[CLEANUP] removed temporary bf16 dir: {tmp_bf16_dir}")


if __name__ == "__main__":
    main()
