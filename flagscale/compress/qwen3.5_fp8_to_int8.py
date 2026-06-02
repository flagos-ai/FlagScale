import os
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

from llmcompressor import model_free_ptq


MODEL_ID = os.environ.get(
    "MODEL_ID",
    "/workspace/lhy/qwen3/Qwen3.5-35B-A3B-FP8",
)

SAVE_DIR = os.environ.get(
    "SAVE_DIR",
    "/workspace/lhy/llm-compressor/examples/model_free_ptq/test/Qwen3.5-35B-FP8-TO-W8A8-debug1yanzheng",
)

DEVICE = os.environ.get("DEVICE", "cuda:4")

IGNORE = [
    "lm_head",
    "re:.*mlp.gate$",
    "re:.*mlp.shared_expert_gate.*",
    "re:.*norm.*",
    "re:.*embed_tokens.*",
    "re:.*visual.*",
    "re:.*conv1d.*",
]


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_safetensor_files(model_dir: str) -> List[str]:
    files = sorted(str(p) for p in Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under: {model_dir}")
    return files


def load_all_tensors(model_dir: str) -> Dict[str, torch.Tensor]:
    state = {}
    for sf in list_safetensor_files(model_dir):
        part = load_file(sf, device="cpu")
        overlap = set(state.keys()) & set(part.keys())
        if overlap:
            raise ValueError(f"Duplicate tensor keys across shards: {list(overlap)[:5]}")
        state.update(part)
    return state


def get_quant_config(cfg: Dict) -> Dict:
    return cfg.get("quantization_config", {})


def is_fp8_model(cfg: Dict) -> bool:
    return get_quant_config(cfg).get("quant_method", "").lower() == "fp8"


def get_weight_block_size(cfg: Dict) -> Tuple[int, int]:
    block = get_quant_config(cfg).get("weight_block_size", [128, 128])
    if not isinstance(block, list) or len(block) != 2:
        raise ValueError(f"Unexpected weight_block_size: {block}")
    return int(block[0]), int(block[1])


def get_skip_modules(cfg: Dict) -> List[str]:
    return list(get_quant_config(cfg).get("modules_to_not_convert", []))


def is_candidate_weight(name: str) -> bool:
    return name.endswith(".weight")


def should_skip_weight(weight_name: str, skip_modules: List[str]) -> bool:
    for mod in skip_modules:
        if weight_name == mod or weight_name.startswith(mod + "."):
            return True
    return False


def candidate_scale_names(weight_name: str) -> List[str]:
    base = weight_name[:-len(".weight")] if weight_name.endswith(".weight") else weight_name
    return [
        f"{base}.weight_scale",
        f"{base}.weight_scales",
        f"{base}.scale",
        f"{base}.scales",
        f"{base}.weight_scale_inv",
        f"{weight_name}_scale",
        f"{weight_name}_scales",
    ]


def find_scale_tensor_name(state: Dict[str, torch.Tensor], weight_name: str) -> Optional[str]:
    for cand in candidate_scale_names(weight_name):
        if cand in state:
            return cand
    return None


def is_probably_inverse_scale(scale_name: str) -> bool:
    return (
        scale_name.endswith("_inv")
        or scale_name.endswith(".scale_inv")
        or scale_name.endswith(".weight_scale_inv")
    )


def is_auxiliary_scale_tensor(name: str) -> bool:
    keywords = [
        ".weight_scale",
        ".weight_scales",
        ".scale",
        ".scales",
        "_scale",
        "_scales",
    ]
    return any(k in name for k in keywords)


def block_dequant_fp8_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    block_h: int,
    block_w: int,
    out_dtype: torch.dtype = torch.bfloat16,
    scale_is_inverse: bool = False,
) -> torch.Tensor:
    if qweight.ndim != 2:
        raise ValueError(f"Only support 2D weight, got shape={tuple(qweight.shape)}")

    H, W = qweight.shape
    nbh = math.ceil(H / block_h)
    nbw = math.ceil(W / block_w)

    if scales.ndim != 2:
        raise ValueError(f"Expected 2D scales, got shape={tuple(scales.shape)}")

    if tuple(scales.shape) != (nbh, nbw):
        raise ValueError(
            f"Scale shape mismatch for {tuple(qweight.shape)}: "
            f"expected ({nbh}, {nbw}), got {tuple(scales.shape)}"
        )

    qweight = qweight.to(torch.bfloat16)
    scales = scales.to(torch.bfloat16)

    out = torch.empty((H, W), dtype=out_dtype)

    for bi in range(nbh):
        for bj in range(nbw):
            h0, h1 = bi * block_h, min((bi + 1) * block_h, H)
            w0, w1 = bj * block_w, min((bj + 1) * block_w, W)

            block = qweight[h0:h1, w0:w1]
            scale = scales[bi, bj]

            if scale_is_inverse:
                out[h0:h1, w0:w1] = block / scale
            else:
                out[h0:h1, w0:w1] = block * scale

    return out


def build_temp_bf16_from_fp8(fp8_model_dir: str) -> str:
   
    cfg = load_json(os.path.join(fp8_model_dir, "config.json"))
    if not is_fp8_model(cfg):
        raise ValueError(f"Input MODEL_ID is not an FP8 checkpoint: {fp8_model_dir}")

    block_h, block_w = get_weight_block_size(cfg)
    skip_modules = get_skip_modules(cfg)

    print(f"[INFO] FP8 block size = ({block_h}, {block_w})")
    print(f"[INFO] skip modules = {len(skip_modules)}")

    fp8_state = load_all_tensors(fp8_model_dir)

    tmp_dir = tempfile.mkdtemp(prefix="qwen35_fp8_bf16_")
    print(f"[INFO] temporary bf16 dir: {tmp_dir}")

    copy_names = [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "README.md",
        "LICENSE",
    ]
    for name in copy_names:
        src = os.path.join(fp8_model_dir, name)
        dst = os.path.join(tmp_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    tmp_cfg = json.loads(json.dumps(cfg))
    if "quantization_config" in tmp_cfg:
        tmp_cfg.pop("quantization_config")
    save_json(tmp_cfg, os.path.join(tmp_dir, "config.json"))

    out_state: Dict[str, torch.Tensor] = {}

    restored = 0
    kept_original = 0
    skipped_scale_tensors = 0
    failed = 0

    for name, tensor in fp8_state.items():
        if is_auxiliary_scale_tensor(name):
            skipped_scale_tensors += 1
            continue

        if not is_candidate_weight(name):
            if tensor.dtype.is_floating_point:
                out_state[name] = tensor.to(torch.bfloat16)
            else:
                out_state[name] = tensor
            kept_original += 1
            continue

        if should_skip_weight(name, skip_modules):
            if tensor.dtype.is_floating_point:
                out_state[name] = tensor.to(torch.bfloat16)
            else:
                out_state[name] = tensor
            kept_original += 1
            continue

        if tensor.ndim != 2:
            if tensor.dtype.is_floating_point:
                out_state[name] = tensor.to(torch.bfloat16)
            else:
                out_state[name] = tensor
            kept_original += 1
            continue

        scale_name = find_scale_tensor_name(fp8_state, name)
        if scale_name is None:
            if tensor.dtype.is_floating_point:
                out_state[name] = tensor.to(torch.bfloat16)
            else:
                out_state[name] = tensor
            kept_original += 1
            continue

        try:
            restored_weight = block_dequant_fp8_weight(
                qweight=tensor,
                scales=fp8_state[scale_name],
                block_h=block_h,
                block_w=block_w,
                out_dtype=torch.bfloat16,
                scale_is_inverse=is_probably_inverse_scale(scale_name),
            )
            out_state[name] = restored_weight
            restored += 1
        except Exception as e:
            print(f"[WARN] restore failed for {name}: {e}")
            if tensor.dtype.is_floating_point:
                out_state[name] = tensor.to(torch.bfloat16)
            else:
                out_state[name] = tensor
            kept_original += 1
            failed += 1

    save_file(out_state, os.path.join(tmp_dir, "model.safetensors"))

    print(
        f"[SUMMARY] restored={restored}, kept_original={kept_original}, "
        f"skipped_scale_tensors={skipped_scale_tensors}, failed={failed}"
    )

    return tmp_dir


def main():
    print(f"[INFO] MODEL_ID={MODEL_ID}")
    print(f"[INFO] SAVE_DIR={SAVE_DIR}")
    print(f"[INFO] DEVICE={DEVICE}")

    tmp_bf16_dir = build_temp_bf16_from_fp8(MODEL_ID)

    try:
        model_free_ptq(
            model_stub=tmp_bf16_dir,
            save_directory=SAVE_DIR,
            scheme="W8A8",
            ignore=IGNORE,
            max_workers=15,
            device=DEVICE,
        )
        print(f"[DONE] W8A8 model saved to: {SAVE_DIR}")
    finally:
        #print(f"[KEEP] temporary bf16 dir kept at: {tmp_bf16_dir}")
        if os.path.exists(tmp_bf16_dir):
            shutil.rmtree(tmp_bf16_dir, ignore_errors=True)
            print(f"[CLEANUP] removed temporary bf16 dir: {tmp_bf16_dir}")


if __name__ == "__main__":
    main()

