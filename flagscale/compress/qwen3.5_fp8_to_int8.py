import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
from llmcompressor import model_free_ptq
from safetensors.torch import load_file, save_file

MODEL_ID = os.environ.get("MODEL_ID", "./models")
SAVE_DIR = os.environ.get("SAVE_DIR", "./output")
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        ensure_dir(dir_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_safetensor_files(model_dir: str) -> list[str]:
    files = sorted(str(p) for p in Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under: {model_dir}")
    return files


def get_quant_config(cfg: dict) -> dict:
    return cfg.get("quantization_config", {})


def is_fp8_model(cfg: dict) -> bool:
    return get_quant_config(cfg).get("quant_method", "").lower() == "fp8"


def get_weight_block_size(cfg: dict) -> tuple[int, int]:
    block = get_quant_config(cfg).get("weight_block_size", [128, 128])
    if not isinstance(block, list) or len(block) != 2:
        raise ValueError(f"Unexpected weight_block_size: {block}")
    return int(block[0]), int(block[1])


def get_skip_modules(cfg: dict) -> list[str]:
    return list(get_quant_config(cfg).get("modules_to_not_convert", []))


def is_candidate_weight(name: str) -> bool:
    return name.endswith(".weight")


def should_skip_weight(weight_name: str, skip_modules: list[str]) -> bool:
    for mod in skip_modules:
        if weight_name == mod or weight_name.startswith(mod + "."):
            return True
    return False


def candidate_scale_names(weight_name: str) -> list[str]:
    base = weight_name[: -len(".weight")] if weight_name.endswith(".weight") else weight_name
    return [
        f"{base}.weight_scale",
        f"{base}.weight_scales",
        f"{base}.scale",
        f"{base}.scales",
        f"{base}.weight_scale_inv",
        f"{weight_name}_scale",
        f"{weight_name}_scales",
    ]


def find_scale_tensor_name(state: dict[str, torch.Tensor], weight_name: str) -> str | None:
    for cand in candidate_scale_names(weight_name):
        if cand in state:
            return cand
    return None


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
    out = torch.empty((H, W), dtype=out_dtype, device=qweight.device)
    for bi in range(nbh):
        for bj in range(nbw):
            h0, h1 = bi * block_h, min((bi + 1) * block_h, H)
            w0, w1 = bj * block_w, min((bj + 1) * block_w, W)
            out[h0:h1, w0:w1] = qweight[h0:h1, w0:w1] * scales[bi, bj]
    return out


def build_temp_bf16_from_fp8(fp8_model_dir: str) -> str:
    cfg = load_json(os.path.join(fp8_model_dir, "config.json"))
    if not is_fp8_model(cfg):
        raise ValueError(f"Input MODEL_ID is not an FP8 checkpoint: {fp8_model_dir}")

    block_h, block_w = get_weight_block_size(cfg)
    skip_modules = get_skip_modules(cfg)
    print(f"[INFO] FP8 block size = ({block_h}, {block_w})")
    print(f"[INFO] skip modules count = {len(skip_modules)}")

    index_path = os.path.join(fp8_model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"model.safetensors.index.json not found in {fp8_model_dir}. "
            "Single-file models are not supported by this multi-shard path."
        )
    index = load_json(index_path)
    orig_weight_map: dict[str, str] = index["weight_map"]

    shard_to_tensors: dict[str, list[str]] = {}
    for tensor_name, shard_file in orig_weight_map.items():
        shard_to_tensors.setdefault(shard_file, []).append(tensor_name)

    shard_files = sorted(shard_to_tensors.keys())
    print(f"[INFO] source shards = {len(shard_files)}")

    tmp_dir = tempfile.mkdtemp(prefix="fp8_to_bf16_")
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
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_dir, name))

    tmp_cfg = json.loads(json.dumps(cfg))
    tmp_cfg.pop("quantization_config", None)
    save_json(tmp_cfg, os.path.join(tmp_dir, "config.json"))

    restored = kept_original = skipped_scale_tensors = failed = 0
    new_weight_map: dict[str, str] = {}
    total_size = 0

    for shard_file in shard_files:
        shard_path = os.path.join(fp8_model_dir, shard_file)
        fp8_shard: dict[str, torch.Tensor] = load_file(shard_path, device="cpu")
        out_shard: dict[str, torch.Tensor] = {}

        for name, tensor in fp8_shard.items():
            if is_auxiliary_scale_tensor(name):
                skipped_scale_tensors += 1
                continue

            if (
                not is_candidate_weight(name)
                or should_skip_weight(name, skip_modules)
                or tensor.ndim != 2
            ):
                out_shard[name] = (
                    tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor
                )
                kept_original += 1
                continue

            scale_name = find_scale_tensor_name(fp8_shard, name)
            if scale_name is None:
                out_shard[name] = (
                    tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor
                )
                kept_original += 1
                continue

            try:
                out_shard[name] = block_dequant_fp8_weight(
                    qweight=tensor,
                    scales=fp8_shard[scale_name],
                    block_h=block_h,
                    block_w=block_w,
                    out_dtype=torch.bfloat16,
                )
                restored += 1
            except Exception as e:
                print(f"[WARN] dequant failed for {name}: {e}")
                out_shard[name] = (
                    tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor
                )
                kept_original += 1
                failed += 1

        out_path = os.path.join(tmp_dir, shard_file)
        save_file(out_shard, out_path)
        for key, t in out_shard.items():
            new_weight_map[key] = shard_file
            total_size += t.nbytes
        print(f"[SHARD] {shard_file}: {len(out_shard)} tensors written")

    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    save_json(new_index, os.path.join(tmp_dir, "model.safetensors.index.json"))

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
        if os.path.exists(tmp_bf16_dir):
            shutil.rmtree(tmp_bf16_dir, ignore_errors=True)
            print(f"[CLEANUP] removed temporary bf16 dir: {tmp_bf16_dir}")


if __name__ == "__main__":
    main()
