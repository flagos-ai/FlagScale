#!/usr/bin/env python3
# ==============================================================
#  restore_from_compressed.py
#
#  * 加载压缩权重目录（prepare_bf16_weights.py 生成）
#  * 解压复原 → 写回模型
#  * 简单 forward 验证
# ==============================================================
import argparse
import json
import os
import struct
import time

import numpy as np
import torch
import zstandard as zstd
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument(
    "--compdir", default="./zstd_comped_weights", help="prepare_bf16_weights.py 生成的目录"
)
parser.add_argument("--test_text", default="Hello, world", help="随便来一句做 forward 验证")
parser.add_argument(
    "--check_diff", action="store_true", help="同时加载原始模型并比对差值（耗显存）"
)
args = parser.parse_args()

# ----------------- 1. 加载模型 + LoRA -----------------
print("→ loading base model …")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
)
peft_cfg = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.0,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_cfg, autocast_adapter_dtype=True)
model.eval()  # 只测试推理

# ----------------- 2. 恢复冻结权重 -----------------
print("→ restoring frozen bf16 parameters …")
dctx = zstd.ZstdDecompressor()
index = json.load(open(os.path.join(args.compdir, "index.json")))
decomp_time = 0
for item in index:
    bin_path = os.path.join(args.compdir, item["file"])
    dtype_name = item["dtype"]
    dtype = getattr(torch, dtype_name, None)
    if item["scheme"] == "raw_torch":
        restored = torch.load(bin_path, map_location="cpu")
        if not isinstance(restored, torch.Tensor):
            raise TypeError(f"raw_torch entry is not a tensor: {item['name']}")
        restored = restored.reshape(item["shape"])
    elif item["scheme"] == "split_zstd":
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"split_zstd does not support dtype {dtype_name!r}")
        with open(bin_path, "rb") as f:
            numel_bytes = f.read(8)
            if len(numel_bytes) != 8:
                raise ValueError(f"Truncated numel header: {bin_path}")
            (numel,) = struct.unpack("<Q", numel_bytes)
            expected_numel = int(np.prod(item["shape"]))
            if numel != expected_numel:
                raise ValueError(
                    f"numel mismatch for {item['name']}: file={numel}, index={expected_numel}"
                )
            sm_size = numel * (1 if dtype == torch.bfloat16 else 3)
            sign_mant = np.frombuffer(f.read(sm_size), dtype=np.uint8)
            if sign_mant.size != sm_size:
                raise ValueError(f"Truncated sign/mantissa payload: {bin_path}")
            exp_comp = f.read()

        t0 = time.time()
        exp_u8 = np.frombuffer(dctx.decompress(exp_comp, max_output_size=numel), dtype=np.uint8)
        decomp_time += time.time() - t0
        if exp_u8.size != numel:
            raise ValueError(f"Exponent size mismatch for {item['name']}")

        if dtype == torch.bfloat16:
            sm_u16 = sign_mant.astype(np.uint16)
            sign_u16 = (sm_u16 & 0x80) << 8
            mant_u16 = sm_u16 & 0x7F
            bits = sign_u16 | (exp_u8.astype(np.uint16) << 7) | mant_u16
            restored = torch.from_numpy(bits).view(torch.bfloat16)
        else:
            sm_u32 = (
                sign_mant.reshape(numel, 3).astype(np.uint32)[:, 0]
                | (sign_mant.reshape(numel, 3).astype(np.uint32)[:, 1] << 8)
                | (sign_mant.reshape(numel, 3).astype(np.uint32)[:, 2] << 16)
            )
            sign = (sm_u32 >> 23) & 0x1
            bits = (sign << 31) | (exp_u8.astype(np.uint32) << 23) | (sm_u32 & 0x7FFFFF)
            restored = torch.from_numpy(bits).view(torch.float32)
        restored = restored.reshape(item["shape"])
    else:
        raise ValueError(f"Unknown compression scheme: {item['scheme']}")

    # ---- 覆写到模型 ----
    module, _, attr = item["name"].rpartition(".")
    target = getattr(dict(model.named_modules())[module], attr)
    if tuple(target.shape) != tuple(restored.shape):
        raise ValueError(
            f"Shape mismatch for {item['name']}: model={tuple(target.shape)}, "
            f"compressed={tuple(restored.shape)}"
        )
    target.data.copy_(restored.to(device=target.device, dtype=target.dtype))

print(f"✅ restored {len(index)} tensors")

# ----------------- 3. (可选) 与原始模型对比 -----------------
if args.check_diff:
    print("→ loading a fresh copy for diff …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    ref_model = get_peft_model(ref_model, peft_cfg, autocast_adapter_dtype=True)

    max_abs = 0.0
    for n, p in ref_model.named_parameters():
        q = dict(model.named_parameters())[n]
        if not p.requires_grad:  # 只比冻结权重
            diff = (p.data - q.data).abs().max().item()
            max_abs = max(max_abs, diff)
    print(f"🔎 max |Δ| on frozen params = {max_abs:e}")

# ----------------- 4. 一次 forward 自验 -----------------
tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
tok.pad_token = tok.unk_token
tok.padding_side = "left"

inputs = tok(args.test_text, return_tensors="pt")
with torch.no_grad():
    outs = model(**{k: v.to(model.device) for k, v in inputs.items()})
    logits_sum = outs.logits.float().sum().item()
print(
    f"🤖 forward OK, logits sum = {logits_sum:.4f}, decompression time = {decomp_time * 1000:.2f} ms"
)
