# === prepare_compressed_weights.py =================================================
import argparse
import json
import os
import struct
import time

import torch
import zstandard as zstd
from transformers import AutoModelForCausalLM

from flagscale.compress.memrift import ops as fs_sp

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--level", default=1)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print(f"{args.model=}, {args.outdir=}")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)

comp_time = 0
uncomp_bytes = 0
cpr = zstd.ZstdCompressor(level=int(args.level))
index = []  # <layer name, binary file, shape, dtype>
MiB = 1024 * 1024

for name, p in model.named_parameters():
    fn = f"{len(index):06}.bin"
    out = os.path.join(args.outdir, fn)

    # if p.requires_grad:
    #     continue

    if p.dtype == torch.float32:
        print(f"{name=}, {p.shape=}")
    else:
        if p.dtype != torch.bfloat16:
            print(f"{name=}, {p.shape=}, {p.dtype=}")

    if p.dtype in (torch.bfloat16, torch.float32):
        # ======= 使用 fs_sp CUDA kernel =======
        stm = torch.cuda.current_stream()
        with torch.cuda.stream(stm):
            cpu_exp, sm_bits = fs_sp.split(p, stm.cuda_stream)
        stm.synchronize()
        sm_cpu = sm_bits.cpu().contiguous()
        del sm_bits

        with open(out, "wb") as f:
            f.write(struct.pack("<Q", p.numel()))
            f.write(sm_cpu.numpy().tobytes())

            t0 = time.time()
            raw = cpu_exp.numpy().tobytes()
            comped_bytes = cpr.compress(raw)
            comp_time += time.time() - t0
            uncomp_bytes += len(raw)
            print(f"Throughput: {len(raw) / (time.time() - t0) / MiB} MiB/s")

            f.write(comped_bytes)
        scheme = "split_zstd"
    else:
        torch.save(p.detach().cpu(), out)
        scheme = "raw_torch"

    index.append(
        dict(
            name=name,
            file=fn,
            shape=list(p.shape),
            dtype=str(p.dtype).replace("torch.", ""),
            scheme=scheme,
        )
    )

json.dump(index, open(os.path.join(args.outdir, "index.json"), "w"))
print(
    f"✅  wrote {len(index)} frozen tensors → “{args.outdir}”, compression time: {comp_time * 1000:.2f} ms, throughput: {uncomp_bytes / comp_time / MiB} MiB/s"
)
