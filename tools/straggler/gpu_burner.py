#!/usr/bin/env python3

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

"""Simple external GPU burner used to trigger straggler behavior."""

from __future__ import annotations

import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="External GPU burner for straggler tests.")
    parser.add_argument(
        "--size",
        type=int,
        default=12288,
        help="Square matrix size used for the burn loop.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Tensor dtype used for the burn loop.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print a progress line every N iterations.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for gpu_burner.py")

    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda", 0)

    a = torch.randn(args.size, args.size, device=device, dtype=dtype)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype)

    print(
        f"Starting GPU burner on device={torch.cuda.get_device_name(device)} "
        f"size={args.size} dtype={args.dtype}",
        flush=True,
    )

    iteration = 0
    while True:
        _ = a @ b
        torch.cuda.synchronize()
        iteration += 1
        if iteration % args.log_interval == 0:
            print(f"burn iterations={iteration} ts={time.time():.0f}", flush=True)


if __name__ == "__main__":
    main()
