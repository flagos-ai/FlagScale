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

"""Standalone smoke test for FlagScale straggler detection.

This script does not depend on ``run.py`` or Megatron training startup.
It initializes torch.distributed directly, records a couple of synthetic
sections, and periodically emits a FlagScale straggler report.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

from flagscale.runner.straggler import OptionalSectionContext, StragglerConfig, StragglerDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone FlagScale straggler smoke test.")
    parser.add_argument("--steps", type=int, default=12, help="Total synthetic steps to run.")
    parser.add_argument(
        "--profiling-interval",
        type=int,
        default=5,
        help="Profile every N steps.",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="Emit a report every N steps.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Slowdown ratio threshold used to mark a rank as a straggler.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Skip the first N steps before profiling.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8,
        help="Recent sample count used when averaging section timings.",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=4096,
        help="Square matrix size used for synthetic GPU work.",
    )
    parser.add_argument(
        "--matmul-iters",
        type=int,
        default=2,
        help="Number of matmuls in the forward_backward section.",
    )
    parser.add_argument(
        "--forward-sleep-ms",
        type=float,
        default=8.0,
        help="Extra CPU sleep in forward_backward section.",
    )
    parser.add_argument(
        "--optimizer-sleep-ms",
        type=float,
        default=3.0,
        help="Extra CPU sleep in optimizer section.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory used to save report json files.",
    )
    return parser.parse_args()


def init_dist():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, use_cuda


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def make_detector(args, rank: int, world_size: int, local_rank: int, use_cuda: bool):
    hostname = socket.gethostname()
    node_name = f"{hostname}:gpu{local_rank}" if use_cuda else hostname
    config = StragglerConfig(
        enabled=True,
        profiling_interval=args.profiling_interval,
        report_interval_steps=args.report_interval,
        warmup_steps=args.warmup_steps,
        sample_size=args.sample_size,
        straggler_threshold=args.threshold,
        enable_gpu_profile=use_cuda,
        monitor_sections=["forward_backward", "optimizer"],
    )
    return StragglerDetector(config=config, rank=rank, world_size=world_size, node_name=node_name)


def allocate_work_tensors(size: int, use_cuda: bool, local_rank: int):
    if not use_cuda:
        return None, None
    device = torch.device("cuda", local_rank)
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    return a, b


def run_forward_backward(a, b, matmul_iters: int, forward_sleep_ms: float, use_cuda: bool):
    if use_cuda:
        for _ in range(matmul_iters):
            _ = a @ b
        torch.cuda.synchronize()
    if forward_sleep_ms > 0:
        time.sleep(forward_sleep_ms / 1000.0)


def run_optimizer(optimizer_sleep_ms: float):
    if optimizer_sleep_ms > 0:
        time.sleep(optimizer_sleep_ms / 1000.0)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank, use_cuda = init_dist()
    detector = make_detector(args, rank, world_size, local_rank, use_cuda)
    a, b = allocate_work_tensors(args.matmul_size, use_cuda, local_rank)

    if rank == 0:
        print(
            f"[rank0] Starting straggler smoke test: world_size={world_size}, "
            f"profiling_interval={args.profiling_interval}, report_interval={args.report_interval}, "
            f"output_dir={output_dir}"
        )

    try:
        for step in range(1, args.steps + 1):
            detector.current_step = step
            should_profile = detector.should_profile(step)

            with OptionalSectionContext(
                detector,
                "forward_backward",
                enabled=should_profile,
                profile_cuda=use_cuda,
            ):
                run_forward_backward(
                    a=a,
                    b=b,
                    matmul_iters=args.matmul_iters,
                    forward_sleep_ms=args.forward_sleep_ms,
                    use_cuda=use_cuda,
                )

            with OptionalSectionContext(
                detector,
                "optimizer",
                enabled=should_profile,
                profile_cuda=False,
            ):
                run_optimizer(optimizer_sleep_ms=args.optimizer_sleep_ms)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if detector.should_report(step):
                report = detector.generate_report(step=step)
                if rank == 0:
                    hostname = socket.gethostname()
                    report_path = output_dir / f"straggler_report_{hostname}_step_{step}.json"
                    detector.save_report(report, str(report_path))
                    print(report.to_text(), flush=True)
                    print(f"[rank0] Saved report to {report_path}", flush=True)
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
