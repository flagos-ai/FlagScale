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

"""Standalone smoke test for the FlagScale performance monitor."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from flagscale.train.perf_monitor.hooks import (
    initialize_perf_monitor,
    perf_monitor_end_iteration,
    perf_monitor_end_training,
    perf_monitor_start_iteration,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone FlagScale perf monitor smoke test.")
    parser.add_argument("--steps", type=int, default=12, help="Synthetic training steps to run.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="How often the perf monitor logs metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory used to save performance monitor logs.",
    )
    parser.add_argument("--matmul-size", type=int, default=4096, help="Synthetic workload size.")
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=8.0,
        help="Extra per-step CPU sleep to make timings visible.",
    )
    parser.add_argument("--seq-length", type=int, default=512, help="Model seq length hint.")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Model hidden size hint.")
    parser.add_argument("--num-layers", type=int, default=8, help="Model layer count hint.")
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=16,
        help="Model attention head hint.",
    )
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size hint.")
    parser.add_argument(
        "--perf-breakdown",
        action="store_true",
        help="Emit estimated FLOPS breakdown to the text log.",
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


def build_perf_args(cli_args, world_size):
    return SimpleNamespace(
        enable_perf_monitor=True,
        perf_log_interval=cli_args.log_interval,
        perf_log_dir=str(Path(cli_args.output_dir).expanduser().resolve()),
        perf_console_output=True,
        perf_log_format="both",
        perf_memory_tracking=True,
        perf_breakdown=cli_args.perf_breakdown,
        perf_max_log_files=10,
        perf_model_type="gpt",
        world_size=world_size,
        seq_length=cli_args.seq_length,
        hidden_size=cli_args.hidden_size,
        num_layers=cli_args.num_layers,
        num_attention_heads=cli_args.num_attention_heads,
        ffn_hidden_size=4 * cli_args.hidden_size,
        padded_vocab_size=50257,
        micro_batch_size=cli_args.micro_batch_size,
        num_micro_batches=1,
        swiglu=False,
    )


def allocate_work_tensors(size, use_cuda, local_rank):
    if not use_cuda:
        return None, None
    device = torch.device("cuda", local_rank)
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    return a, b


def run_step(a, b, sleep_ms, use_cuda):
    if use_cuda:
        _ = a @ b
        _ = a @ b
        torch.cuda.synchronize()
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)


def main():
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank, use_cuda = init_dist()
    perf_args = build_perf_args(cli_args, world_size)
    perf_callback = initialize_perf_monitor(perf_args)
    a, b = allocate_work_tensors(cli_args.matmul_size, use_cuda, local_rank)

    if rank == 0:
        print(
            f"[rank0] Starting perf monitor smoke test: world_size={world_size}, "
            f"log_interval={cli_args.log_interval}, output_dir={output_dir}"
        )

    try:
        for iteration in range(1, cli_args.steps + 1):
            if perf_callback is not None:
                perf_monitor_start_iteration(iteration)
            run_step(a, b, cli_args.sleep_ms, use_cuda)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if perf_callback is not None:
                perf_monitor_end_iteration(iteration)
    finally:
        perf_monitor_end_training()
        cleanup_dist()


if __name__ == "__main__":
    main()
