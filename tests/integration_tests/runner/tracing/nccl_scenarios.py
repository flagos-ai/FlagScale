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

"""Small two-rank NCCL workloads for exercising the preload probe on GPU hosts."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from flagscale.train import gpu_heartbeat


def _initialize() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
    return dist.get_rank(), local_rank


def run(scenario: str) -> None:
    rank, local_rank = _initialize()
    tensor = torch.ones(4, device=f"cuda:{local_rank}")

    # Align both workers and create one known-good collective before the fault.
    dist.all_reduce(tensor)

    if scenario == "sanity":
        expected = torch.full_like(tensor, 4.0)
        dist.all_reduce(tensor)
        if not torch.equal(tensor, expected):
            raise AssertionError(f"unexpected AllReduce result on rank {rank}: {tensor}")
        dist.destroy_process_group()
        return

    if scenario == "not_enter":
        if rank == 0:
            dist.all_reduce(tensor)
        else:
            time.sleep(30.0)
        return

    if scenario == "api_mismatch":
        if rank == 0:
            dist.all_reduce(tensor)
        else:
            dist.broadcast(tensor, src=0)
        return

    if scenario == "parameter_mismatch":
        mismatched = torch.ones(4 if rank == 0 else 8, device=f"cuda:{local_rank}")
        dist.all_reduce(mismatched)
        return

    if scenario == "p2p_missing":
        # Complete one matched pair first so ProcessGroupNCCL has initialized its
        # lazily-created two-rank P2P communicator on both workers.
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            time.sleep(30.0)
        return

    if scenario == "p2p_multi_target":
        if rank == 0:
            operations = [
                dist.P2POp(dist.isend, tensor, 1),
                dist.P2POp(dist.isend, tensor, 2),
            ]
        else:
            tensor.zero_()
            operations = [dist.P2POp(dist.irecv, tensor, 0)]
        for request in dist.batch_isend_irecv(operations):
            request.wait()
        expected = torch.full_like(tensor, float(dist.get_world_size()))
        if rank != 0 and not torch.equal(tensor, expected):
            raise AssertionError(f"unexpected P2P result on rank {rank}: {tensor}")
        dist.barrier()
        dist.destroy_process_group()
        return

    raise ValueError(f"unknown scenario: {scenario}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=(
            "sanity",
            "not_enter",
            "api_mismatch",
            "parameter_mismatch",
            "p2p_missing",
            "p2p_multi_target",
        ),
    )
    gpu_heartbeat.initialize_from_env()
    run(parser.parse_args().scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
