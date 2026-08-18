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

"""Small torchrun workload for exercising GPU-progress heartbeat semantics."""

from __future__ import annotations

import argparse
import os
import time

from flagscale.train import gpu_heartbeat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("normal", "stall"), required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    gpu_heartbeat.initialize_from_env()
    gpu_heartbeat.set_phase("train")
    for iteration in range(25):
        # In the stall scenario rank 1 remains process-alive but its training
        # progress hook stops advancing after two completed iterations.
        if args.scenario == "normal" or rank == 0 or iteration < 2:
            gpu_heartbeat.mark_progress("train", iteration=iteration + 1)
        time.sleep(0.1)
    gpu_heartbeat.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
