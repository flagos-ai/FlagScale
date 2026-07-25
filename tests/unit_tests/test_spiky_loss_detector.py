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

import torch

from megatron.training.spiky_loss import SpikyLossDetector
from tests.unit_tests.test_utilities import Utils


def test_spiky_loss_detector(pp_size=2, threshold=0.2):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        expert_tensor_parallel_size=1,
    )

    detector = SpikyLossDetector(threshold=threshold, loss=10.0)
    device = Utils.accelerator_device()

    # test case 1: loss is not spiky
    # losses should contain tensors with shape [2] (loss_value, num_tokens)
    losses = [
        {"lm loss": torch.tensor([10.23, 100.0], device=device)},
        {"lm loss": torch.tensor([10.32, 100.0], device=device)},
        {"lm loss": torch.tensor([10.30, 100.0], device=device)},
    ]
    reduced_loss = detector.reduce_losses(losses)
    is_spiky_loss = detector.is_spkiy_loss(reduced_loss)
    is_spiky_loss_tensor = torch.tensor(is_spiky_loss, dtype=torch.int, device=device)
    torch.distributed.all_reduce(is_spiky_loss_tensor, op=torch.distributed.ReduceOp.MAX)
    is_spiky_loss = is_spiky_loss_tensor.item()
    assert is_spiky_loss == 0, f"Expected 0, got {is_spiky_loss}"

    # test case 2: loss is spiky
    losses = [
        {"lm loss": torch.tensor([14.23, 100.0], device=device)},
        {"lm loss": torch.tensor([14.32, 100.0], device=device)},
        {"lm loss": torch.tensor([14.30, 100.0], device=device)},
    ]
    reduced_loss = detector.reduce_losses(losses)
    is_spiky_loss = detector.is_spkiy_loss(reduced_loss)
    is_spiky_loss_tensor = torch.tensor(is_spiky_loss, dtype=torch.int, device=device)
    torch.distributed.all_reduce(is_spiky_loss_tensor, op=torch.distributed.ReduceOp.MAX)
    is_spiky_loss = is_spiky_loss_tensor.item()
    assert is_spiky_loss == 1, f"Expected 1, got {is_spiky_loss}"

    Utils.destroy_model_parallel()
