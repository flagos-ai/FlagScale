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

    # test case 1: loss is not spiky
    losses = [{"lm loss": 10.23}, {"lm loss": 10.32}, {"lm loss": 10.30}]
    reduced_loss = detector.reduce_losses(losses)
    is_spiky_loss = detector.is_spkiy_loss(reduced_loss)
    is_spiky_loss_tensor = torch.tensor(is_spiky_loss, dtype=torch.int, device="cuda")
    torch.distributed.all_reduce(is_spiky_loss_tensor, op=torch.distributed.ReduceOp.MAX)
    is_spiky_loss = is_spiky_loss_tensor.item()
    assert is_spiky_loss == 0, f"Expected 0, got {is_spiky_loss}"

    # test case 2: loss is spiky
    losses = [{"lm loss": 14.23}, {"lm loss": 14.32}, {"lm loss": 14.30}]
    reduced_loss = detector.reduce_losses(losses)
    is_spiky_loss = detector.is_spkiy_loss(reduced_loss)
    is_spiky_loss_tensor = torch.tensor(is_spiky_loss, dtype=torch.int, device="cuda")
    torch.distributed.all_reduce(is_spiky_loss_tensor, op=torch.distributed.ReduceOp.MAX)
    is_spiky_loss = is_spiky_loss_tensor.item()
    assert is_spiky_loss == 1, f"Expected 1, got {is_spiky_loss}"

    Utils.destroy_model_parallel()
