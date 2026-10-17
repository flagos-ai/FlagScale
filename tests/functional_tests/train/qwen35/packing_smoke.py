# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run real Qwen3.5 packed training with TP broadcast and optimizer assertions.

Use examples/qwen35/conf/train_packing_smoke.yaml with a prepared image-text
Energon dataset and the platform's training runtime. This is an opt-in device
test, not a CPU pytest module. The normal model, loss, backward and optimizer
implementations are reused; hooks only check their inputs, outputs and results.
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch

# Match the import path of directly executing train_qwen35.py (gpt_builders).
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "flagscale/train/megatron"))

import megatron.training.training as training_runtime
from megatron.core.enums import ModelType
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.training.utils import unwrap_model

from flagscale.train.megatron import train_qwen35 as training


def check_tp_broadcast(model):
    """Exercise actual packed/unpacked broadcasts with data=None on TP peers."""
    for packed in (False, True):
        iterator = None
        if get_tensor_model_parallel_rank() == 0:
            payload = {
                "text": torch.tensor([[11, 12, 21, 22]], dtype=torch.int64),
                "target": torch.tensor([[12, 21, 22, -100]], dtype=torch.int64),
                "imgs": torch.empty((0, 12), dtype=torch.float32),
                "videos": torch.empty((0, 12), dtype=torch.float32),
                "image_thw_grids": torch.empty((0, 3), dtype=torch.int64),
                "video_thw_grids": torch.empty((0, 3), dtype=torch.int64),
                "image_input_mask": torch.zeros((1, 4), dtype=torch.bool),
                "video_input_mask": torch.zeros((1, 4), dtype=torch.bool),
                "second_per_grid_ts": torch.empty(0, dtype=torch.float32),
            }
            if packed:
                payload["cu_seqlens"] = torch.tensor([0, 2, 4], dtype=torch.int32)
                payload["target"][0, 1] = -100
            iterator = iter([payload])
        batch = training.get_batch(iterator, model=model)
        assert batch[0].tolist() == [[11, 12, 21, 22]]
        assert batch[1].tolist() == [[12, -100 if packed else 21, 22, -100]]
        expected_positions = [0, 1, 0, 1] if packed else [0, 1, 2, 3]
        assert batch[4].tolist() == [[expected_positions]] * 3
        if packed:
            assert batch[-1].tolist() == [0, 2, 4]
            assert batch[-1].dtype == torch.int32
        else:
            assert batch[-1] is None
    print(
        "PACKING_TP_BROADCAST_OK "
        + json.dumps(
            {"rank": torch.distributed.get_rank(), "tp_rank": get_tensor_model_parallel_rank()}
        ),
        flush=True,
    )


def main():
    """Train the complete model and fail on missing packing or invalid updates."""
    args = training.parse_and_validate_args(
        extra_args_provider=training.add_qwen35_extra_args,
        args_defaults={"tokenizer_type": "Qwen2VLTokenizer"},
    )
    training.validate_packing_args(args)
    if args.packing_buffer_size <= 0 or not args.enable_vision or (args.mtp_num_layers or 0) < 1:
        raise ValueError("This smoke requires multimodal packing and at least one MTP layer")
    if args.train_iters < 1 or args.clip_grad <= 0 or not args.check_for_nan_in_loss_and_grad:
        raise ValueError("Enable training, gradient-norm computation and NaN/Inf checks")
    if args.eval_iters != 0:
        raise ValueError("Use eval_iters=0; the smoke probes unpacked TP broadcasts separately")
    # Keep this functional test independent of existing checkpoint directories.
    args.load = None
    args.save = None
    full_config = training.pretrain_cfg_container_from_args(args)
    observations = {"forwards": 0, "multi_sample_forwards": 0, "optimizer_steps": 0}

    def observe_input(module, inputs, kwargs):
        params = kwargs.get("packed_seq_params")
        assert params is not None and params.qkv_format == "thd"
        assert params.cu_seqlens_q_padded is None and params.cu_seqlens_kv_padded is None
        cu = params.cu_seqlens_q
        tokens = kwargs["input_ids"]
        assert cu.dtype == torch.int32 and cu[0].item() == 0
        assert cu[-1].item() == tokens.numel() and (cu[1:] > cu[:-1]).all().item()
        assert torch.equal(cu, params.cu_seqlens_kv)
        assert kwargs["loss_mask"][0, cu[1:].long() - 1].eq(0).all().item()
        observations["rank"] = torch.distributed.get_rank()
        observations["forwards"] += 1
        observations["multi_sample_forwards"] += int(cu.numel() > 2)
        if observations["forwards"] == 1:
            print(
                "PACKING_BATCH "
                + json.dumps({"rank": observations["rank"], "cu_seqlens": cu.tolist()}),
                flush=True,
            )

    def observe_output(module, inputs, output):
        assert torch.isfinite(output).all().item(), "Nonfinite model loss output"

    def model_provider(*positional, **kwargs):
        model = training.model_provider(*positional, **kwargs)
        model.register_forward_pre_hook(observe_input, with_kwargs=True)
        model.register_forward_hook(observe_output)
        return model

    probed = False

    def forward_step(iterator, model):
        nonlocal probed
        if not probed:
            check_tp_broadcast(unwrap_model(model))
            probed = True
        return training.forward_step(iterator, model)

    original_optimizer_factory = training_runtime.get_megatron_optimizer

    def checked_optimizer_factory(*positional, **kwargs):
        optimizer = original_optimizer_factory(*positional, **kwargs)
        original_step = optimizer.step

        def checked_step():
            result = original_step()
            updated, grad_norm, _ = result
            assert updated, "Optimizer skipped a training update"
            assert grad_norm is not None and math.isfinite(float(grad_norm)), (
                "Nonfinite gradient norm"
            )
            observations["optimizer_steps"] += 1
            observations["last_grad_norm"] = float(grad_norm)
            return result

        optimizer.step = checked_step
        return optimizer

    training.train_valid_test_dataloaders_provider.is_distributed = True
    with patch.object(training_runtime, "get_megatron_optimizer", checked_optimizer_factory):
        training.pretrain(
            full_config,
            training.train_valid_test_dataloaders_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            process_non_loss_data_func=training.write_online_eval_to_tensorboard,
            non_loss_data_func=training.run_online_eval,
            get_embedding_ranks=training.get_embedding_ranks,
        )
    assert observations["optimizer_steps"] == args.train_iters
    assert observations["multi_sample_forwards"] > 0, (
        "Use a dataset with samples short enough to pack"
    )
    print("PACKING_SMOKE_COMPLETE " + json.dumps(observations), flush=True)


if __name__ == "__main__":
    main()
