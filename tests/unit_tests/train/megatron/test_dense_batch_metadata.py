# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_batch_function(args, rank, stage, broadcast):
    """Execute the native batch function with CPU devices and a TP transport stub."""
    path = Path(__file__).resolve().parents[4] / (
        "flagscale/train/megatron/training/utils/common_utils.py"
    )
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "get_batch_on_this_tp_rank"
    )
    namespace = {
        "torch": torch,
        "get_args": lambda: args,
        "cur_platform": SimpleNamespace(device=lambda: "cpu"),
        "mpu": SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: rank,
            get_tensor_model_parallel_src_rank=lambda: 0,
            get_tensor_model_parallel_group=lambda: None,
            is_pipeline_first_stage=lambda: stage == "first",
            is_pipeline_last_stage=lambda: stage == "last",
            get_dualpipev_pipeline_model_parallel_world_size=lambda: None,
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_batch_on_this_tp_rank"]


def batch_args(stage, mode="dense"):
    return SimpleNamespace(
        hybrid_context_parallel=mode == "hybrid",
        sft=mode == "sft",
        pipeline_model_parallel_size=1 if stage == "single" else 2,
        micro_batch_size=1,
        seq_length=4,
        create_attention_mask_in_dataloader=False,
    )


def batch_data(mode="dense"):
    shape = (4,) if mode == "hybrid" else (1, 4)
    data = {
        "tokens": torch.arange(4).reshape(shape),
        "labels": torch.arange(4).reshape(shape),
        "loss_mask": torch.ones(shape),
        "position_ids": torch.arange(4).reshape(shape),
    }
    if mode != "dense":
        data.update(
            cu_seqlens=torch.tensor([[0, 2, 4]], dtype=torch.int32),
            max_seqlen=torch.tensor([2], dtype=torch.int32),
        )
    if mode == "hybrid":
        data["local_cp_size"] = torch.tensor([2], dtype=torch.int32)
    return data


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "stage,mtp,count",
    [("single", False, 4), ("first", False, 2), ("last", False, 2), ("last", True, 4)],
)
def test_dense_batch_does_not_broadcast_packed_metadata(monkeypatch, rank, stage, mtp, count):
    broadcasts = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, *a, **kw: broadcasts.append(tensor)
    )
    monkeypatch.setattr(
        torch.Tensor, "item", lambda *a: pytest.fail("dense batch must not read a device scalar")
    )
    function = load_batch_function(batch_args(stage), rank, stage, broadcasts)
    batch = function(iter([batch_data()]) if rank == 0 else None, mtp_on_this_rank=mtp)
    assert len(broadcasts) == count
    assert batch["cu_seqlens"] is None
    assert batch["max_seqlen"] is None


@pytest.mark.parametrize("mode", ["sft", "hybrid"])
@pytest.mark.parametrize("stage", ["single", "first"])
def test_packed_batch_keeps_matching_source_and_receiver_protocol(monkeypatch, mode, stage):
    messages = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, *a, **kw: messages.append(tensor.clone())
    )
    source = load_batch_function(batch_args(stage, mode), 0, stage, messages)
    source_batch = source(iter([batch_data(mode)]))
    received = iter(messages)

    def receive(tensor, *args, **kwargs):
        tensor.copy_(next(received))

    monkeypatch.setattr(torch.distributed, "broadcast", receive)
    receiver = load_batch_function(batch_args(stage, mode), 1, stage, receive)
    receiver_batch = receiver(None)
    assert next(received, None) is None
    assert torch.equal(receiver_batch["cu_seqlens"], source_batch["cu_seqlens"])
    assert torch.equal(receiver_batch["max_seqlen"], source_batch["max_seqlen"])
