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

"""Exercise the real batch functions and TP helper without initializing a GPU.

Only platform placement, process-group metadata and broadcast transport are
replaced. FIFO broadcast replay checks sender/receiver shape, dtype and order.
Extracting the functions avoids importing the full training/vision dependency tree.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "flagscale/train/megatron"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def make_args(**overrides):
    args = dict(
        enable_vision=False,
        position_embedding_type="mrope",
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        sft=False,
        hybrid_context_parallel=False,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        use_dualpipev=False,
        create_attention_mask_in_dataloader=False,
        micro_batch_size=2,
        seq_length=6,
    )
    return SimpleNamespace(**(args | overrides))


class BroadcastReplay:
    def __init__(self, state):
        self.state = state
        self.frames = {}
        self.offsets = {}

    def broadcast(self, tensor, src, group):
        assert src == group[0]
        rank = self.state.tp_rank
        if rank == 0:
            self.frames.setdefault(group, []).append(tensor.clone())
        else:
            key = (group, rank)
            index = self.offsets.get(key, 0)
            expected = self.frames[group][index]
            assert tensor.shape == expected.shape
            assert tensor.dtype == expected.dtype
            tensor.copy_(expected)
            self.offsets[key] = index + 1


def batch(dp_rank, microbatch, attention_mask):
    tokens = torch.arange(12, dtype=torch.int64).view(2, 6) + 100 * dp_rank + 20 * microbatch
    # Different document boundaries across DP streams, rows and microbatches.
    patterns = [
        [[0, 1, 0, 1, 2, 0], [0, 1, 2, 0, 1, 2]],
        [[0, 0, 1, 2, 3, 0], [0, 1, 0, 1, 0, 1]],
        [[0, 1, 2, 3, 0, 1], [0, 0, 1, 2, 0, 1]],
    ]
    positions = torch.tensor(patterns[(dp_rank + microbatch) % len(patterns)])
    return {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(2, 6),
        "attention_mask": torch.zeros(2, 1, 6, 6, dtype=torch.bool) if attention_mask else None,
        "position_ids": positions,
    }


@pytest.fixture
def harness(monkeypatch):
    spec = importlib.util.find_spec("megatron")
    if spec is None:
        pytest.skip("Megatron-LM-FL must be installed or present on PYTHONPATH")
    utils = next(
        (
            Path(root) / "core/utils.py"
            for root in spec.submodule_search_locations
            if (Path(root) / "core/utils.py").exists()
        ),
        None,
    )
    assert utils is not None, "Cannot locate the installed Megatron TP batch helper"
    args = make_args()
    state = SimpleNamespace(pp_rank=0, tp_rank=0, dp_rank=0)
    transport = BroadcastReplay(state)
    monkeypatch.setattr(torch.distributed, "broadcast", transport.broadcast)
    first = lambda: state.pp_rank == 0
    last = lambda: state.pp_rank == args.pipeline_model_parallel_size - 1
    mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: state.tp_rank,
        get_tensor_model_parallel_src_rank=lambda: state.dp_rank * 100 + state.pp_rank * 10,
        get_tensor_model_parallel_group=lambda: (
            state.dp_rank * 100 + state.pp_rank * 10,
            state.dp_rank,
            state.pp_rank,
        ),
        is_pipeline_first_stage=first,
        is_pipeline_last_stage=last,
        get_dualpipev_pipeline_model_parallel_world_size=lambda: None,
    )
    ns = {
        "torch": torch,
        "get_args": lambda: args,
        "cur_platform": SimpleNamespace(device=lambda *a: "cpu", current_device=lambda: 0),
        "mpu": mpu,
        "core_transformer_config_from_args": lambda a: SimpleNamespace(
            pipeline_model_parallel_layout=None, mtp_num_layers=a.mtp_num_layers
        ),
        "mtp_on_this_rank_func": lambda **kw: bool(args.mtp_num_layers) and last(),
        "is_first_or_last_pipeline_stage": lambda *a: first() or last(),
        "get_context_parallel_group": lambda: None,
        "get_hybrid_data_context_parallel_groups": lambda: None,
        "get_batch_on_this_cp_rank": lambda b, **kw: b,
    }
    load_functions(utils, ["get_batch_on_this_tp_rank"], ns)
    load_functions(TRAIN / "train_gpt.py", ["get_batch", "is_dataset_built_on_rank"], ns)
    ns["get_batch_gpt"] = ns["get_batch"]
    load_functions(
        TRAIN / "train_qwen35.py", ["text_batch_needs_pipeline_positions", "get_batch_text"], ns
    )
    return args, state, transport, ns


@pytest.mark.parametrize("pp_size,tp_size", [(1, 2), (2, 2), (4, 2), (4, 4)])
@pytest.mark.parametrize("with_mask", [False, True])
def test_native_reset_positions_reach_every_stage_and_tp_rank(harness, pp_size, tp_size, with_mask):
    args, state, transport, ns = harness
    args.pipeline_model_parallel_size = pp_size
    args.create_attention_mask_in_dataloader = with_mask
    args.build_dataset_on_all_pipeline_stages = ns["text_batch_needs_pipeline_positions"](args)
    for dp_rank in range(2):
        state.dp_rank = dp_rank
        for pp_rank in range(pp_size):
            state.pp_rank = pp_rank
            for tp_rank in range(tp_size):
                state.tp_rank = tp_rank
                assert ns["is_dataset_built_on_rank"]() == (tp_rank == 0)
                iterator = (
                    iter([batch(dp_rank, i, with_mask) for i in range(3)]) if tp_rank == 0 else None
                )
                for i in range(3):
                    expected = batch(dp_rank, i, with_mask)
                    tokens, labels, loss_mask, mask, positions, packed = ns["get_batch_text"](
                        iterator
                    )
                    assert positions.shape == (3, 2, 6)
                    assert positions.is_contiguous()
                    for axis in range(3):
                        torch.testing.assert_close(positions[axis], expected["position_ids"])
                    assert (tokens is not None) == (pp_rank == 0)
                    assert (labels is not None) == (pp_rank == pp_size - 1)
                    assert (loss_mask is not None) == (pp_rank == pp_size - 1)
                    if tokens is not None:
                        torch.testing.assert_close(tokens, expected["tokens"])
                    if labels is not None:
                        torch.testing.assert_close(labels, expected["labels"])
                    assert (mask is not None) == with_mask
                    assert packed is None
                if tp_rank:
                    group = ns["mpu"].get_tensor_model_parallel_group()
                    assert transport.offsets[group, tp_rank] == len(transport.frames[group])
    # Complete-batch transport must not alter the real topology.
    assert args.pipeline_model_parallel_size == pp_size


def test_generic_gpt_default_keeps_existing_stage_selection(harness):
    args, state, _, ns = harness
    for stage in range(args.pipeline_model_parallel_size):
        state.pp_rank = stage
        for tp_rank in range(2):
            state.tp_rank = tp_rank
            iterator = iter([batch(0, 0, False)]) if tp_rank == 0 else None
            result = ns["get_batch"](iterator)
            assert len(result) == 10
            if stage in (1, 2):
                assert result == [None] * 10
                if iterator is not None:
                    assert next(iterator)["tokens"] is not None
            elif stage == 0:
                torch.testing.assert_close(result[8], batch(0, 0, False)["position_ids"])
            else:
                assert result[8] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"sft": True},
        {"hybrid_context_parallel": True},
        {"context_parallel_size": 2},
    ],
)
def test_retain_positions_rejects_unsupported_batch_layouts(harness, overrides):
    args, _, _, ns = harness
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match="dense batch with context_parallel_size=1"):
        ns["get_batch"](None, keep_position_ids=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_vision": True},
        {"position_embedding_type": "rope"},
        {"pipeline_model_parallel_size": 1},
        {"context_parallel_size": 2},
        {"sft": True},
        {"hybrid_context_parallel": True},
        {"mtp_num_layers": 1},
        {"virtual_pipeline_model_parallel_size": 2},
        {"use_dualpipev": True},
    ],
)
def test_other_modes_do_not_enable_replicated_text_batches(overrides):
    ns = {}
    load_functions(TRAIN / "train_qwen35.py", ["text_batch_needs_pipeline_positions"], ns)
    assert not ns["text_batch_needs_pipeline_positions"](make_args(**overrides))


@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize("vision", [False, True])
def test_entrypoint_selects_dataset_policy_before_pretrain(monkeypatch, pp_size, vision):
    args = make_args(pipeline_model_parallel_size=pp_size, enable_vision=vision)
    ns = {"__name__": "__main__"}
    load_functions(TRAIN / "train_qwen35.py", ["text_batch_needs_pipeline_positions"], ns)
    text_provider = lambda *a: None
    vision_provider = lambda *a: None
    seen = []

    def make_config(actual_args):
        assert actual_args is args
        assert getattr(args, "build_dataset_on_all_pipeline_stages", False) == (
            not vision and pp_size > 1
        )
        return object()

    def pretrain(config, provider, *a, **kw):
        seen.append(provider)
        assert getattr(args, "build_dataset_on_all_pipeline_stages", False) == (
            not vision and pp_size > 1
        )

    ns.update(
        parse_and_validate_args=lambda **kw: args,
        add_qwen35_extra_args=lambda a: a,
        pretrain_cfg_container_from_args=make_config,
        pretrain=pretrain,
        train_valid_test_dataloaders_provider=vision_provider,
        train_valid_test_datasets_provider_gpt=text_provider,
        model_provider=None,
        ModelType=SimpleNamespace(encoder_or_decoder=None),
        forward_step=None,
        forward_step_text=None,
        write_online_eval_to_tensorboard=None,
        run_online_eval=None,
        get_embedding_ranks=None,
    )
    argv = ["train_qwen35.py"] + ([] if vision else ["--no-enable-vision"])
    monkeypatch.setattr(sys, "argv", argv)
    tree = ast.parse((TRAIN / "train_qwen35.py").read_text())
    main = next(
        n
        for n in tree.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "__name__"
    )
    exec(compile(ast.Module(body=[main], type_ignores=[]), "train_qwen35.py", "exec"), ns)
    assert seen == [vision_provider if vision else text_provider]
