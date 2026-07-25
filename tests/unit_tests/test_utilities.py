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

import os
from datetime import timedelta

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

import megatron.core.parallel_state as ps

PLATFORM_RUNTIME_MAP = {
    "cuda": {
        "dist_backend": "nccl",
        "torch_device_type": "cuda",
        "torch_accelerator_attr": "cuda",
        "optional_runtime_module": None,
    },
    "ascend": {
        "dist_backend": "hccl",
        "torch_device_type": "npu",
        "torch_accelerator_attr": "npu",
        "optional_runtime_module": "torch_npu",
    },
    "metax": {
        "dist_backend": "nccl",
        "torch_device_type": "cuda",
        "torch_accelerator_attr": "cuda",
        "optional_runtime_module": None,
    },
}

DEFAULT_PLATFORM_RUNTIME = PLATFORM_RUNTIME_MAP["cuda"]


class TestModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        bias: bool,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )
        if shared_embedding:
            self.layers[-1].weight.shared_embedding = True


class Utils:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    inited = False
    store = None

    @staticmethod
    def platform():
        return os.getenv("FLAGSCALE_TEST_PLATFORM", "cuda")

    @staticmethod
    def platform_runtime():
        return PLATFORM_RUNTIME_MAP.get(Utils.platform(), DEFAULT_PLATFORM_RUNTIME)

    @staticmethod
    def dist_backend():
        return os.getenv(
            "FLAGSCALE_TEST_DIST_BACKEND",
            Utils.platform_runtime()["dist_backend"],
        )

    @staticmethod
    def torch_device_type():
        return os.getenv(
            "FLAGSCALE_TEST_TORCH_DEVICE_TYPE",
            Utils.platform_runtime()["torch_device_type"],
        )

    @staticmethod
    def accelerator():
        runtime = Utils.platform_runtime()
        optional_runtime_module = runtime.get("optional_runtime_module")
        if optional_runtime_module:
            try:
                __import__(optional_runtime_module)
            except Exception:
                pass

        accelerator_attr = os.getenv(
            "FLAGSCALE_TEST_TORCH_ACCELERATOR_ATTR",
            runtime["torch_accelerator_attr"],
        )
        return getattr(torch, accelerator_attr, torch.cuda)

    @staticmethod
    def accelerator_device_count():
        accelerator = Utils.accelerator()
        device_count = getattr(accelerator, "device_count", None)
        if callable(device_count):
            count = device_count()
            return count if count > 0 else 1
        return 1

    @staticmethod
    def has_accelerator():
        accelerator = Utils.accelerator()
        device_count = getattr(accelerator, "device_count", None)
        return callable(device_count) and device_count() > 0

    @staticmethod
    def accelerator_device(rank=None):
        rank = Utils.rank if rank is None else rank
        index = rank % Utils.accelerator_device_count()
        return torch.device(f"{Utils.torch_device_type()}:{index}")

    @staticmethod
    def initialize_distributed():
        os.environ.pop("NVTE_FLASH_ATTN", None)
        os.environ.pop("NVTE_FUSED_ATTN", None)
        os.environ.pop("NVTE_UNFUSED_ATTN", None)

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f"Initializing torch.distributed with rank: {Utils.rank}, "
                f"world_size: {Utils.world_size}"
            )
            accelerator = Utils.accelerator()
            set_device = getattr(accelerator, "set_device", None)
            if callable(set_device):
                set_device(Utils.rank % Utils.accelerator_device_count())
            init_method = "tcp://"
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method += master_ip + ":" + master_port
            rendezvous_iterator = rendezvous(
                init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1)
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=1))

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                backend=Utils.dist_backend(),
                world_size=Utils.world_size,
                rank=Utils.rank,
                store=store,
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = Utils.accelerator_device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ["LOCAL_RANK"])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        os.environ.pop("NVTE_FLASH_ATTN", None)
        os.environ.pop("NVTE_FUSED_ATTN", None)
        os.environ.pop("NVTE_UNFUSED_ATTN", None)
        if not Utils.inited:
            return
        torch.distributed.barrier()
        ps.destroy_model_parallel()
        Utils.inited = False

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        # Need to unset these variables to make sure previous
        # tests setting them doesn't interfere current test.
        os.environ.pop("NVTE_FLASH_ATTN", None)
        os.environ.pop("NVTE_FUSED_ATTN", None)
        os.environ.pop("NVTE_UNFUSED_ATTN", None)

        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        Utils.inited = True

    @staticmethod
    def fake_initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
    ):
        """Used for layer-wise UT as a proxy for NeMo-style initialization."""
        ps.set_tensor_model_parallel_world_size(tensor_model_parallel_size)
        ps.set_tensor_model_parallel_rank(0)

        ps.set_expert_model_parallel_world_size(expert_model_parallel_size)
        ps.set_expert_model_parallel_rank(0)
        if virtual_pipeline_model_parallel_size is not None:
            ps.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
        ps.set_virtual_pipeline_model_parallel_rank(0)

        ps.set_pipeline_model_parallel_world_size(pipeline_model_parallel_size)
