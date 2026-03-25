import torch

from .platform_base import PlatformBase


class PlatformNPU(PlatformBase):
    def name(self) -> str:
        return "npu"

    def is_available(self) -> bool:
        try:
            import torch_npu  # noqa: F401

            return torch.npu.is_available() and torch.npu.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("npu", device_index)

    def device_count(self) -> int:
        import torch_npu  # noqa: F401

        return torch.npu.device_count()

    def dist_backend(self) -> str:
        return "hccl"

    def manual_seed_all(self, seed):
        import torch_npu  # noqa: F401

        torch.npu.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "npu"
