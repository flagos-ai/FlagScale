import torch

from .platform_base import PlatformBase


class PlatformCUDA(PlatformBase):
    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("cuda", device_index)

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def dist_backend(self) -> str:
        return "nccl"

    def manual_seed_all(self, seed):
        torch.cuda.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "cuda"
