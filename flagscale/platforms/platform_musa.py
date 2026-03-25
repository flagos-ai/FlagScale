import torch

from .platform_base import PlatformBase


class PlatformMUSA(PlatformBase):
    def name(self) -> str:
        return "musa"

    def is_available(self) -> bool:
        try:
            import torch_musa  # noqa: F401

            return torch.musa.is_available() and torch.musa.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        import torch_musa  # noqa: F401

        torch.musa.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("musa", device_index)

    def device_count(self) -> int:
        import torch_musa  # noqa: F401

        return torch.musa.device_count()

    def dist_backend(self) -> str:
        return "mccl"

    def manual_seed_all(self, seed):
        import torch_musa  # noqa: F401

        torch.musa.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "musa"

    def supports_distributions_on_device(self) -> bool:
        return False
