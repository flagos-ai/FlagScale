import torch

from .platform_base import PlatformBase


class PlatformTXDA(PlatformBase):
    def __init__(self):
        self._name = "txda"
        try:
            import flag_gems  # noqa: F401
            from torch_txda import transfer_to_txda  # noqa: F401
        except ImportError:
            pass

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        try:
            import torch_txda  # noqa: F401

            return torch.txda.is_available() and torch.txda.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("txda", device_index)

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def dist_backend(self) -> str:
        return "flagcx"

    def manual_seed_all(self, seed):
        torch.cuda.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "cuda"

