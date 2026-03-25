import abc
from abc import ABC


class PlatformBase(ABC):
    """FlagScale hardware platform abstraction base class.

    Provides a unified interface for device operations across different
    hardware backends (CUDA, NPU, MUSA, etc.).
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Return the platform name, e.g. 'cuda', 'npu', 'musa'."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this platform is available on the current machine."""
        ...

    # ---- Device APIs ----

    @abc.abstractmethod
    def set_device(self, device_index):
        """Set the current device."""
        ...

    @abc.abstractmethod
    def device(self, device_index=None):
        """Return a torch.device for the given index."""
        ...

    @abc.abstractmethod
    def device_count(self) -> int:
        """Return the number of available devices."""
        ...

    # ---- Distributed ----

    @abc.abstractmethod
    def dist_backend(self) -> str:
        """Return the distributed backend name, e.g. 'nccl', 'hccl', 'mccl'."""
        ...

    # ---- RNG ----

    @abc.abstractmethod
    def manual_seed_all(self, seed):
        """Set the seed for all devices."""
        ...

    # ---- AMP ----

    @abc.abstractmethod
    def amp_device_type(self) -> str:
        """Return the device type string for torch.amp.autocast, e.g. 'cuda', 'npu'."""
        ...

    # ---- Compatibility ----

    def supports_distributions_on_device(self) -> bool:
        """Whether torch.distributions (e.g. Beta) can run directly on this device.

        Returns False for platforms that require CPU fallback (e.g. MUSA).
        Defaults to True.
        """
        return True
