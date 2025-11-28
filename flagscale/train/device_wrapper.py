import functools
import logging

from typing import Any, Callable

import torch

logging.basicConfig(level=logging.INFO, format='[CUDA-LOG] %(message)s')
logger = logging.getLogger(__name__)


CUDA_TO_NEW_MAPPING = {
    # Memory
    'memory_allocated': 'memory_allocated',
    'max_memory_allocated': 'max_memory_allocated',
    'empty_cache': 'empty_cache',
    'reset_peak_memory_stats': 'reset_peak_memory_stats',
    'reset_max_memory_allocated': 'reset_max_memory_allocated',
    # Device
    'device_count': 'device_count',
    'current_device': 'current_device',
    'get_device_name': 'get_device_name',
    'get_device_properties': 'get_device_properties',
    # Stream
    'Stream': 'Stream',
    'current_stream': 'current_stream',
    'default_stream': 'default_stream',
    'stream': 'stream',
    'synchronize': 'synchronize',
    # Event
    'Event': 'Event',
    # AMP
    'amp': 'amp',
    # Random
    'manual_seed': 'manual_seed',
    'manual_seed_all': 'manual_seed_all',
}


def create_cuda_to_new_function(new_device, name: str) -> Any:
    if name in CUDA_TO_NEW_MAPPING:
        new_attr_name = CUDA_TO_NEW_MAPPING[name]
    else:
        new_attr_name = name

    if hasattr(new_device, new_attr_name):
        return getattr(new_device, new_attr_name)

    return None


class NewDevice:
    def __init__(self, _original_cuda, _new_device):
        self._original_cuda = _original_cuda
        self._new_device = _new_device
        self._patched_attrs = {}

    def __getattr__(self, name: str) -> Any:
        if name in self._patched_attrs:
            return self._patched_attrs[name]
        new_attr = create_cuda_to_new_function(self._new_device, name)
        if new_attr is not None:
            print(f"[AUTO] Redirecting torch.cuda.{name} -> torch.new.{name}")
            self._patched_attrs[name] = new_attr
            return new_attr
        else:
            print(f"[FALLBACK] torch.new.{name} not in ,using original torch.cuda.{name}")
            orig_attr = getattr(self._original_cuda, name)
            self._patched_attrs[name] = orig_attr
            return orig_attr


def patch_cuda_to_new_device():
    if not hasattr(torch, 'cuda'):
        raise RuntimeError("torch.cuda not found")
    _original_cuda = torch.cuda
    _new_device = torch.cuda
    torch.cuda = NewDevice(_original_cuda, _new_device)
    return True
