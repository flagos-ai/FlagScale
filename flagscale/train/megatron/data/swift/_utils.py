# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""The handful of ``swift.utils`` helpers the vendored template needs.

``swift.utils`` is a 15-module grab bag (distributed helpers, hub utilities, env
probing, memory accounting). The template imports exactly seven names from it, so
they are reproduced here instead of vendoring the package.

Vendored from ms-swift (Apache-2.0): ``swift/utils/{logger,utils,torch_utils,
processor_utils}.py``.
"""

import hashlib
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping, Optional, Union

import torch
import torch.distributed as dist
from transformers import (
    FeatureExtractionMixin,
    PreTrainedTokenizerBase,
    ProcessorMixin as HfProcessorMixin,
)

try:
    from transformers import BaseImageProcessor

    Processor = Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, HfProcessorMixin]
except ImportError:
    Processor = Union[PreTrainedTokenizerBase, FeatureExtractionMixin, HfProcessorMixin]

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

_LOGGER_NAME = 'flagscale.sft'
_info_once_seen = set()
_warning_once_seen = set()


class _Logger(logging.Logger):
    """A logger with the ``*_once`` helpers ms-swift's template calls.

    ``info_once`` in particular is load-bearing for readable startup logs: the
    template logs its resolved settings on every ``encode``, and without the
    de-duplication a packed run would emit them once per sample.
    """

    def info_once(self, msg, *args, **kwargs):
        hash_id = kwargs.pop('hash_id', None) or msg
        if hash_id in _info_once_seen:
            return
        _info_once_seen.add(hash_id)
        self.info(msg, *args, **kwargs)

    def warning_once(self, msg, *args, **kwargs):
        hash_id = kwargs.pop('hash_id', None) or msg
        if hash_id in _warning_once_seen:
            return
        _warning_once_seen.add(hash_id)
        self.warning(msg, *args, **kwargs)

    def warning_if(self, msg, cond, *args, **kwargs):
        if cond:
            self.warning(msg, *args, **kwargs)


def get_logger(log_file: Optional[str] = None, log_level: Optional[int] = None, file_mode: str = 'w'):
    """Return the shared SFT logger (signature kept compatible with ms-swift)."""
    original = logging.getLoggerClass()
    logging.setLoggerClass(_Logger)
    try:
        logger = logging.getLogger(_LOGGER_NAME)
    finally:
        logging.setLoggerClass(original)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('[%(levelname)s:flagscale-sft] %(message)s'))
        logger.addHandler(handler)
        logger.propagate = False
    if log_level is None:
        log_level = logging.INFO
    logger.setLevel(log_level)
    return logger


logger = get_logger()

_safe_ddp_synchronization_enabled = ContextVar('safe_ddp_synchronization_enabled', default=True)


def _strtobool(value: str) -> bool:
    value = value.lower()
    if value in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if value in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError(f'invalid truth value {value!r}')


def get_env_args(args_name: str, type_func, default_value):
    """Read a template hyperparameter from an UPPER_CASE environment variable.

    This is how ``REASONING_EFFORT`` reaches the DeepSeek-V4 template.
    """
    args_name_upper = args_name.upper()
    value = os.getenv(args_name_upper)
    if value is None:
        value = default_value
        log_info = (f'Setting {args_name}: {default_value}. '
                    f'You can adjust this hyperparameter through the environment variable: `{args_name_upper}`.')
    else:
        if type_func is bool:
            value = _strtobool(value)
        value = type_func(value)
        log_info = f'Using environment variable `{args_name_upper}`, Setting {args_name}: {value}.'
    logger.info_once(log_info)
    return value


def remove_response(messages) -> Optional[str]:
    """Pop and return the last message when it is an assistant turn."""
    last_role = messages[-1]['role'] if messages else None
    if last_role == 'assistant':
        return messages.pop()['content']


def retry_decorator(retry=3):

    def _retry(func):

        @wraps(func)
        def new_func(*args, **kwargs):
            i = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == retry:
                        raise
                    i += 1

        return new_func

    return _retry


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device."""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data


def get_dist_setting():
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_master() -> bool:
    return get_dist_setting()[0] in {-1, 0}


def is_local_master() -> bool:
    return get_dist_setting()[1] in {-1, 0}


def is_dist() -> bool:
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


@contextmanager
def safe_ddp_context(hash_id: Optional[str], use_barrier: bool = True):
    """Serialize a one-time side effect (dataset cache build) across ranks.

    Non-master ranks wait on a barrier so only one rank materializes the cache;
    the second barrier pair covers multi-node runs where each node has its own
    storage. Falls back to a file lock when torch.distributed is not initialized,
    which is the case while the dataset is being built.
    """
    if not _safe_ddp_synchronization_enabled.get():
        yield
    elif use_barrier and dist.is_available() and dist.is_initialized():
        if is_dist():
            if not is_master():
                dist.barrier()
            if not is_local_master():
                dist.barrier()
        yield
        if is_dist():
            if is_master():
                dist.barrier()
            if is_local_master():
                dist.barrier()
    elif hash_id is not None:
        from filelock import FileLock

        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        os.makedirs(lock_dir, exist_ok=True)
        file_path = hashlib.sha256(hash_id.encode('utf-8')).hexdigest() + '.lock'
        with FileLock(os.path.join(lock_dir, file_path)):
            yield
    else:
        yield


@contextmanager
def disable_safe_ddp_synchronization():
    """Disable nested dataset barriers when an outer rank-0 cache build owns synchronization."""
    token = _safe_ddp_synchronization_enabled.set(False)
    try:
        yield
    finally:
        _safe_ddp_synchronization_enabled.reset(token)


def get_seed(random_state=None) -> int:
    import numpy as np

    if random_state is None:
        random_state = np.random.RandomState()
    seed_max = np.iinfo(np.int32).max
    return random_state.randint(0, seed_max)


def split_list(ori_list, num_shards: int, contiguous=True):
    import numpy as np

    shard = []
    if contiguous:
        idx_list = np.linspace(0, len(ori_list), num_shards + 1, dtype=np.int64)
        for i in range(len(idx_list) - 1):
            shard.append(ori_list[idx_list[i]:idx_list[i + 1]])
    else:
        ori_list = np.array(ori_list)
        for i in range(num_shards):
            shard.append(ori_list[np.arange(i, len(ori_list), num_shards)].tolist())
    return shard


def use_hf_hub() -> bool:
    """Whether to resolve dataset ids against HuggingFace rather than ModelScope.

    The vendored loader only reads local files, so this just preserves the flag's
    meaning for the ``#N`` syntax parser.
    """
    return os.getenv('USE_HF', 'false').lower() in {'1', 'true', 'yes'}


def get_cache_dir(model_id: Optional[str] = None) -> str:
    """Cache root, matching ``modelscope.hub.utils.utils.get_cache_dir``.

    Reproduced so the template does not import modelscope for this one call. The
    only caller is ``_save_pil_image`` (multimodal), which this path never
    reaches, but the precedence is kept identical in case it ever does:
    ``MODELSCOPE_CACHE`` then ``~/.cache/modelscope/hub``.
    """
    default_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'modelscope')
    base_path = os.getenv('MODELSCOPE_CACHE', os.path.join(default_cache_dir, 'hub'))
    return base_path if model_id is None else os.path.join(base_path, model_id + '/')


class RowPreprocessor:
    """Only the one static helper the vendored collator calls.

    Vendored from ms-swift ``swift/dataset/preprocessor/core.py``; the full class
    is the jsonl/HF-dataset preprocessing pipeline, which the data path reaches
    through ``swift.dataset`` instead (see Phase 2d).
    """

    @staticmethod
    def rows_to_batched(rows):
        batched = {}
        for i, row in enumerate(rows):
            for k, v in row.items():
                if k not in batched:
                    batched[k] = [None] * i
                batched[k].append(v)
            # Make all the lengths of v the same.
            for k in set(batched.keys()) - set(row.keys()):
                batched[k].append(None)
        return batched


class ProcessorMixin:

    @property
    def tokenizer(self):
        tokenizer = self.processor
        if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        if self.processor is self.tokenizer:
            self.processor = value
        elif self.tokenizer is not value:
            raise AttributeError('Please use `self.processor` for assignment.')
