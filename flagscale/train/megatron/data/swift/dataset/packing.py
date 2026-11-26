# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Vendored from ms-swift (Apache-2.0): swift/dataset/packing.py.
# FlagScale additionally supports shared index caches and physical-length
# accounting for padding applied to each row before rows are packed.
import math
import multiprocessing as mp
import os
import pickle
import tempfile
from itertools import chain
from typing import Optional

import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from .._utils import get_logger, is_dist, is_master, split_list
from ..template import MaxLengthError

logger = get_logger()


def _resolve_mp_context(multiprocessing_context: Optional[str] = None):
    """Decide the multiprocessing context used by packing workers.

    When ``multiprocessing_context`` is unset, keep the platform default (``fork`` on Python <=3.13,
    ``forkserver`` on 3.14+), and only switch to ``spawn`` when fork is actually unsafe (CUDA or
    torch.distributed already initialized), which otherwise dead-locks the forked worker.
    """
    if multiprocessing_context is not None:
        return mp.get_context(multiprocessing_context)
    try:
        import torch
        cuda_initialized = torch.cuda.is_initialized()
    except Exception:
        cuda_initialized = False
    if cuda_initialized or (dist.is_available() and dist.is_initialized()):
        return mp.get_context('spawn')
    # keep the platform default (fork on Python <=3.13, forkserver on 3.14+); use the default context object.
    return mp.get_context()


def _spawn_workers(ctx, *, target, jobs):
    """Start packing workers and return them.

    ``jobs`` is a list of ``args`` tuples (one per worker); all workers share ``ctx`` and the caller's
    queues must be created from that same context. A failure to start a worker (e.g. an un-picklable
    arg under ``spawn``) propagates: we deliberately do NOT fall back to fork, because a forked child
    that inherited an active torch threadpool / CUDA state can dead-lock silently -- a loud error is
    strictly better than a hang.
    """
    workers = []
    for args in jobs:
        worker = ctx.Process(target=target, daemon=True, args=args)
        worker.start()
        workers.append(worker)
    return workers


def calculate_matched_group(sequences, packing_length: int, is_finished: bool = True, strategy: str = 'binpack'):
    if len(sequences) == 0:
        return [], []
    if strategy == 'sequential':
        # Order-preserving greedy packing (next-fit): keep a single open pack and flush it
        # when the next sample doesn't fit, so the global sample order and pack boundaries
        # follow the input order (a sequential sampler). (Use packing_num_proc=1 for
        # a single global ordering.)
        packs, cur, cur_len = [], [], 0
        for item in sequences:  # item = (idx, length); weight_pos=1 -> length at item[1]
            seq_len = item[1]
            if cur and cur_len + seq_len > packing_length:
                packs.append(cur)
                cur, cur_len = [], 0
            cur.append(item)
            cur_len += seq_len
            if cur_len >= packing_length:
                packs.append(cur)
                cur, cur_len = [], 0
        if is_finished:
            if cur:
                packs.append(cur)
            return packs, []
        return packs, cur
    # default: best-fit-decreasing bin packing (https://arxiv.org/pdf/2404.10830)
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


def _round_up_length(length: int, alignment: int) -> int:
    """Return the physical row length after pre-pack alignment."""
    if alignment <= 0:
        raise ValueError(f'packing_alignment must be positive, got {alignment}')
    return ((length + alignment - 1) // alignment) * alignment


class PackingDataset(Dataset):
    PACKING_BATCH_SIZE = 1000

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        strict: bool = False,
        load_from_cache_file: bool = True,
        packing_length: Optional[int] = None,
        packing_num_proc: int = 1,
        packing_strategy: str = 'binpack',
        packing_alignment: int = 1,
        packing_cache_file: Optional[str] = None,
        packing_cache_key: Optional[str] = None,
        multiprocessing_context: Optional[str] = None,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.load_from_cache_file = load_from_cache_file
        self.packing_strategy = packing_strategy
        self.packing_alignment = packing_alignment
        self.packing_cache_file = packing_cache_file
        self.packing_cache_key = packing_cache_key
        self.multiprocessing_context = multiprocessing_context
        self.packing_length = packing_length or self.template.max_length
        self.packing_num_proc = min(packing_num_proc, math.ceil(len(dataset) / self.PACKING_BATCH_SIZE))
        distributed = dist.is_available() and dist.is_initialized() and is_dist()
        rank = dist.get_rank() if distributed else 0
        use_disk_cache = bool(self.load_from_cache_file and self.packing_cache_file)

        if use_disk_cache:
            error = None
            if rank == 0:
                try:
                    from filelock import FileLock

                    lock_path = self.packing_cache_file + '.lock'
                    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
                    with FileLock(lock_path):
                        if not self._load_packing_cache():
                            self._build_packed_indices()
                            self._save_packing_cache()
                except Exception as exc:
                    error = f'Failed to build packing cache {self.packing_cache_file}: {exc!r}'
            if distributed:
                status = [error]
                dist.broadcast_object_list(status, src=0)
                error = status[0]
            if error is not None:
                raise RuntimeError(error)
            if rank != 0 and not self._load_packing_cache():
                raise RuntimeError(
                    f'Packing cache is not visible on rank {rank}: {self.packing_cache_file}. '
                    'Use a cache directory shared by every training node.')
            return

        if is_master():
            self._build_packed_indices()
        else:
            self.packed_idx, self.packed_length = None, None
        if distributed:
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]

    def _build_packed_indices(self):
        raw_lengths = self.dataset['lengths']
        lengths = []
        for length in raw_lengths:
            length = sum(length) if isinstance(length, (list, tuple)) else length
            physical_length = _round_up_length(int(length), self.packing_alignment)
            if physical_length > self.packing_length:
                raise ValueError(
                    f'Sample physical length {physical_length} exceeds packing_length '
                    f'{self.packing_length} after alignment to {self.packing_alignment} '
                    f'(raw length: {length}).'
                )
            lengths.append(physical_length)
        chunked_lengths = split_list(lengths, self.packing_num_proc)
        offsets = []
        offset = 0
        for chunk in chunked_lengths:
            offsets.append(offset)
            offset += len(chunk)
        jobs = [(i, offsets[i], chunked_lengths[i], self.packing_length, self.packing_strategy)
                for i in range(self.packing_num_proc)]

        ctx = _resolve_mp_context(self.multiprocessing_context)
        self._out_queue = ctx.Queue()
        # Pass the out-queue/params as explicit args (not via `self`) so a non-fork context does not
        # need to pickle the whole dataset held by `self`.
        _spawn_workers(ctx, target=self.create_packed_idx, jobs=self._worker_jobs(jobs))
        self.packed_idx = [[] for _ in range(self.packing_num_proc)]
        self.packed_length = [[] for _ in range(self.packing_num_proc)]
        desc = 'Packing: ' if self.packing_num_proc == 1 else f'Packing (num_proc={self.packing_num_proc}): '
        with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as prog_bar:
            finished_workers = 0
            while finished_workers < self.packing_num_proc:
                rank, sequences, data_len = self._out_queue.get()
                if data_len == -1:
                    finished_workers += 1
                    continue
                prog_bar.update(data_len)
                self.packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                self.packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]
        self.packed_idx = list(chain.from_iterable(self.packed_idx))
        self.packed_length = list(chain.from_iterable(self.packed_length))

    def _load_packing_cache(self) -> bool:
        if not self.packing_cache_file or not os.path.isfile(self.packing_cache_file):
            return False
        try:
            with open(self.packing_cache_file, 'rb') as f:
                cached = pickle.load(f)
            if (cached.get('cache_key') != self.packing_cache_key
                    or cached.get('dataset_len') != len(self.dataset)):
                return False
            self.packed_idx = cached['packed_idx']
            self.packed_length = cached['packed_length']
            logger.info(f'Loaded packing cache: {self.packing_cache_file}')
            return True
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError, TypeError):
            return False

    def _save_packing_cache(self):
        parent = os.path.dirname(self.packing_cache_file)
        os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix='.packing.', suffix='.tmp', dir=parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump({
                    'cache_key': self.packing_cache_key,
                    'dataset_len': len(self.dataset),
                    'packed_idx': self.packed_idx,
                    'packed_length': self.packed_length,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.packing_cache_file)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _worker_jobs(self, jobs):
        # Bind the shared out-queue into each per-worker arg tuple at spawn time (queue may be rebuilt on
        # the fork fallback), keeping it out of `self` so non-fork contexts don't pickle the dataset.
        return [(rank, offset, lengths, self._out_queue, packing_length, packing_strategy)
                for rank, offset, lengths, packing_length, packing_strategy in jobs]

    @staticmethod
    def create_packed_idx(rank, offset, lengths, out_queue, packing_length, packing_strategy):
        data = [(i + offset, sum(length) if isinstance(length, list) else length) for i, length in enumerate(lengths)]
        i = 0
        input_data = []
        while True:
            new_data = data[i:i + PackingDataset.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += PackingDataset.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            sequences, input_data = calculate_matched_group(
                input_data, packing_length, is_finished=is_finished, strategy=packing_strategy)
            out_queue.put((rank, sequences, len(new_data)))
        out_queue.put((rank, [], -1))

    def __getitem__(self, index):
        sequence = self.packed_idx[index]
        row = [self.dataset[i] for i in sequence]
        return row

    def __len__(self):
        return len(self.packed_idx)


class IterablePackingDataset(IterableDataset):

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        packing_length: Optional[int] = None,
        strict: bool = False,
        cyclic: bool = False,
        packing_strategy: str = 'binpack',
        multiprocessing_context: Optional[str] = None,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.multiprocessing_context = multiprocessing_context
        self.packing_length = packing_length or self.template.max_length

        self.packing_interval = packing_interval
        self.cyclic = cyclic
        self.packing_strategy = packing_strategy
        self.workers = []

        ctx = _resolve_mp_context(self.multiprocessing_context)
        self._in_queue = ctx.Queue()
        self._out_queue = ctx.Queue()
        # Pass the queues/template as explicit args (not via `self`) so a non-fork context only needs to
        # pickle the (model-stripped, see Template.__getstate__) template, not the whole dataset.
        self.workers = _spawn_workers(ctx, target=self._processor, jobs=self._worker_jobs())

    def _worker_jobs(self):
        return [(self._in_queue, self._out_queue, self.template, self.strict)] * self.num_proc

    @staticmethod
    def _processor(in_queue, out_queue, template, strict):
        while True:
            i, data = in_queue.get()
            encoded_data = {}
            try:
                encoded_data = template.encode(data, return_length=True)
            except Exception as e:
                if strict and not isinstance(e, MaxLengthError):
                    raise
            out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator) -> int:
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res, num_samples):
        res = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            if not data:
                continue
            res[i] = data if isinstance(data, list) else [data]
        res = [(item, len(item['input_ids'])) for group in res if group for item in group]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    def set_epoch(self, epoch: int):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        data = []
        while True:
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = calculate_matched_group(
                data, self.packing_length, is_finished=finished, strategy=self.packing_strategy)
            res = []
            for row in sequences:
                res.append([r[0] for r in row])
            yield from res
            if finished:
                break
