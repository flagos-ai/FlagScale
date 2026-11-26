"""Bounded, best-fit packing of sequential JSONL conversations for text SFT."""

import json
import math
import os
import random
from contextlib import ExitStack, closing
from itertools import count

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .sft_dataset import IGNORE_INDEX, SFTDataset, sft_batch


def _rows(path, shard, shards, split):
    """Assign a line to the byte range containing its first byte."""
    with open(path, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        lo, hi = (int(size * fraction) for fraction in split)
        start = lo + (hi - lo) * shard // shards
        end = lo + (hi - lo) * (shard + 1) // shards
        if start:
            source.seek(start - 1)
            source.readline()
        while source.tell() < end:
            line = source.readline()
            if not line:
                break
            if line.strip():
                yield json.loads(line)["messages"]


def _mixed_rows(blend, seed, shard, shards, split):
    paths, weights = blend
    with ExitStack() as stack:
        streams = [stack.enter_context(closing(_rows(path, shard, shards, split))) for path in paths]
        weights = list(weights) if weights is not None else [1] * len(paths)
        rng = random.Random(seed)
        while streams:
            index = rng.choices(range(len(streams)), weights=weights)[0]
            messages = next(streams[index], None)
            if messages is None:
                streams.pop(index)
                weights.pop(index)
            else:
                yield messages


def _best_fit(window, capacity):
    bins, free = [], []
    for sample in sorted(window, key=lambda sample: -len(sample[0])):
        size = len(sample[0])
        dest = min((i for i, space in enumerate(free) if space >= size),
                   key=free.__getitem__, default=None)
        if dest is None:
            bins.append([sample])
            free.append(capacity - size)
        else:
            bins[dest].append(sample)
            free[dest] -= size
    return bins, free


def window_packs(samples, capacity, window_size):
    """Bound pending tokens; carry the least full bin across windows."""
    pending, used = [], 0
    for sample in samples:
        size = len(sample[0])
        if not 0 < size <= capacity:
            continue
        if used + size > capacity * window_size:
            if window_size == 1:
                yield pending
                pending, used = [], 0
            else:
                bins, free = _best_fit(pending, capacity)
                carry = max(range(len(free)), key=free.__getitem__)
                pending = bins.pop(carry)
                used = capacity - free[carry]
                yield from bins
                del bins
        pending.append(sample)
        used += size
    if pending:
        if window_size == 1:
            yield pending
        else:
            yield from _best_fit(pending, capacity)[0]


def make_batch_item(pack, capacity, pad, pad_to_multiple=None):
    if pad_to_multiple is not None:
        used = sum(len(ids) for ids, _ in pack)
        capacity = ((used + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    tokens = torch.full((capacity,), pad, dtype=torch.long)
    labels = torch.full_like(tokens, IGNORE_INDEX)
    boundaries = [0]
    for ids, targets in pack:
        start = boundaries[-1]
        end = start + len(ids)
        tokens[start:end] = torch.as_tensor(ids)
        # Next-token targets never cross a conversation boundary.
        labels[start:end - 1] = torch.as_tensor(targets[1:])
        boundaries.append(end)

    # Account for tail padding in THD, as in SFTDataset.
    boundaries[-1] = capacity
    return sft_batch(tokens, labels, boundaries, pad)


class StreamingPackedSFTDataset(IterableDataset):
    def __init__(self, blend, tokenizer, capacity, window_size, seed, rank, world_size, split=(0., 1.),
                 pad_to_multiple=None):
        super().__init__()
        if not hasattr(tokenizer, "tokenize_conversation"):
            raise ValueError("Online SFT packing requires a conversation tokenizer")
        paths, weights = blend
        if not paths or (weights is not None and
                         any(not math.isfinite(w) or w <= 0 for w in weights)):
            raise ValueError("Online packing needs JSONL paths with positive finite source weights")
        self.blend = blend
        self.tokenizer = tokenizer
        if pad_to_multiple is not None and (pad_to_multiple < 1 or capacity % pad_to_multiple):
            raise ValueError("Packing capacity must be divisible by the sequence alignment")
        self.pad_to_multiple = pad_to_multiple
        self.capacity = capacity
        self.window_size = window_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.split_range = split
        self.consumed_samples = 0

    def _samples(self, shard, shards, epoch):
        for messages in _mixed_rows(self.blend, self.seed + epoch, shard, shards, self.split_range):
            for chat in SFTDataset._split_conversations(messages):
                yield self.tokenizer.tokenize_conversation(chat, return_target=True,
                                                           add_generation_prompt=False)

    def __iter__(self):
        worker = get_worker_info()
        workers, ident = (worker.num_workers, worker.id) if worker else (1, 0)
        # The DataLoader returns workers in round-robin order. Rotate its first
        # worker on restart and replay only locally consumed packs.
        local, remainder = divmod(self.consumed_samples, self.world_size)
        if remainder:
            raise ValueError("Consumed packs must be divisible by data parallel size")
        skip, first = divmod(local, workers)
        slot = (ident + first) % workers
        skip += int(slot < first)
        shard = self.rank * workers + slot
        for epoch in count():
            seen = False
            samples = self._samples(shard, self.world_size * workers, epoch)
            for pack in window_packs(samples, self.capacity, self.window_size):
                seen = True
                if skip:
                    skip -= 1
                    continue
                yield make_batch_item(pack, self.capacity, self.tokenizer.pad, self.pad_to_multiple)
            if not seen:
                raise ValueError(f"No eligible SFT conversations on input shard {shard}; "
                                 "check sequence length and the number of DP ranks/workers")
