# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Local-file dataset loading with ms-swift's ``#N`` sampling syntax.

Upstream ``swift/dataset/loader.py`` (384 lines, plus ``dataset_meta``,
``dataset_syntax``, ``register`` and ``swift.hub``) exists mostly to resolve a
dataset *id* against ModelScope or HuggingFace: downloading repos, matching
registered ``DatasetMeta`` entries, picking subsets and splits, interleaving
several sources. FlagScale's SFT path points at local jsonl files, so only the
local branch is reproduced here:

    hf_load_dataset('json', data_files=...) -> AutoPreprocessor -> #N sampling
      -> optional train/val split

The pieces that are reproduced keep upstream's semantics exactly -- ``#N``
parsing, the sampling behavior including repeat-sampling when ``N`` exceeds the
file, and the train/val split arithmetic -- because the ``#N`` form is already
used by the reference training scripts.

Deliberately not supported (each raises rather than silently misbehaving):
remote dataset ids, ``hf::``/``ms::`` prefixes, ``:subset`` selection, streaming,
and interleaving. See :func:`load_dataset`.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from datasets import load_dataset as hf_load_dataset

from .._utils import get_cache_dir, get_logger, get_seed, safe_ddp_context
from .preprocessor import AutoPreprocessor, RowPreprocessor
from .utils import sample_dataset

logger = get_logger()

# Extensions datasets can read directly; 'jsonl' and 'txt' need the alias.
_FILE_TYPES = {'jsonl': 'json', 'txt': 'text'}


@dataclass
class DatasetSyntax:
    """The ``path#N`` form, parsed.

    Mirrors the subset of ``swift/dataset/dataset_syntax.py`` that applies to a
    local file. ``dataset_sample`` is the ``N`` in ``train.jsonl#512``.
    """

    dataset: str
    dataset_sample: Optional[int] = None

    @classmethod
    def parse(cls, dataset: str) -> 'DatasetSyntax':
        raw = dataset
        if os.path.exists(dataset):
            return cls(dataset.strip(), None)

        for prefix in ('hf::', 'ms::'):
            if dataset.lower().startswith(prefix):
                raise ValueError(
                    f'{raw!r}: hub-prefixed dataset ids are not supported by the FlagScale '
                    'SFT loader, which reads local files only. Download the dataset and pass a path.'
                )

        path, _, sample = dataset.rpartition('#')
        if not path:
            # No '#', so the whole string was meant to be a path.
            raise FileNotFoundError(
                f'{raw!r}: dataset not found. The FlagScale SFT loader accepts local file paths, '
                "optionally with ms-swift's '#N' sampling suffix (e.g. 'train.jsonl#512')."
            )
        if not sample.isdigit():
            raise ValueError(f'{raw!r}: expected an integer after "#", got {sample!r}.')
        path = path.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f'{raw!r}: {path!r} does not exist.')
        dataset_sample = int(sample)
        if dataset_sample <= 0:
            raise ValueError(f'{raw!r}: the sample count after "#" must be positive.')
        return cls(path, dataset_sample)


def _load_file(path: str, *, num_proc: int, load_from_cache_file: bool, strict: bool) -> HfDataset:
    ext = os.path.splitext(path)[1].lstrip('.')
    file_type = _FILE_TYPES.get(ext) or ext
    kwargs = {'split': 'train', 'streaming': False, 'num_proc': num_proc}
    if file_type == 'csv':
        kwargs['na_filter'] = False
    # Only one rank should materialize the cache; the rest reuse it.
    with safe_ddp_context(None, True):
        kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
        dataset = hf_load_dataset(file_type, data_files=path, **kwargs)
    # AutoPreprocessor normalizes the row shape (messages / alpaca / sharegpt ...)
    # and is what validates the per-message keys, including 'loss'/'loss_scale'.
    dataset = AutoPreprocessor()(
        dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict
    )
    return RowPreprocessor.remove_useless_columns(dataset)


def _inject_dataset_routing_tag(dataset: HfDataset, ds_name: str) -> HfDataset:
    """Inject the ``dataset`` column for multi-teacher routing (constant per source)."""
    return dataset.add_column('dataset', [ds_name] * len(dataset))


def _post_process(
    train_dataset: HfDataset,
    *,
    dataset_sample: Optional[int],
    split_dataset_ratio: float,
    shuffle: bool,
    random_state,
) -> Tuple[Optional[HfDataset], Optional[HfDataset]]:
    """Apply ``#N`` sampling and the optional train/val split.

    Kept arithmetically identical to ms-swift's ``BaseDatasetLoader.post_process``
    so a given seed and ``#N`` yield the same rows here as there.
    """
    assert dataset_sample is None or dataset_sample > 0
    assert 0 <= split_dataset_ratio <= 1
    if dataset_sample is None:
        dataset_sample = len(train_dataset)
    if split_dataset_ratio == 0:
        return sample_dataset(train_dataset, dataset_sample, shuffle, random_state), None
    if split_dataset_ratio == 1:
        val_dataset = train_dataset
        assert dataset_sample <= len(val_dataset), (
            f'val_sample: {dataset_sample}, len(val_dataset): {len(val_dataset)}')
        return None, sample_dataset(val_dataset, dataset_sample, shuffle, random_state)

    # Avoid duplication in the val_dataset.
    train_len = min(len(train_dataset), dataset_sample)
    val_sample = max(int(train_len * split_dataset_ratio), 1)
    train_sample = dataset_sample - val_sample
    assert train_sample > 0
    with safe_ddp_context(None, True):
        train_dataset, val_dataset = train_dataset.train_test_split(
            test_size=val_sample, shuffle=shuffle, seed=get_seed(random_state)).values()
    return sample_dataset(train_dataset, train_sample, shuffle, random_state), val_dataset


def load_dataset(
    datasets: Union[List[str], str],
    *,
    split_dataset_ratio: float = 0.,
    seed: Union[int, np.random.RandomState, None] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    strict: bool = False,
) -> Tuple[Optional[HfDataset], Optional[HfDataset]]:
    """Load one or more local dataset files, concatenated.

    Args:
        datasets: Paths, each optionally carrying a ``#N`` sampling suffix.
        split_dataset_ratio: Fraction split off as validation. 0 means no split.
        seed: Seed or RandomState driving sampling and splitting.
        num_proc: Worker processes for loading and preprocessing.
        load_from_cache_file: Reuse a previously preprocessed cache.
        shuffle: Shuffle while sampling.
        streaming: Unsupported; must be False.
        strict: Raise on a malformed row instead of dropping it.

    Returns:
        ``(train_dataset, val_dataset)``; ``val_dataset`` is None unless
        ``split_dataset_ratio > 0``.

    The signature mirrors the subset of ms-swift's ``load_dataset`` this path
    uses, so the call site reads the same either way.
    """
    if streaming:
        raise NotImplementedError(
            'streaming=True is not supported by the FlagScale SFT loader. Packing needs '
            'per-sample lengths up front, which requires a map-style dataset.'
        )
    if isinstance(datasets, str):
        datasets = [datasets]
    if not datasets:
        raise ValueError('`datasets` must not be empty.')

    random_state = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)

    train_parts: List[HfDataset] = []
    val_parts: List[HfDataset] = []
    for entry in datasets:
        syntax = DatasetSyntax.parse(entry)
        raw = _load_file(
            syntax.dataset,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            strict=strict,
        )
        train_part, val_part = _post_process(
            raw,
            dataset_sample=syntax.dataset_sample,
            split_dataset_ratio=split_dataset_ratio,
            shuffle=shuffle,
            random_state=random_state,
        )
        if train_part is not None:
            train_parts.append(_inject_dataset_routing_tag(train_part, syntax.dataset))
        if val_part is not None:
            val_parts.append(_inject_dataset_routing_tag(val_part, syntax.dataset))

    def _concat(parts):
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    return _concat(train_parts), _concat(val_parts)
