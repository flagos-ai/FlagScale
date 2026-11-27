# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Dataset loading, encoding and packing, vendored from ms-swift (Apache-2.0).

``utils.py``, ``packing.py`` and ``preprocessor/`` are copied unchanged apart
from their imports. ``loader.py`` is the exception: it is a focused rewrite that
keeps the local-file path and the ``#N`` sampling/splitting arithmetic, and drops
the hub-resolution machinery (remote ids, registered dataset metas, subsets,
streaming, interleaving) that FlagScale's local-jsonl path never uses.

``tests/unit_tests/train/megatron/test_swift_sft_align.py`` compares this package
against ``swift.dataset`` on real data.
"""

from .loader import DatasetSyntax, load_dataset
from .packing import IterablePackingDataset, PackingDataset, calculate_matched_group
from .preprocessor import AutoPreprocessor, MessagesPreprocessor, RowPreprocessor
from .utils import AddLengthPreprocessor, EncodePreprocessor, LazyLLMDataset, sample_dataset

__all__ = [
    "DatasetSyntax",
    "load_dataset",
    "IterablePackingDataset",
    "PackingDataset",
    "calculate_matched_group",
    "AutoPreprocessor",
    "MessagesPreprocessor",
    "RowPreprocessor",
    "AddLengthPreprocessor",
    "EncodePreprocessor",
    "LazyLLMDataset",
    "sample_dataset",
]
