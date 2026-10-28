# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Online SFT data pipeline for FlagScale, ported from ms-swift.

Phase 1 reuses ms-swift (``swift``) as a library to guarantee behavioral parity
with the validated ``megatron sft`` training run. The FlagScale-native glue lives
here:

- :class:`SwiftSFTDataset` -- online jsonl load + ``#N`` sampling, template
  tokenization (with ``loss_scale``) and best-fit-decreasing packing.
- :func:`build_swift_sft_collator` -- converts the ms-swift collator output into
  the batch keys FlagScale's Megatron loop consumes.
- :func:`build_processor` -- a local stand-in for ``swift.model.get_processor``,
  so the data path no longer depends on ``swift.model`` (and through it ``peft``
  and ``swift.sequence_parallel``).

Phase 2 is retiring the ``swift`` runtime dependency module by module. The
alignment tests in ``tests/unit_tests/train/megatron/test_swift_sft_align.py``
pin each replacement against ms-swift, so a migrated piece has to reproduce the
original byte-for-byte before it lands.
"""

from .swift_sft_dataset import SwiftSFTConfig, SwiftSFTDataset, build_swift_sft_datasets
from .swift_collator import build_swift_sft_collator
from .swift_processor_shim import build_processor

__all__ = [
    "SwiftSFTConfig",
    "SwiftSFTDataset",
    "build_swift_sft_datasets",
    "build_swift_sft_collator",
    "build_processor",
]
