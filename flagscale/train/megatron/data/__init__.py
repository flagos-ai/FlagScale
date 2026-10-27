# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Online SFT data pipeline for FlagScale, ported from ms-swift.

Phase 1 reuses ms-swift (``swift``) as a library to guarantee behavioral parity
with the validated ``megatron sft`` training run. The FlagScale-native glue lives
here:

- :class:`SwiftSFTDataset` -- online jsonl load + ``#N`` sampling, template
  tokenization (with ``loss_scale``) and best-fit-decreasing packing.
- :func:`build_swift_sft_collator` -- converts the ms-swift collator output into
  the batch keys FlagScale's Megatron loop consumes.

A later phase will replace the packing layer with a FlagScale-native balanced
packing scheme while keeping the dataset/collator boundary stable.
"""

from .swift_sft_dataset import SwiftSFTConfig, SwiftSFTDataset, build_swift_sft_datasets
from .swift_collator import build_swift_sft_collator

__all__ = [
    "SwiftSFTConfig",
    "SwiftSFTDataset",
    "build_swift_sft_datasets",
    "build_swift_sft_collator",
]
