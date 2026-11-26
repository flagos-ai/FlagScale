# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Online SFT data pipeline for FlagScale, ported from ms-swift.

The tokenizer/template, dataset, packing, and loss-scale pieces required by SFT
are vendored here so training does not depend on the external ``swift`` package.
The FlagScale-facing API consists of:

- :class:`SwiftSFTDataset` -- online jsonl load + ``#N`` sampling, template
  tokenization (with ``loss_scale``) and best-fit-decreasing packing.
- :func:`build_swift_sft_collator` -- converts the ms-swift collator output into
  the batch keys FlagScale's Megatron loop consumes.
- :func:`build_processor` -- a local stand-in for ``swift.model.get_processor``,
  so the data path no longer depends on ``swift.model`` (and through it ``peft``
  and ``swift.sequence_parallel``).

The alignment tests in ``tests/unit_tests/train/megatron/test_swift_sft_align.py``
compare this implementation against ms-swift when that optional reference
package is installed.
"""

from .collator import build_swift_sft_collator
from .processor import build_processor
from .sft_dataset import SwiftSFTConfig, SwiftSFTDataset, build_swift_sft_datasets

__all__ = [
    "SwiftSFTConfig",
    "SwiftSFTDataset",
    "build_swift_sft_datasets",
    "build_swift_sft_collator",
    "build_processor",
]
