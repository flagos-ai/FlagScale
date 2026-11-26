# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Loss-scale strategies, vendored from ms-swift (Apache-2.0).

``--sft-loss-scale`` resolves through :func:`get_loss_scale`. The modules are
byte-identical to ``swift/loss_scale/`` apart from their imports: the three names
the originals took from ``swift.template`` live in :mod:`.types` here, so this
package has no ``swift`` dependency at all.

``tests/unit_tests/train/megatron/test_swift_sft_align.py`` asserts this package
and ``swift.loss_scale`` agree on every registered strategy.
"""

from .base import ALL_BASE_STRATEGY, ConcatLossScale, ConfigLossScale, LossScale
from .mapping import get_loss_scale, loss_scale_map
from .types import ContextType, Messages, get_last_user_round
from .utils import calculate_loss_scale, split_str_parts_by

__all__ = [
    "ALL_BASE_STRATEGY",
    "ConcatLossScale",
    "ConfigLossScale",
    "LossScale",
    "get_loss_scale",
    "loss_scale_map",
    "ContextType",
    "Messages",
    "get_last_user_round",
    "calculate_loss_scale",
    "split_str_parts_by",
]
