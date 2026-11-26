# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Vendored verbatim from ms-swift (Apache-2.0): swift/loss_scale/other.py.
from .base import ConfigLossScale


class IgnoreEmptyThinkLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_empty_think.json'


class IgnoreThinkPrefixLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_think_prefix.json'
