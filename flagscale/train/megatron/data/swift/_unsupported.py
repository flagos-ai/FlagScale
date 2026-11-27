# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Placeholders for ms-swift entry points the vendored template never calls.

The vendored ``template/base.py`` and ``template/register.py`` keep a few
function-scope imports reaching into parts of ms-swift this port deliberately
leaves behind:

``get_model_processor``
    ``Template._get_model`` builds a meta-device dummy model when a template
    needs to inspect model internals. Text-only DeepSeek-V4 encoding never does.
``ModelInfo`` / ``ModelMeta``
    TYPE_CHECKING-only annotations in ``register.py``; replaced by the shim
    records from :mod:`.processor` at runtime.
``BaseArguments``
    A TYPE_CHECKING-only annotation on an ms-swift CLI helper.

They raise on use rather than silently degrading, so if the port ever grows a
path that needs one, it fails at that line instead of further downstream. The
type names are plain aliases because they only ever appear in annotations.
"""

from typing import Any


def get_model_processor(*args, **kwargs):
    raise NotImplementedError(
        "get_model_processor() is not available in the FlagScale port: it belongs "
        "to swift.model, which the SFT data path intentionally does not depend on. "
        "Reaching it means a template asked for a dummy model, which the text-only "
        "DeepSeek-V4 encode path does not do."
    )


# Annotation-only aliases (``if TYPE_CHECKING`` in the vendored modules).
ModelInfo = Any
ModelMeta = Any
BaseArguments = Any
