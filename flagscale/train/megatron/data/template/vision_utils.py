# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Fail-loud stubs for the media loaders this text-only path never reaches.

``swift/template/vision_utils.py`` is ~520 lines of image/video/audio fetching
(HTTP with retries, base64, pixel rescaling, torchaudio resampling). The vendored
``base.py`` imports five names from it, and every call site is guarded by a
condition that is falsy for text-only input::

    load_image / rescale_image   <- only when ``inputs.images`` is non-empty
    _check_path                  <- only when ``self.root_image_dir`` is set,
                                    which init_env_args leaves None unless
                                    ``model_meta.is_multimodal``
    load_audio / load_batch      <- only when ``mode == 'vllm'`` and
                                    ``inputs.audios`` is non-empty

DeepSeek-V4 is text-only (no vision/audio sub-config, no multimodal tower), and
:func:`..swift_processor_shim.build_processor` refuses a multimodal config
outright, so none of these can fire.

They raise rather than return a placeholder: a silent no-op here would mean
images were dropped from a sample and the model trained on a truncated prompt,
which is far worse than a crash. If multimodal support is ever needed, vendor the
real module from ms-swift rather than filling these in.
"""


def _unsupported(name):

    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f"{name}() is not available: FlagScale's SFT data path is text-only. "
            "Reaching it means a sample carried images/video/audio. Vendor "
            "swift/template/vision_utils.py to add multimodal support."
        )

    _raise.__name__ = name
    return _raise


load_image = _unsupported('load_image')
load_audio = _unsupported('load_audio')
load_batch = _unsupported('load_batch')
rescale_image = _unsupported('rescale_image')
_check_path = _unsupported('_check_path')
