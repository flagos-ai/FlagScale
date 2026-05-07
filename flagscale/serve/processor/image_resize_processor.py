from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from flagscale.logger import logger
from flagscale.models.configs.types import PipelineFeatureType, PolicyFeature
from flagscale.models.utils.constants import OBS_IMAGES
from flagscale.train.processor import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="image_resize_processor")
class ImageResizeProcessorStep(ObservationProcessorStep):
    """Resizes observation images to a fixed ``[width, height]``.

    Only processes keys starting with ``observation.images.``.
    Expects images as ``np.ndarray`` in **HWC uint8** layout (H, W, 3).
    Output is ``np.ndarray`` in the same layout and dtype.
    Non-ndarray values and ``None`` are passed through unchanged.

    Example serve config::

        serve_preprocessor:
          steps:
            - registry_name: image_resize_processor
              config:
                image_size: [224, 224]
    """

    image_size: list[int] = field(default_factory=lambda: [224, 224])

    def observation(self, observation):
        target = tuple(self.image_size)
        for key in list(observation.keys()):
            if not key.startswith(OBS_IMAGES):
                continue
            img = observation[key]
            if isinstance(img, np.ndarray) and img.ndim == 3:
                observation[key] = np.array(Image.fromarray(img).resize(target))
            elif isinstance(img, np.ndarray):
                # Image layout (HWC uint8) is verified at the serving layer, so just warn here.
                logger.warning(
                    f"Skipping resize for '{key}': expected ndim=3 (HWC), got ndim={img.ndim} shape={img.shape}"
                )
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"image_size": self.image_size}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
