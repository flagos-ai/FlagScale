from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from flagscale.logger import logger
from flagscale.models.configs.types import PipelineFeatureType, PolicyFeature
from flagscale.models.utils.constants import OBS_IMAGES
from flagscale.train.processor import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="image_layout_processor")
class ImageLayoutProcessorStep(ObservationProcessorStep):
    """Converts image layout and optionally casts uint8 pixels to float32 [0, 1].

    Operates on ``observation.images.*`` keys. Input must be ``np.ndarray`` with ndim==3.

    Args:
        src_layout: Layout of the incoming image, ``"hwc"`` or ``"chw"``.
        dst_layout: Desired output layout, ``"hwc"`` or ``"chw"``.
        add_batch_dim: If True, prepend a batch dimension of size 1.
        to_float: If True, convert ``uint8`` [0, 255] to ``float32`` [0, 1].

    Example serve config::

        serve_preprocessor:
          steps:
            - registry_name: image_layout_processor
              config:
                src_layout: hwc
                dst_layout: chw
                add_batch_dim: true
                to_float: true
    """

    src_layout: Literal["hwc", "chw"] = "hwc"
    dst_layout: Literal["hwc", "chw"] = "chw"
    add_batch_dim: bool = False
    to_float: bool = False

    def observation(self, observation):
        for key in list(observation.keys()):
            if not key.startswith(OBS_IMAGES):
                continue
            img = observation[key]
            if not isinstance(img, np.ndarray):
                logger.warning(
                    f"image_layout_processor: skipping '{key}': expected np.ndarray, got {type(img).__name__}"
                )
                continue
            if img.ndim != 3:
                logger.warning(
                    f"image_layout_processor: skipping '{key}': expected ndim=3, got ndim={img.ndim} shape={img.shape}"
                )
                continue

            if self.to_float:
                if img.dtype != np.uint8:
                    logger.warning(
                        f"image_layout_processor: to_float=True but '{key}' has dtype={img.dtype}, expected uint8"
                    )
                img = img.astype(np.float32) / 255.0

            if self.src_layout != self.dst_layout:
                if self.src_layout == "hwc" and self.dst_layout == "chw":
                    img = img.transpose(2, 0, 1)
                elif self.src_layout == "chw" and self.dst_layout == "hwc":
                    img = img.transpose(1, 2, 0)

            if self.add_batch_dim:
                img = img[np.newaxis]

            observation[key] = img
        return observation

    def get_config(self) -> dict[str, Any]:
        return {
            "src_layout": self.src_layout,
            "dst_layout": self.dst_layout,
            "add_batch_dim": self.add_batch_dim,
            "to_float": self.to_float,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
