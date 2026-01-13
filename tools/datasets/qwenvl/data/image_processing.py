# Copied from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/image_processing.py. Below is the original copyright:
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.

import math
import random

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import Compose, RandAugment, Resize, ToPILImage

# config :https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
# Imagenet's mean and std.
pixel_mean = [0.48145466, 0.4578275, 0.40821073]
pixel_std = [0.26862954, 0.26130258, 0.27577711]

# Reshape for broadcasting.
pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)


# https://github.com/QwenLM/Qwen2.5-VL/blob/477fd9d4317266508705366ce36cac5b68d70936/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L89C1-L95C40
def convert_to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def _transform_train_aug():
    return Compose(
        [
            ToPILImage(),
            Resize(scale=random.random() / 2 + 0.5),
            convert_to_rgb,
            RandAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
        ]
    )


def _transform_test():
    return Compose([ToPILImage(), convert_to_rgb])


def standardize_image(img):
    """Standardize image pixel values."""
    return (T.ToTensor()(img) - pixel_mean) / pixel_std


def get_visual_transform(
    img,  # Path
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
    augment=False,
):
    # TODO(lizhiyu): Need to limit the aspect ratio of the image.
    # (reference https://github.com/QwenLM/Qwen2.5-VL/blob/477fd9d4317266508705366ce36cac5b68d70936/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L72)
    img = Image.open(img)
    img = convert_to_rgb(img)
    w, h = img.size
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    img = img.resize((w_bar, h_bar))

    # Standardize pixel values.
    img = standardize_image(img)
    imgs = [img]
    return imgs


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
