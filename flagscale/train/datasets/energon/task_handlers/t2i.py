import json
import random

from flagscale.train.datasets.energon.data_utils import pil_img2rgb
from flagscale.train.datasets.energon.sample_types import BagelSample
from flagscale.train.datasets.energon.task_handlers.base import BaseTaskHandler


class T2IHandler(BaseTaskHandler):
    def encode(self, sample, **kwargs):
        """Encode text-to-image sample: caption + VAE image."""
        transform = kwargs.get("transform")

        data_item = sample.get("json_data")
        images = sample.get("images", [])
        caption_dict = data_item.get("caption_dict", "")
        # print(f"{images=}, {caption_dict=}")

        assert images is not None and len(images) == 1

        image_tensor_list = []
        text_ids_list = []
        sequence_plan = []
        num_tokens = 0

        # Load image
        raw_image = pil_img2rgb(images[0])

        # transform image for VAE
        transform_stride = transform.stride
        image_tensor = transform(raw_image)
        image_tensor_list.append(image_tensor)
        height, width = image_tensor.shape[1:]
        num_tokens += width * height // transform_stride**2

        # Tokenize caption
        caption_dict = json.loads(caption_dict)
        caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
        assert len(caps_token) >= 1
        caption_token = random.choice(caps_token)

        # text_ids = self.tokenizer.encode(caption.get("caption"))
        if len(caption_token) > 0:
            text_ids_list.append(caption_token)
            num_tokens += len(caption_token)
            sequence_plan.append(
                {
                    "type": "text",
                    "enable_cfg": 1,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

        # VAE image plan
        if image_tensor_list:
            sequence_plan.append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

        return BagelSample(
            image_tensor_list=image_tensor_list,
            text_ids_list=text_ids_list,
            sequence_plan=sequence_plan,
            num_tokens=num_tokens,
            is_mandatory=sample.get("__subflavors__", {}).get("is_mandatory", False),
            subflavor=sample.get("__subflavors__", {}).get("task", "t2i"),
            __key__=sample.get("__key__", ""),
            __restore_key__=sample.get("__restore_key__", ()),
        )
