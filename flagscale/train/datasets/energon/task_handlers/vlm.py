from flagscale.train.datasets.energon.data_utils import pil_img2rgb
from flagscale.train.datasets.energon.sample_types import BagelSample
from flagscale.train.datasets.energon.task_handlers.base import BaseTaskHandler


class VLMHandler(BaseTaskHandler):
    def _parse_conversations(self, conversations: list[dict], num_images: int) -> list[dict]:
        """Parse conversation format into flat element list."""
        elements = []
        for conversation in conversations:
            role = conversation.get("from", "")
            value = conversation.get("value", "")
            if role == "human":
                if "<image>" not in value:
                    elements.append({"type": "text", "has_loss": 0, "text": value})
                else:
                    text_list = value.split("<image>")
                    for idx, text in enumerate(text_list):
                        if text.strip() != "":
                            elements.append({"type": "text", "has_loss": 0, "text": text.strip()})
                        if idx != len(text_list) - 1 and idx < num_images:
                            elements.append({"type": "image"})
            elif role == "gpt":
                elements.append({"type": "text", "has_loss": 1, "text": value})
        return elements

    def encode(self, sample, **kwargs):
        transform = kwargs.get("transform")
        frame_sampler = kwargs.get("frame_sampler")

        data_item = sample.get("json_data")
        images = sample.get("images", [])
        video_bytes = sample.get("video_bytes", None)
        conversations = data_item.get("conversations", [])

        image_tensor_list = []
        text_ids_list = []
        sequence_plan = []
        num_tokens = 0

        # Load images
        raw_images = None
        if images:
            raw_images = [pil_img2rgb(image) for image in images]
        elif video_bytes:
            raw_images = frame_sampler(video_bytes)
            special_tokens = "<image>" * len(raw_images)
            for item in conversations:
                if "<video>" in item["value"]:
                    item["value"] = item["value"].replace("<video>", special_tokens)
                    break
                else:
                    raise ValueError("Cannot find <video> in the conversation!")

        # Transform images
        transform_stride = transform.stride
        if raw_images:
            for raw_image in raw_images:
                image_tensor = transform(raw_image, img_num=len(raw_images))
                image_tensor_list.append(image_tensor)
                height, width = image_tensor.shape[1:]
                num_tokens += width * height // transform_stride**2

        # print(f"{len(image_tensor_list)=}")
        # Parse conversations into elements
        elements = self._parse_conversations(conversations, len(image_tensor_list))
        # print(f"{elements=}")

        # Build sequence_plan and text_ids_list
        for item in elements:
            if item["type"] == "text":
                text_ids = self.tokenizer.encode(item["text"])
                if len(text_ids) > 0:
                    text_ids_list.append(text_ids)
                    num_tokens += len(text_ids)
                    sequence_plan.append(
                        {
                            "type": "text",
                            "enable_cfg": 0,
                            "loss": item["has_loss"],
                            "special_token_loss": 0,
                            "special_token_label": None,
                        }
                    )
            elif item["type"] == "image":
                sequence_plan.append(
                    {
                        "type": "vit_image",
                        "enable_cfg": 0,
                        "loss": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                    }
                )

        has_loss = [item["loss"] for item in sequence_plan]
        if sum(has_loss) == 0:
            raise ValueError(
                f"No loss defined in current sample: {sample.get('__key__', '')=}, "
                f"{sample.get('__shard__', '')=}"
            )

        return BagelSample(
            image_tensor_list=image_tensor_list,
            text_ids_list=text_ids_list,
            sequence_plan=sequence_plan,
            num_tokens=num_tokens,
            is_mandatory=sample.get("__subflavors__", {}).get("is_mandatory", False),
            subflavor=sample.get("__subflavors__", {}).get("task", "vlm"),
            __key__=sample.get("__key__", ""),
            __restore_key__=sample.get("__restore_key__", ()),
        )
