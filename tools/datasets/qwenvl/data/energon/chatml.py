# Copied from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/energon/chatml.py.

import pickle
import re
from dataclasses import dataclass

from webdataset.autodecode import Decoder

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class ChatMLSample(Sample):
    """multi-turn complex samples with images and videos"""

    imgs: list[str]
    videos: list[list[str]]
    conversation: str  # JSON string of GPT-format conversations


class NestedImagesPathHandler:
    def __init__(self, imagespec):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = ["jpgs", "videos"]
        self.extensions_mapping = {"jpgs": "jpg", "videos": "jpg"}

    def __call__(self, key, data):
        """Perform nested image decoding.

        :param key: file name extension
        :param data: binary data
        """
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        data = pickle.loads(data)
        return data


# During training, data is automatically decoded to from default webdataset to 'ChatMLSample' when loaded using energon-dataloader,
# and this is not done during preparation!!!
# After decoding, the data is passed into the TaskEncoder for further processing.
class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    __sample_type__ = ChatMLSample

    def __init__(
        self,
        path: EPath,
        *,
        auto_decode: bool = True,
        image_decode="torchrgb",
        ignore_decoder_errors: bool = False,
        av_decode="AVDecoder",
        video_decode_audio: bool = False,
        **kwargs,
    ):
        super().__init__(
            path,
            auto_decode=auto_decode,
            image_decode=image_decode,
            ignore_decoder_errors=ignore_decoder_errors,
            av_decode=av_decode,
            video_decode_audio=video_decode_audio,
            **kwargs,
        )
        if auto_decode:
            self._decoder = Decoder([NestedImagesPathHandler(self.image_decode)])
