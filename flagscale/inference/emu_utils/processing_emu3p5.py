import os.path as osp

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

from vllm.outputs import RequestOutput

from flagscale.logger import logger

try:
    from src.utils.generation_utils import multimodal_decode
    from src.utils.input_utils import format_image_string, smart_resize
    from src.vision_tokenizer.ibq import IBQ
except ImportError as e:
    print(f"ImportError: {e}")
    raise ImportError(
        """
        Please clone the Emu3.5 repository and put the 'src' folder under FlagScale.
        ```
        git clone --no-checkout https://github.com/baaivision/Emu3.5.git tmp_repo
        cd tmp_repo
        git sparse-checkout init --cone
        git sparse-checkout set src assets
        git checkout main
        mv assets src ../
        cd ..
        rm -rf tmp_repo
        ```
        """
    )


aspect_ratios = {
    "4:3": "55*73",
    "21:9": "41*97",
    "16:9": "47*85",
    "3:2": "52*78",
    "1:1": "64*64",
    "3:4": "73*55",
    "9:16": "85*47",
    "2:3": "78*52",
    "default": "55*73",
    "auto": None,
}

special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

resolution_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]

UNCOND_PROMPT_1 = "<|extra_203|>You are a helpful assistant. USER: "
UNCOND_PROMPT_2 = " ASSISTANT: <|extra_100|>"

PROMPT_TEMPLATE_1 = "<|extra_203|>You are a helpful assistant for {type} task. USER: "
PROMPT_TEMPLATE_2 = "{question} ASSISTANT: <|extra_100|>"


def build_image(image, image_area, tokenizer, vq_model):
    image = smart_resize(image, image_area)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image = torch.tensor(np.array(image) / 127.5 - 1.0).to(device, dtype).permute(2, 0, 1)
    _, _, token = vq_model.encode(image[None])
    token = token[-1].view(h // 16, w // 16).detach().cpu().numpy()
    return format_image_string(tokenizer, token)


class Emu3p5Processor:
    def __init__(
        self,
        task_type: str,
        tokenizer_path: str,
        vq_model_path: str,
        image_area: int = 720 * 720,
        ratio: str = "default",
        vq_type: str = "ibq",
        device: str = "cuda:0",
    ):
        self.task_type = task_type
        self.tokenizer_path = tokenizer_path
        self.vq_model_path = vq_model_path
        self.vq_type = vq_type
        self.image_area = image_area
        self.device = device
        self.special_token_ids = {}

        self.resolution_map = {}
        self.ratio = None
        self.image_width = None
        self.image_height = None
        if self.task_type == "t2i":
            self.ratio = aspect_ratios[ratio]
            logger.info(f"[INFO] Set image ratio to {self.ratio} for task type {self.task_type}")
            if self.ratio is not None:
                height_str, width_str = self.ratio.split("*")
                self.image_width, self.image_height = int(width_str), int(height_str)

        self.build_text_tokenizer()
        self.build_image_tokenizer()
        self.build_special_tokens()

        if self.task_type in ["t2i", "x2i"]:
            self.stop_token_id = self.special_token_ids["EOI"]
        else:
            self.stop_token_id = self.special_token_ids["EOS"]

    def build_text_tokenizer(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            special_tokens_file=osp.join(self.tokenizer_path, "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
        self.text_tokenizer.bos_token = "<|extra_203|>"
        self.text_tokenizer.eos_token = "<|extra_204|>"
        self.text_tokenizer.pad_token = "<|endoftext|>"
        self.text_tokenizer.eol_token = "<|extra_200|>"
        self.text_tokenizer.eof_token = "<|extra_201|>"
        self.text_tokenizer.tms_token = "<|extra_202|>"
        self.text_tokenizer.img_token = "<|image token|>"
        self.text_tokenizer.boi_token = "<|image start|>"
        self.text_tokenizer.eoi_token = "<|image end|>"
        self.text_tokenizer.bss_token = "<|extra_100|>"
        self.text_tokenizer.ess_token = "<|extra_101|>"
        self.text_tokenizer.bog_token = "<|extra_60|>"
        self.text_tokenizer.eog_token = "<|extra_61|>"
        self.text_tokenizer.boc_token = "<|extra_50|>"
        self.text_tokenizer.eoc_token = "<|extra_51|>"

    def build_image_tokenizer(self):
        match self.vq_type:
            case "ibq":
                cfg = OmegaConf.load(osp.join(self.vq_model_path, "config.yaml"))
                self.img_tokenizer = IBQ(**cfg)
                ckpt = torch.load(
                    osp.join(self.vq_model_path, "model.ckpt"),
                    map_location="cpu",
                    weights_only=True,
                )
                self.img_tokenizer.load_state_dict(ckpt)
                self.img_tokenizer.eval().to(self.device)
                self.img_tokenizer.requires_grad_(False)
            case _:
                raise NotImplementedError(f"Unsupported vision tokenizer type: {self.vq_type}")

    def build_special_tokens(self):
        for k, v in special_tokens.items():
            self.special_token_ids[k] = self.text_tokenizer.encode(v)[0]

        for digit_str in resolution_str:
            self.resolution_map[self.text_tokenizer.encode(digit_str)[0]] = digit_str

    def process_inputs(self, question):
        reference_image = []
        if not isinstance(question, str):
            if isinstance(question["reference_image"], list):
                print(f"[INFO] {len(question['reference_image'])} reference images are provided")
                for img in question["reference_image"]:
                    reference_image.append(Image.open(img).convert("RGB"))
            else:
                print("[INFO] 1 reference image is provided")
                img = question["reference_image"]
                reference_image.append(Image.open(img).convert("RGB"))
            question = question["prompt"]
        else:
            print("[INFO] No reference image is provided")

        prompt_1 = PROMPT_TEMPLATE_1.format(type=self.task_type)
        prompt_2 = PROMPT_TEMPLATE_2.format(question=question)

        img_str = ""
        for img in reference_image:
            img_str += build_image(img, self.image_area, self.text_tokenizer, self.img_tokenizer)
            torch.cuda.empty_cache()

        prompt = prompt_1 + img_str + prompt_2
        uncond_prompt = UNCOND_PROMPT_1 + img_str + UNCOND_PROMPT_2

        input_ids = self.text_tokenizer.encode(prompt, add_special_tokens=False)
        uncond_input_ids = self.text_tokenizer.encode(uncond_prompt, add_special_tokens=False)
        torch.cuda.empty_cache()

        if input_ids[0] != self.special_token_ids["BOS"]:
            input_ids = [self.special_token_ids["BOS"], *input_ids]

        return input_ids, uncond_input_ids

    def process_results(self, results: list[RequestOutput]):
        cond_res, _ = results

        token_ids = torch.tensor(cond_res.outputs[0].token_ids, device=self.device)
        outputs = self.text_tokenizer.decode(token_ids, skip_special_tokens=False)

        mm_outputs = multimodal_decode(outputs, self.text_tokenizer, self.img_tokenizer)
        return mm_outputs
