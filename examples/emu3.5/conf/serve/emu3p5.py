# Modified from FlagSale/flagscale/inference/inference_emu3p5.py

import os
import sys
import uuid
from typing import Any

import torch

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.emu_utils import Emu3p5Processor
from flagscale.logger import logger

# Configuration parameters can refer to https://github.com/baaivision/Emu3.5/tree/main/configs
DEFAULT_CONFIG = {
    "task_type": "x2i",
    # "task_type": "t2i",
    # "task_type": "howto",
    # "task_type": "story",
    "image_area": 1048576,
    "ratio": "auto",
    "llm": {
        "model": "/path/to/models/Emu3.5",  # https://www.modelscope.cn/models/BAAI/Emu3.5/files
        # "model": "/path/to/models/Emu3.5-Image",   # https://modelscope.cn/models/BAAI/Emu3.5-Image/files
        "tokenizer": "/path/to/src/tokenizer_emu3_ibq",  # https://github.com/baaivision/Emu3.5/src/tokenizer_emu3_ibp
        "vq_model": "/path/to/models/Emu3.5-VisionTokenizer",  # https://www.modelscope.cn/models/BAAI/Emu3.5-VisionTokenizer/files
        "trust_remote_code": True,
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.7,
        "disable_log_stats": False,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "seed": 42,
        "dtype": "auto",
    },
    "sampling": {
        "detokenize": False,
        "guidance_scale": 3.0,
        "text_top_k": 1024,
        "text_top_p": 0.9,
        "text_temperature": 1.0,
        "image_top_k": 5120,
        "image_top_p": 1.0,
        "image_temperature": 1.0,
        "top_k": 131072,
        "top_p": 1.0,
        "temperature": 1.0,
        "max_tokens": 5120,
    },
}


class Emu3_5InferencePipeline:
    def __init__(self, config: dict[str, Any] = DEFAULT_CONFIG):
        self.cfg = config
        self.processor: Emu3p5Processor | None = None
        self.llm: LLM | None = None

        self.task_type = self.cfg["task_type"]
        assert self.task_type in [
            "t2i",
            "x2i",
            "story",
            "howto",
        ], f"Unsupported task_type: {self.task_type}. Options: 't2i', 'x2i', 'story', and 'howto'."

        self.ratio = self.cfg["ratio"]

        self._load_models()

    def _load_models(self):
        cfg = self.cfg
        llm_cfg = cfg.get("llm", {})

        tokenizer_path = llm_cfg.get("tokenizer", None)
        vq_model_path = llm_cfg.pop("vq_model", None)
        assert tokenizer_path and vq_model_path, (
            "Please set the tokenzier and vq_model in llm config."
        )

        image_area = cfg["image_area"]

        try:
            self.processor = Emu3p5Processor(
                task_type=self.task_type,
                tokenizer_path=tokenizer_path,
                vq_model_path=vq_model_path,
                image_area=image_area,
                ratio=self.ratio,
            )

            if self.task_type in ["t2i", "x2i"]:
                self.processor.stop_token_id = self.processor.special_token_ids["EOI"]
            else:
                self.processor.stop_token_id = self.processor.special_token_ids["EOS"]

            logger.info(
                f"Emu3p5Processor initialized for task: {self.task_type} (Ratio: {self.ratio})."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Emu3p5Processor: {e}")
            sys.exit(1)

        try:
            self.llm = LLM(
                **llm_cfg,
                max_num_batched_tokens=26000,
                max_num_seqs=2,
                generation_config="vllm",
                scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
                compilation_config={
                    "full_cuda_graph": True,
                    "backend": "cudagraph",
                    "cudagraph_capture_sizes": [1, 2],
                },
                additional_config={
                    "boi_token_id": self.processor.special_token_ids["BOI"],
                    "soi_token_id": self.processor.special_token_ids["IMG"],
                    "eol_token_id": self.processor.special_token_ids["EOL"],
                    "eoi_token_id": self.processor.special_token_ids["EOI"],
                    "resolution_map": self.processor.resolution_map,
                },
            )
            self.llm.set_tokenizer(self.processor.text_tokenizer)
            logger.info("vLLM LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM LLM: {e}")
            sys.exit(1)

    @torch.no_grad()
    def forward(
        self, prompt: str, reference_image: list[str] | None = None
    ) -> list[dict[str, Any]]:
        question_data: Any
        if reference_image and len(reference_image) > 0:
            question_data = {"prompt": prompt, "reference_image": reference_image}
        else:
            question_data = prompt

        logger.info(f">>> Processing: {prompt=}, {reference_image=}")

        input_ids, uncond_input_ids = self.processor.process_inputs(question_data)

        inputs = {"prompt_token_ids": input_ids, "uncond_prompt_token_ids": uncond_input_ids}

        sampling_cfg = self.cfg["sampling"]
        extra_args = {
            "guidance_scale": sampling_cfg["guidance_scale"],
            "text_top_k": sampling_cfg["text_top_k"],
            "text_top_p": sampling_cfg["text_top_p"],
            "text_temperature": sampling_cfg["text_temperature"],
            "visual_top_k": sampling_cfg["image_top_k"],
            "visual_top_p": sampling_cfg["image_top_p"],
            "visual_temperature": sampling_cfg["image_temperature"],
        }
        sampling_params = SamplingParams(
            top_k=sampling_cfg["top_k"],
            top_p=sampling_cfg["top_p"],
            temperature=sampling_cfg["temperature"],
            max_tokens=sampling_cfg["max_tokens"],
            detokenize=False,
            stop_token_ids=[self.processor.stop_token_id],
            seed=42,
            extra_args=extra_args,
        )
        logger.info(f"{sampling_params=}")
        results = self.llm.generate(inputs, sampling_params=sampling_params)

        logger.info("-" * 40)

        mm_outputs = self.processor.process_results(results)

        formatted_outputs = []
        for i, (out_type, output) in enumerate(mm_outputs):
            item = {"type": out_type}
            if out_type in ["text", "global_cot", "image_cot"]:
                item["content"] = output
                logger.info(f">>> 📄[OUTPUT-{i}][{out_type}]: {output}")
            elif out_type == "image":
                current_dir = os.path.dirname(os.path.abspath(__file__))
                outputs_dir = os.path.join(current_dir, "outputs")
                os.makedirs(outputs_dir, exist_ok=True)
                output_name = os.path.join(outputs_dir, f"task_{i}_{uuid.uuid4()}.png")
                output_image = output.convert("RGB")
                output_image.save(output_name)
                item["content"] = os.path.abspath(output_name)
                logger.info(f">>> 📷[OUTPUT-{i}][{out_type}]: saved to {output_name}")
            else:
                item["content"] = str(output)
                raise ValueError(f"Unknown output type: {out_type}")

            formatted_outputs.append(item)
        return formatted_outputs


def main(task_type: str = "x2i"):
    DEFAULT_CONFIG["task_type"] = task_type

    pipeline = Emu3_5InferencePipeline(DEFAULT_CONFIG)

    sampler = {
        "prompt": "As shown in the second figure: The ripe strawberry rests on a green leaf in the garden. Replace the chocolate truffle in first image with ripe strawberry from 2nd image",
        "reference_image": ["/path/to/assets/ref_img.png"],
    }

    logger.info(f"--- Starting Fixed Task ({pipeline.task_type}) Test Case ---")
    pipeline.forward(prompt=sampler["prompt"], reference_image=sampler["reference_image"])


if __name__ == "__main__":
    task_type = "t2i"  # task_type should be in (x2i, t2i, howto, story)

    main(task_type)
