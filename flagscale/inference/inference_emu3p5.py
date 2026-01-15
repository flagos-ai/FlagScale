import os

from omegaconf import DictConfig
from tqdm import tqdm

import vllm
from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config
from flagscale.inference.emu_utils import Emu3p5Processor
from flagscale.inference.emu_utils.prompt_case import EMU_TASKS
from flagscale.logger import logger


def generate(
    model: LLM, processor: Emu3p5Processor, prompts: list, sampling_cfg: DictConfig, save_dir: str
):
    os.makedirs(save_dir, exist_ok=True)
    for name, question in tqdm(prompts, total=len(prompts)):
        logger.info(f">>> Processing: {name=}, {question=}")

        save_path = os.path.join(save_dir, name)
        os.makedirs(save_path, exist_ok=True)

        input_ids, uncond_input_ids = processor.process_inputs(question)

        inputs = {"prompt_token_ids": input_ids, "uncond_prompt_token_ids": uncond_input_ids}

        extra_args = {
            "guidance_scale": sampling_cfg.guidance_scale,
            "text_top_k": sampling_cfg.text_top_k,
            "text_top_p": sampling_cfg.text_top_p,
            "text_temperature": sampling_cfg.text_temperature,
            "visual_top_k": sampling_cfg.image_top_k,
            "visual_top_p": sampling_cfg.image_top_p,
            "visual_temperature": sampling_cfg.image_temperature,
            "width": processor.image_width,
            "height": processor.image_height,
            "area": processor.image_area if processor.image_width else None,
        }
        sampling_params = SamplingParams(
            top_k=sampling_cfg.top_k,
            top_p=sampling_cfg.top_p,
            temperature=sampling_cfg.temperature,
            max_tokens=sampling_cfg.max_tokens,
            detokenize=False,
            stop_token_ids=[processor.stop_token_id],
            extra_args=extra_args,
        )
        logger.info(f"{sampling_params=}")
        results = model.generate(inputs, sampling_params=sampling_params)

        logger.info("-" * 40)
        mm_outputs = processor.process_results(results)
        for i, (out_type, output) in enumerate(mm_outputs):
            if out_type in ["text", "global_cot", "image_cot"]:
                with open(f"{save_path}/output_{name}_{i}.txt", "w") as f:
                    f.write(output)
                logger.info(
                    f">>> 📄[OUTPUT-{i}][{out_type}]: saved to {save_path}/output_{name}_{i}.txt"
                )
            elif out_type == "image":
                output_image = output.convert("RGB")
                output_image.save(f"{save_path}/output_{name}_{i}.png")
                logger.info(
                    f">>> 📷[OUTPUT-{i}][{out_type}]: saved to {save_path}/output_{name}_{i}.png"
                )
            else:
                raise ValueError(f"Unknown output type: {out_type}")


if __name__ == "__main__":
    logger.info(f"[vllm.__file__] {vllm.__file__}")

    cfg = parse_config()
    task_type = cfg.generate.get("task_type", None)
    assert task_type in [
        "t2i",
        "x2i",
        "story",
        "howto",
    ], f"Unsupported task_type: {task_type}. Options: 't2i', 'x2i', 'story', and 'howto'."

    cases = EMU_TASKS[task_type]
    if isinstance(cases, dict):
        prompts = [(n, p) for n, p in cases.items()]
    elif isinstance(cases, list):
        prompts = [(f"{idx:03d}", p) for idx, p in enumerate(cases)]
    else:
        raise ValueError("prompts should be a list or dict.")

    llm_cfg = cfg.get("llm", {})
    save_dir = llm_cfg.pop("save", ".")
    tokenizer_path = llm_cfg.get("tokenizer", None)
    vq_model_path = llm_cfg.pop("vq_model", None)
    assert tokenizer_path and vq_model_path, "Please set the tokenzier and vq_model in llm config."

    image_area = cfg.generate.get("image_area", 720 * 720)
    ratio = cfg.generate.get("ratio", "default")
    processor = Emu3p5Processor(task_type, tokenizer_path, vq_model_path, image_area, ratio)

    llm = LLM(
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
            "cfg_batch_size": 2,
            "boi_token_id": processor.special_token_ids["BOI"],
            "soi_token_id": processor.special_token_ids["IMG"],
            "eol_token_id": processor.special_token_ids["EOL"],
            "eoi_token_id": processor.special_token_ids["EOI"],
            "resolution_map": processor.resolution_map,
        },
    )
    llm.set_tokenizer(processor.text_tokenizer)
    vllm_config = llm.llm_engine.vllm_config
    print(f"{vllm_config=}")

    generate(llm, processor, prompts, cfg.generate.get("sampling", {}), save_dir)
