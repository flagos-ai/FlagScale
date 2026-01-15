import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

import vllm
from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config
from flagscale.inference.processing_emu3 import (
    CachedPrefixConstrainedLogitsProcessor,
    Emu3Processor,
)

POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."


def prepare_processor(llm_cfg, vq_model):
    tokenizer = AutoTokenizer.from_pretrained(
        llm_cfg.model, trust_remote_code=True, padding_side="left"
    )
    image_processor = AutoImageProcessor.from_pretrained(vq_model, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        vq_model, device_map="cuda:0", trust_remote_code=True
    ).eval()
    return Emu3Processor(image_processor, image_tokenizer, tokenizer)


def inference_t2i(cfg):
    """
    text-to-image task
    """

    # Step 1: Parse inference config
    prompts = cfg.generate.get("prompts", [])
    ratios = cfg.generate.get("ratios", [])
    assert len(prompts) == len(ratios), "Please set the same length of prompts and ratios."

    # Step 2: initialize the LLM engine
    llm_cfg = cfg.get("llm", {})
    vq_model = llm_cfg.pop("vq_model", None)
    assert vq_model, "Please set the vq_model in llm config."
    llm = LLM(**llm_cfg)

    # Step 3: initialize the emu3 processor
    emu3_processor = prepare_processor(llm_cfg, vq_model)

    # Step 4: Prepare inputs and sampling_parameters
    prompts = [p + POSITIVE_PROMPT for p in prompts]
    model_config = llm.llm_engine.get_model_config()
    negative_prompts = [NEGATIVE_PROMPT] * len(prompts)

    sampling_params = []
    positive_input_ids = []
    negative_input_ids = []
    for i in range(len(prompts)):
        kwargs = dict(
            mode="G",
            ratio=ratios[i],
            image_area=model_config.hf_config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = emu3_processor(text=prompts[i], **kwargs)
        neg_inputs = emu3_processor(text=negative_prompts[i], **kwargs)

        # prepare inputs
        positive_input_ids.append(pos_inputs.input_ids.tolist()[0])
        negative_input_ids.append(neg_inputs.input_ids.tolist()[0])

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = emu3_processor.build_prefix_constrained_fn(h, w)

        # initialize the sampling_parameters
        sampling_cfg = cfg.generate.get("sampling", {})
        assert not sampling_cfg.get("logits_processors", None), (
            "logits_processors is not supported yet."
        )
        sampling_params.append(
            SamplingParams(
                **sampling_cfg,
                logits_processors=[
                    CachedPrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1)
                ],
            )
        )

    # Step 6: build vllm inputs
    inputs = [
        {"prompt_token_ids": p_ids, "negative_prompt_token_ids": n_ids}
        for p_ids, n_ids in zip(positive_input_ids, negative_input_ids)
    ]
    print(f"=> {inputs=}")
    print(f"=> {sampling_params=}")

    # Step 7: generate outputs
    outputs = llm.generate(inputs, sampling_params)
    for idx_i, out in enumerate(outputs):
        output = torch.tensor(
            list(out.prompt_token_ids) + list(out.outputs[0].token_ids),
            dtype=pos_inputs.input_ids.dtype,
            device=pos_inputs.input_ids.device,
        )
        mm_list = emu3_processor.decode(output)
        for idx_j, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            im.save(f"result_{idx_i}_{idx_j}.png")
            print(f"Saved result_{idx_i}_{idx_j}.png")


def inference_i2t(cfg):
    """
    image understanding task
    """

    # Step 1: Parse inference config
    prompts = cfg.generate.get("prompts", [])
    images = cfg.generate.get("images", [])
    assert len(prompts) == len(images), "Please set the same length of prompts and images."

    # Step 2: initialize the LLM engine
    llm_cfg = cfg.get("llm", {})
    vq_model = llm_cfg.pop("vq_model", None)
    assert vq_model, "Please set the vq_model in llm config."
    llm = LLM(**llm_cfg)

    # Step 3: initialize the emu3 processor
    emu3_processor = prepare_processor(llm_cfg, vq_model)

    # Step 4: Prepare inputs and sampling_parameters
    sampling_params = []
    positive_input_ids = []
    for i in range(len(prompts)):
        kwargs = dict(
            mode="U",
            text=prompts[i],
            image=Image.open(images[i]),
            padding_image=False,
            return_tensors="pt",
        )
        pos_inputs = emu3_processor(**kwargs)

        # prepare inputs
        positive_input_ids.append(pos_inputs.input_ids.tolist()[0])

        # initialize the sampling_parameters
        sampling_cfg = cfg.generate.get("sampling", {})
        assert not sampling_cfg.get("logits_processors", None), (
            "logits_processors is not supported yet."
        )
        sampling_params.append(SamplingParams(**sampling_cfg))

    # Step 6: build vllm inputs
    inputs = [{"prompt_token_ids": p_ids} for p_ids in positive_input_ids]
    print(f"=> {inputs=}")
    print(f"=> {sampling_params=}")

    # Step 7: generate outputs
    outputs = llm.generate(inputs, sampling_params)
    for idx_i, out in enumerate(outputs):
        output = torch.tensor(
            list(out.outputs[0].token_ids),
            dtype=pos_inputs.input_ids.dtype,
            device=pos_inputs.input_ids.device,
        )
        answer = emu3_processor.decode(output, skip_special_tokens=True)
        print(f"Answer_{idx_i}:\n {answer}\n")


if __name__ == "__main__":
    cfg = parse_config()
    mode = cfg.generate.get("mode", None)
    print(f"[vllm.__file__] {vllm.__file__}")
    if mode == "G":
        inference_t2i(cfg)
    elif mode == "U":
        inference_i2t(cfg)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Options: 'G'(generation) and 'U'(understanding)."
        )
