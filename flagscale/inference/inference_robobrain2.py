from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def build_multimodal_prompt(question: str, modality: str = "image"):
    """
    Build a multimodal prompt with placeholders for image/video.
    """
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def clean_user_prompt(prompt: str):
    """
    Extract only the user's original input text from the full prompt.
    """
    start_token = "<|vision_end|>"
    end_token = "<|im_end|>"
    start_idx = prompt.find(start_token)
    if start_idx != -1:
        start_idx += len(start_token)
        end_idx = prompt.find(end_token, start_idx)
        if end_idx == -1:
            end_idx = len(prompt)
        user_text = prompt[start_idx:end_idx].strip()
        return user_text
        return prompt.strip()


def clean_text(text: str):
    """
    Remove placeholders or template tags from model output.
    """
    tags = [
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ]
    for tag in tags:
        text = text.replace(tag, "")
    return text.strip()


def inference(cfg):
    # Initialize LLM
    llm_cfg = cfg.get("llm", {})
    llm = LLM(**llm_cfg)

    # Initialize tokenizer if needed
    tokenizer_cfg = llm_cfg.get("tokenizer", None)
    if tokenizer_cfg:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg, trust_remote_code=True)
        llm.set_tokenizer(tokenizer)

    # Initialize processor if needed
    # processor_cfg = llm_cfg.get("processor", None)
    # processor = (
    #     AutoProcessor.from_pretrained(processor_cfg, trust_remote_code=True)
    #     if processor_cfg
    #     else None
    # )

    # Load prompts and multimodal data
    questions = cfg.generate.get("prompts", [])
    mm_data_paths = cfg.generate.get("mm_data", [])
    modality = cfg.generate.get("modality", "image")
    assert questions and len(questions) == len(mm_data_paths), (
        "Prompts and mm_data must match in length"
    )

    # Build inputs
    inputs = []
    for idx, question in enumerate(questions):
        prompt = build_multimodal_prompt(question, modality)
        item = {"prompt": prompt}

        if modality == "image":
            img = Image.open(mm_data_paths[idx]).convert("RGB")
            item["multi_modal_data"] = {"image": [img]}
        elif modality == "video":
            item["multi_modal_data"] = {"video": [mm_data_paths[idx]]}

        inputs.append(item)

    # Sampling parameters
    sampling_cfg = cfg.generate.get("sampling", {})
    sampling_params = SamplingParams(**sampling_cfg)

    # Generate
    outputs = llm.generate(inputs, sampling_params)

    # Print clean outputs
    for output in outputs:
        print("*" * 50)
        clean_prompt = clean_user_prompt(output.prompt)
        clean_output = clean_text(output.outputs[0].text) if output.outputs else ""
        print(f"output.prompt='{clean_prompt}'")
        print(f"output.outputs[0].text='{clean_output}'")
        if output.outputs:
            print(f"output.outputs[0].token_ids={output.outputs[0].token_ids}")
        else:
            print("No output generated")
    print("#" * 50)


if __name__ == "__main__":
    cfg = parse_config()
    inference(cfg)
