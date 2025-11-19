import os
import statistics
import time

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def build_multimodal_prompt(question: str, modality: str = "image"):
    """Build a multimodal prompt with placeholders for image/video."""
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


def inference_benchmark(cfg):
    """Benchmark multimodal vLLM throughput and latency."""
    # Initialize LLM
    llm_cfg = cfg.get("llm", {})
    llm = LLM(**llm_cfg)

    # Tokenizer
    tokenizer_cfg = llm_cfg.get("tokenizer", None)
    if tokenizer_cfg:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg, trust_remote_code=True)
        llm.set_tokenizer(tokenizer)

    # Processor (for images/videos)
    processor_cfg = llm_cfg.get("processor", None)
    processor = (
        AutoProcessor.from_pretrained(processor_cfg, trust_remote_code=True)
        if processor_cfg
        else None
    )

    # Load prompts and multimodal data
    questions = cfg.generate.get("prompts", [])
    mm_data_paths = cfg.generate.get("mm_data", [])
    modality = cfg.generate.get("modality", "image")
    assert questions and len(questions) == len(
        mm_data_paths
    ), "Prompts and mm_data must match in length"

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

    # Warmup
    llm.generate(inputs, sampling_params)

    # Benchmark
    num_iters = cfg.generate.get("num_iters", 10)
    latencies = []
    total_tokens = 0

    for _ in range(num_iters):
        start = time.time()
        outputs = llm.generate(inputs, sampling_params)
        end = time.time()

        latency = end - start
        latencies.append(latency)

        # Count generated tokens (text only)
        for output in outputs:
            if output.outputs:
                total_tokens += len(output.outputs[0].token_ids)

    # Compute stats
    if not latencies:
        print("No benchmark iterations were run. Skipping stats.")
        return
    avg_latency = statistics.mean(latencies)
    p50_latency = statistics.median(latencies)
    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    # Print benchmark summary
    print("#" * 60)
    print("Benchmark Results")
    print(f"Total iters: {num_iters}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"P50 latency: {p50_latency:.3f}s")
    print("#" * 60)


if __name__ == "__main__":
    cfg = parse_config()
    inference_benchmark(cfg)
