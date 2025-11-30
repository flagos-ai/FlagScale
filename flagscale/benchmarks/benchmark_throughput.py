import os
import statistics
import sys
import time

from transformers import AutoTokenizer

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def inference_benchmark(cfg):
    """Benchmark vLLM throughput and latency."""
    # step 1: parse inference config
    prompts = cfg.generate.get("prompts", [])
    assert prompts, "Please set the prompts in the config yaml."

    # step 2: initialize the LLM engine
    llm_cfg = cfg.get("llm", {})
    llm = LLM(**llm_cfg)

    tokenizer_cfg = llm_cfg.get("tokenizer", None)
    if tokenizer_cfg:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg, trust_remote_code=True)
        llm.set_tokenizer(tokenizer)

    # step 3: sampling parameters
    sampling_cfg = cfg.generate.get("sampling", {})
    assert not sampling_cfg.get(
        "logits_processors", None
    ), "logits_processors is not supported yet."
    sampling_params = SamplingParams(**sampling_cfg)
    print(f"=> sampling_params={sampling_params}")

    # step 4: build inputs
    inputs = [{"prompt": prompt} for prompt in prompts]
    print(f"=> inputs={inputs}")

    # step 5: warmup
    print("=> Running warmup...")
    llm.generate(inputs, sampling_params)

    # step 6: benchmark start
    print("=> Running benchmark...")
    N = cfg.generate.get("num_iters", 10)
    latencies = []
    total_tokens = 0

    for i in range(N):
        start = time.time()
        outputs = llm.generate(inputs, sampling_params)
        end = time.time()

        latency = end - start
        latencies.append(latency)

        # count generated tokens
        for output in outputs:
            gen_len = len(output.outputs[0].token_ids)
            total_tokens += gen_len

        print(f"[Iter {i}] latency={latency:.3f}s")

    # step 7: stats
    if not latencies:
        print("No benchmark iterations were run. Skipping stats.")
        return
    avg_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    print("#" * 60)
    print("Benchmark Results")
    print(f"Total iters: {N}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"P50 latency: {p50:.3f}s")
    print("#" * 60)


if __name__ == "__main__":
    cfg = parse_config()
    inference_benchmark(cfg)
