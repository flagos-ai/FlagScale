import os

import torch
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

from flagscale.compress.adapter import LLMCompressorAdapter
from flagscale.inference.processing_emu3 import (
    Emu3Processor,
)


def test_llmcompressor_adapter_without_dataset():
    model_path = "BAAI/Emu3-Gen"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    quant_args = {"targets": ["Linear"], "scheme": "FP8_DYNAMIC", "ignore": ["lm_head"]}
    adapter = LLMCompressorAdapter(model=model, **quant_args)
    adapter.model.save_pretrained("test_output", save_compressed=True)
    os.remove("test_output")


def test_llmcompressor_adapter_with_dataset():
    EMU_HUB = "BAAI/Emu3-Gen"
    VQ_HUB = "BAAI/Emu3-VisionTokenizer"
    # prepare model and processor
    model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        VQ_HUB, device_map="cuda:0", trust_remote_code=True
    ).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    # prepare input
    POSITIVE_PROMPT = " masterpiece, film grained, best quality."
    # NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    prompt = ["a portrait of young girl.", "a shiba inu"]
    prompt = [p + POSITIVE_PROMPT for p in prompt]
    kwargs = dict(
        mode="G",
        ratio=["1:1", "16:9"],
        image_area=model.config.image_area,
        return_tensors="pt",
        padding="longest",
    )
    pos_inputs = processor(text=prompt, **kwargs)
    # neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)
    quant_args = {
        "targets": ["Linear"],
        "scheme": "W4A16",
        "ignore": ["lm_head"],
        "algo": {"gptq": {"blocksize": 128, "percdamp": 0.01}},
        "dataset": pos_inputs.input_ids.to("cuda:0"),
    }
    adapter = LLMCompressorAdapter(model=model, **quant_args)
    adapter.model.save_pretrained("test_output", save_compressed=True)
    os.remove("test_output")
