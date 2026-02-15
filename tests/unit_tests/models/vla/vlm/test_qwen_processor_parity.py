"""End-to-end parity test: PIL pipeline vs tensor pipeline for Qwen processor.

Verifies that building processor inputs via:
  Path A: tensor -> PIL -> messages -> apply_chat_template(tokenize=True)
  Path B: tensor -> processor(text=..., images=..., do_rescale=False)
produces identical input_ids, attention_mask, and pixel_values.

Usage:
    pytest test_qwen_processor_parity.py \
        --model-id /path/to/Qwen3-VL \
        --batch-path /path/to/batch.pt
"""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

IMAGE_KEYS = ["observation.images.image", "observation.images.wrist_image"]


def pytest_addoption(parser):
    parser.addoption("--model-id", required=True, help="Path to Qwen VL model")
    parser.addoption("--batch-path", required=True, help="Path to saved batch .pt file")


@pytest.fixture(scope="session")
def processor(request):
    model_id = request.config.getoption("--model-id")
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = "left"
    return proc


@pytest.fixture(scope="session")
def batch(request):
    batch_path = request.config.getoption("--batch-path")
    return torch.load(batch_path, weights_only=False)


# ── Pipeline implementations ────────────────────────────────────────────


def to_pil_preserve(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _tensor_to_pil_list(batch_tensor: torch.Tensor) -> list:
    if not isinstance(batch_tensor, torch.Tensor):
        return batch_tensor
    if batch_tensor.ndim == 3:
        batch_tensor = batch_tensor.unsqueeze(0)
    pil_images = []
    for item in batch_tensor:
        if item.shape[-1] in (1, 3, 4):
            img = item
        else:
            img = item.permute(1, 2, 0)
        pil_images.append(to_pil_preserve(img.detach().cpu().numpy()))
    return pil_images


def run_path_a(processor, batch: dict) -> dict:
    """Current pipeline: tensor -> PIL -> messages -> apply_chat_template(tokenize=True)"""
    instructions = batch["task"]
    if isinstance(instructions, torch.Tensor):
        instructions = instructions.detach().cpu().tolist()
    if isinstance(instructions, str):
        instructions = [instructions]

    batch_images = None
    for key in IMAGE_KEYS:
        key_images = _tensor_to_pil_list(batch[key])
        if batch_images is None:
            batch_images = [[img] for img in key_images]
        else:
            for sample_images, img in zip(batch_images, key_images):
                sample_images.append(img)

    messages = []
    for imgs, instruction in zip(batch_images, instructions):
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": instruction})
        messages.append([{"role": "user", "content": content}])

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def run_path_b(processor, batch: dict) -> dict:
    """Proposed: tensor images -> processor(text=..., images=...) directly"""
    instructions = batch["task"]
    if isinstance(instructions, torch.Tensor):
        instructions = instructions.detach().cpu().tolist()
    if isinstance(instructions, str):
        instructions = [instructions]

    B = len(instructions)
    num_images = len(IMAGE_KEYS)
    messages = []
    for instruction in instructions:
        content = [{"type": "image", "image": "placeholder"}] * num_images
        content.append({"type": "text", "text": instruction})
        messages.append([{"role": "user", "content": content}])

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    batch_images = []
    for i in range(B):
        sample_imgs = []
        for key in IMAGE_KEYS:
            t = batch[key][i]
            sample_imgs.append(t)
        batch_images.append(sample_imgs)

    return processor(
        text=texts,
        images=batch_images,
        padding=True,
        return_tensors="pt",
        do_rescale=False,
    )


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def results(processor, batch):
    a = run_path_a(processor, batch)
    b = run_path_b(processor, batch)
    return a, b


def test_output_keys_match(results):
    result_a, result_b = results
    assert set(result_a.keys()) == set(result_b.keys())


@pytest.mark.parametrize("key", ["input_ids", "attention_mask"])
def test_integer_tensor_exact_match(results, key):
    result_a, result_b = results
    if key not in result_a:
        pytest.skip(f"'{key}' not in output")
    a, b = result_a[key].cpu(), result_b[key].cpu()
    assert a.shape == b.shape, f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"
    assert torch.equal(a, b), f"value mismatch: {(a != b).sum().item()} elements differ"


def test_pixel_values_exact_match(results):
    result_a, result_b = results
    key = "pixel_values"
    if key not in result_a:
        pytest.skip("'pixel_values' not in output")
    a, b = result_a[key].cpu(), result_b[key].cpu()
    assert a.shape == b.shape, f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"
    diff = (a.float() - b.float()).abs()
    assert torch.equal(a, b), f"max diff={diff.max():.6f}, mean diff={diff.mean():.6f}"


def test_image_grid_thw_match(results):
    result_a, result_b = results
    key = "image_grid_thw"
    if key not in result_a:
        pytest.skip("'image_grid_thw' not in output")
    a, b = result_a[key].cpu(), result_b[key].cpu()
    assert a.shape == b.shape, f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"
    assert torch.equal(a, b)
