# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Strict checkpoint loading shared by both autoregressive pipelines."""
from accelerate import init_empty_weights
from safetensors import safe_open
from ..model_spec import model_config, validate_krea_checkpoint

def load_krea_model(checkpoint, model_cls, *, dtype, **model_kwargs):
    """Load all distilled tensors strictly without loading the 57 GB teacher first."""
    validate_krea_checkpoint(checkpoint)
    with init_empty_weights():
        model = model_cls(**model_config("krea-14b"), **model_kwargs)
    expected = model.state_dict()
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        actual = {key.removeprefix("model.") for key in tensors.keys()}
        if actual != set(expected):
            raise ValueError(
                f"Krea checkpoint keys mismatch: missing={sorted(set(expected) - actual)}, "
                f"unexpected={sorted(actual - set(expected))}"
            )
        # Validate EVERY shape before reading tens of gigabytes of tensor data.
        for name, tensor in expected.items():
            shape = tuple(tensors.get_slice(f"model.{name}").get_shape())
            if shape != tuple(tensor.shape):
                raise ValueError(f"Krea checkpoint shape mismatch for {name}: {shape}")
        weights = {name: tensors.get_tensor(f"model.{name}").to(dtype=dtype) for name in expected}
    model.load_state_dict(weights, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Krea loading left uninitialized parameters")
    return model.eval().requires_grad_(False)
