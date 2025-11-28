import math
import os
import unittest

import torch

from flagscale.transformations.diffusion.timestep_embedding_flip_sine_cosine_pass import (
    TimestepEmbeddingFlipSineCosinePass,
)


def my_custom_backend(gm: torch.fx.GraphModule, example_inputs):
    torch._inductor.config.post_grad_custom_post_pass = TimestepEmbeddingFlipSineCosinePass()
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(gm, example_inputs)


# https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/models/embeddings.py#L69
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TestTimestepEmbeddingFlipSineCosinePass(unittest.TestCase):
    def test_timestep_embedding_flip_sine_cosine_pass(self):
        # Force cold start
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

        timeseps = torch.tensor([1000])
        embedding_dim = 80
        emb_gt = get_timestep_embedding(timeseps, embedding_dim, True)

        @torch.compile(backend=my_custom_backend)
        def wrapper():
            emb = get_timestep_embedding(timeseps, embedding_dim, True)
            return emb

        emb = wrapper()

        torch.testing.assert_close(emb, emb_gt)
