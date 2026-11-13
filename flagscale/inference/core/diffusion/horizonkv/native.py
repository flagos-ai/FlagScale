# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Model-size adaptations around the optional native HorizonKV implementation.

Import only inside an activated runtime after checking the native CUDA ABI.
"""

from torch import nn

from flagscale.inference.core.diffusion.horizonkv.wan.backbone import WanDiffusionWrapper
from flagscale.inference.core.diffusion.horizonkv.algorithm.pipeline import CausalInferencePipeline
from flagscale.inference.core.diffusion.horizonkv.wan.scheduler import FlowMatchScheduler
from flagscale.inference.core.diffusion.horizonkv.wan.causal_model import CausalWanModel

from .wan.loading import load_krea_model as load_wan_checkpoint


def load_krea_model(checkpoint, *, dtype, local_attn_size, sink_size, long_video_gen):
    return load_wan_checkpoint(
        checkpoint, CausalWanModel, dtype=dtype, local_attn_size=local_attn_size,
        sink_size=sink_size, long_video_gen=long_video_gen,
    )


class LoadedWanDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.uniform_timestep = False
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()


class SizedCausalInferencePipeline(CausalInferencePipeline):
    """Compatibility name for the geometry-aware causal pipeline.

    Cache allocation is implemented once by :class:`CausalInferencePipeline`.
    Keeping this adapter avoids changing existing configuration paths while
    preventing the 14B and 1.3B implementations from drifting apart.
    """
