# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Install the resolved HorizonKV configuration on a native causal backbone."""

from flagscale.transformations.transformation import Transformation


class HorizonKVTransformation(Transformation):
    def apply(self, module):
        runtime = getattr(module, "_flagscale_horizonkv_runtime", None)
        if runtime is None:
            raise ValueError(
                "HorizonKVTransformation requires HorizonKVPipeline and its native causal backbone"
            )
        if runtime.installed:
            raise ValueError("HorizonKVTransformation is already installed")
        with runtime.activated():
            from flagscale.inference.core.diffusion.horizonkv.algorithm.config import (
                attach_kv_sparse_to_model,
            )
            from flagscale.inference.core.diffusion.horizonkv.wan.causal_model import (
                CausalWanModel,
            )

            if not isinstance(module, CausalWanModel):
                raise TypeError(
                    "HorizonKVTransformation currently supports native CausalWanModel only"
                )
            attach_kv_sparse_to_model(module, runtime.plan["kv_config"] or None)
            runtime.installed = True
        return True
