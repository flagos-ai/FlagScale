"""AdaSpa plug-and-play integration for FlagScale.

Follows the AdaSpa README approach:
  1. init_sparse_attn_paras(config)
  2. Replace F.scaled_dot_product_attention → adaptive_sparse_attn

The patch is scoped so that only SDPA calls originating from the
transformer backbone are routed through AdaSpa; text-encoder / VAE
SDPA calls pass through to PyTorch's native implementation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig

from flagscale.runner.utils import logger
from flagscale.transformations.transformation import Transformation


class AdaSpaTransformation(Transformation):
    """Enable AdaSpa sparse attention via the plug-and-play approach."""

    def __init__(
        self,
        basic: DictConfig | dict[str, Any] | None = None,
        adaspa: DictConfig | dict[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self._basic = self._to_plain_dict(basic)
        self._adaspa = self._to_plain_dict(adaspa)
        self._strict = strict

    @staticmethod
    def _to_plain_dict(data: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
        if data is None:
            return {}
        if isinstance(data, DictConfig):
            return {k: data.get(k) for k in data}
        return dict(data)

    def apply(self, module: nn.Module) -> bool:
        adaspa_pkg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaspa")
        adaspa_third_party_root = os.path.join(
            adaspa_pkg_root, "third_party", "block_sparse_attention"
        )
        if adaspa_third_party_root not in sys.path:
            sys.path.insert(0, adaspa_third_party_root)

        try:
            from flagscale.inference.core.diffusion.adaspa.adasparse_args import (
                init_sparse_attn_paras,
            )
            from flagscale.inference.core.diffusion.adaspa.adasparse.attn_func import (
                adaptive_sparse_attn,
            )
            import flagscale.inference.core.diffusion.adaspa.adaspa_handler as _handler_mod
        except Exception as exc:
            msg = (
                f"Failed to import vendored adaspa package from {adaspa_pkg_root}. "
                "Ensure block_sparse_attn / triton are installed."
            )
            if self._strict:
                raise RuntimeError(msg) from exc
            logger.warning(msg)
            return False

        # ---- 1. Initialise AdaSpa global parameters ----
        init_sparse_attn_paras({"basic": self._basic, "adaspa": self._adaspa})
        _handler_mod.ADASPA_PROCESSOR = "Wan2.1"

        # ---- 2. Scoped SDPA monkey-patch ----
        _original_sdpa = F.scaled_dot_product_attention
        _adaspa_active = [False]
        _in_adaspa = [False]
        _fallback_warn_count = [0]

        def _is_supported_tensor(x: Any) -> bool:
            return isinstance(x, torch.Tensor) and x.ndim == 4

        def _is_supported_self_attn_call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
            # AdaSpa search/mask path assumes self-attention-like shapes where
            # q/k/v share the same sequence length. Skip cross-attention calls.
            return query.shape[2] == key.shape[2] == value.shape[2]

        def _patched_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            # AdaSpa only supports 4D tensor inputs [B, H, S, D]. For any other
            # SDPA call pattern, fall back to PyTorch native SDPA.
            if (
                not _adaspa_active[0]
                or _in_adaspa[0]
                or enable_gqa
                or not _is_supported_tensor(query)
                or not _is_supported_tensor(key)
                or not _is_supported_tensor(value)
                or not _is_supported_self_attn_call(query, key, value)
            ):
                return _original_sdpa(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            _in_adaspa[0] = True
            try:
                # try:
                return adaptive_sparse_attn(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
                # except Exception as exc:
                #     # Runtime safety net: some diffusers dispatch paths can still
                #     # hit incompatible SDPA argument patterns. Fall back to native
                #     # SDPA so inference continues, while keeping AdaSpa on supported calls.
                #     if _fallback_warn_count[0] < 3:
                #         logger.warning(
                #             f"AdaSpa fallback to native SDPA due to incompatible call: {type(exc).__name__}: {exc}"
                #         )
                #         _fallback_warn_count[0] += 1
                #     return _original_sdpa(
                #         query, key, value,
                #         attn_mask=attn_mask,
                #         dropout_p=dropout_p,
                #         is_causal=is_causal,
                #         scale=scale,
                #         enable_gqa=enable_gqa,
                #     )
            finally:
                _in_adaspa[0] = False

        F.scaled_dot_product_attention = _patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

        # ---- 3. Register forward hooks to scope the patch ----
        def _pre_hook(_module, _args):
            _adaspa_active[0] = True

        def _post_hook(_module, _args, output):
            _adaspa_active[0] = False

        module.register_forward_pre_hook(_pre_hook)
        module.register_forward_hook(_post_hook)

        # ---- 4. Count attention layers for logging ----
        attn_count = 0
        try:
            from diffusers.models.attention import Attention
            for _, m in module.named_modules():
                if isinstance(m, Attention):
                    attn_count += 1
        except ImportError:
            pass

        logger.info(
            f"AdaSpaTransformation (plug-and-play): patched F.scaled_dot_product_attention, "
            f"scoped to transformer backbone ({attn_count} Attention modules found). "
            f"model_name={_handler_mod.get_model_name()}, "
            f"sparsity={self._adaspa.get('sparsity')}, "
            f"search_steps={self._adaspa.get('search_steps')}"
        )
        return True
