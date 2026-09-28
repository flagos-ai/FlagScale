from typing import Any

from torch.nn import functional as F

from .adasparse.attn_func import AdaptiveSparseAttention


class WanAdaSpaAttnProcessor:
    """Wrapper that keeps original processor logic and swaps SDPA with AdaSpa."""

    def __init__(self, base_processor: Any = None):
        self.base_processor = base_processor
        self.sparse_attn = AdaptiveSparseAttention()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        rotary_emb=None,
        **kwargs,
    ):
        if self.base_processor is None:
            raise RuntimeError("WanAdaSpaAttnProcessor requires a base_processor instance.")

        original_sdpa = F.scaled_dot_product_attention

        def _patched_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            *extra_args,
            **extra_kwargs,
        ):
            if extra_kwargs.get("enable_gqa", False):
                return original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    *extra_args,
                    **extra_kwargs,
                )
            if extra_kwargs.get("scale", None) is not None:
                return original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    *extra_args,
                    **extra_kwargs,
                )
            return self.sparse_attn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        # Different diffusers versions use different arg names for RoPE.
        if image_rotary_emb is None and rotary_emb is not None:
            image_rotary_emb = rotary_emb

        # Patch multiple possible SDPA call sites:
        # 1) local torch.nn.functional alias used by our code (`F`)
        # 2) globals in the original processor function/module
        #    (some diffusers versions import `scaled_dot_product_attention` directly)
        patched_items: list[tuple[str, Any, Any]] = []

        F.scaled_dot_product_attention = _patched_sdpa
        patched_items.append(("attr", F, original_sdpa))

        base_call = getattr(self.base_processor, "__call__", None)
        base_globals = getattr(base_call, "__globals__", {}) if base_call is not None else {}

        if isinstance(base_globals, dict):
            if "scaled_dot_product_attention" in base_globals:
                old = base_globals["scaled_dot_product_attention"]
                base_globals["scaled_dot_product_attention"] = _patched_sdpa
                patched_items.append(("global_scaled_dot_product_attention", base_globals, old))

            g_f = base_globals.get("F", None)
            if g_f is not None and hasattr(g_f, "scaled_dot_product_attention"):
                old = g_f.scaled_dot_product_attention
                g_f.scaled_dot_product_attention = _patched_sdpa
                patched_items.append(("global_F_attr", g_f, old))

        try:
            try:
                return self.base_processor(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    rotary_emb=rotary_emb,
                    **kwargs,
                )
            except TypeError:
                try:
                    return self.base_processor(
                        attn,
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        image_rotary_emb=image_rotary_emb,
                        **kwargs,
                    )
                except TypeError:
                    return self.base_processor(
                        attn,
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        rotary_emb=rotary_emb,
                        **kwargs,
                    )
        finally:
            # Best-effort restoration to avoid leaking monkey patches.
            for kind, owner, old in reversed(patched_items):
                if kind == "attr":
                    owner.scaled_dot_product_attention = old
                elif kind == "global_scaled_dot_product_attention":
                    owner["scaled_dot_product_attention"] = old
                elif kind == "global_F_attr":
                    owner.scaled_dot_product_attention = old
