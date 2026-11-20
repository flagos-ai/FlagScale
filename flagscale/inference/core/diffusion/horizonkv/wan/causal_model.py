# Modified for the HorizonKV inference artifact; see NOTICE.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from .baseline_causal_model import CausalWanModel as DenseCausalWanModel
from flagscale.inference.core.diffusion.horizonkv.wan.model import (
    WanRMSNorm,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
)
from diffusers.configuration_utils import register_to_config
import torch.nn as nn
import torch
import math
from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import (
    copy_kv_to_cpu_cache_async,
    queue_deferred_clean_kv_to_cpu_cache,
    sparse_kv_self_attention,
    update_gpu_kv_cache_frames,
)


def _is_clean_kv_update_timestep(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool((value.detach().abs() <= 1e-06).all().item())
    if isinstance(value, (list, tuple)):
        return bool(value) and all((_is_clean_kv_update_timestep(v) for v in value))
    if isinstance(value, (int, float)):
        return abs(float(value)) <= 1e-06
    try:
        return abs(float(value)) <= 1e-06
    except (TypeError, ValueError):
        return False


def _mean_pool_along_frames(x: torch.Tensor, frame_seqlen: int) -> torch.Tensor:
    if frame_seqlen <= 0:
        raise ValueError("frame_seqlen must be positive")
    if x.shape[1] == 0:
        return x[:, :0]
    batch, length, heads, dim = x.shape
    n_full = length // frame_seqlen
    rest = length - n_full * frame_seqlen
    parts = []
    if n_full > 0:
        parts.append(
            x[:, : n_full * frame_seqlen]
            .contiguous()
            .reshape(batch, n_full, frame_seqlen, heads, dim)
            .mean(dim=2)
        )
    if rest > 0:
        parts.append(x[:, n_full * frame_seqlen :].mean(dim=1, keepdim=True))
    return torch.cat(parts, dim=1)


class CausalWanSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        eps=1e-06,
        long_video_gen=False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.long_video_gen = bool(long_video_gen)
        self.qk_norm = qk_norm
        self.eps = eps
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        self._cached_frame_seqlen: int | None = None
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        layer_idx=None,
        kv_sparse_config=None,
        num_frame_per_block: int = 1,
        kv_sparse_timestep=None,
        next_kv_cache=None,
        kv_cache_update: bool | None = None,
    ):
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)
        if cache_start is None:
            cache_start = current_start

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        if torch.is_grad_enabled():
            raise RuntimeError("HorizonKV CPU KV offload supports no-grad inference only")
        frame_seqlen = kv_cache.get("frame_seqlen", None)
        if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
            frame_seqlen = self._cached_frame_seqlen
        if not isinstance(frame_seqlen, int) or frame_seqlen <= 0:
            frame_seqlen = int(math.prod(grid_sizes[0][1:].detach().cpu().tolist()))
            self._cached_frame_seqlen = frame_seqlen
        current_end = current_start + q.shape[1]
        cache_end = cache_start + q.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        local_start_index = int(cache_start)
        local_end_index = int(cache_end)
        cache_capacity = int(kv_cache["token_capacity"])
        if local_start_index < 0 or local_end_index > cache_capacity:
            raise ValueError(
                f"CPU offloaded KV cache capacity is too small: span=[{local_start_index}, {local_end_index}) capacity={cache_capacity}"
            )
        write_clean_kv = (
            bool(kv_cache_update)
            if kv_cache_update is not None
            else _is_clean_kv_update_timestep(kv_sparse_timestep)
        )
        new_frame_k_mean = _mean_pool_along_frames(k.type_as(v), frame_seqlen).type_as(v)
        local_end_frame = int(math.ceil(local_end_index / float(frame_seqlen)))
        local_start_frame = local_end_frame - int(new_frame_k_mean.shape[1])
        if write_clean_kv:
            queued_clean_d2h = queue_deferred_clean_kv_to_cpu_cache(
                kv_cache,
                next_kv_cache,
                local_start_index=local_start_index,
                local_end_index=local_end_index,
                key=k,
                value=v,
            )
            if not queued_clean_d2h:
                copy_kv_to_cpu_cache_async(
                    kv_cache,
                    local_start_index=local_start_index,
                    local_end_index=local_end_index,
                    key=k,
                    value=v,
                )
            kv_cache["frame_k_mean"][:, local_start_frame:local_end_frame] = new_frame_k_mean
        x = sparse_kv_self_attention(
            roped_query=q,
            frame_seqlen=frame_seqlen,
            kv_start=0,
            sink_tokens=sink_tokens,
            current_start=int(current_start),
            current_end=int(current_end),
            local_end_index=int(local_end_index),
            num_frame_per_block=int(num_frame_per_block),
            layer_idx=int(layer_idx),
            kv_sparse_config=kv_sparse_config,
            out_dtype=q.dtype,
            rope_freqs=freqs,
            rope_grid_sizes=grid_sizes,
            kv_cache=kv_cache,
            live_k=k,
            live_v=v,
            live_frame_k_mean=None if write_clean_kv else new_frame_k_mean,
            live_start_index=local_start_index,
        )
        if x is None:
            raise RuntimeError(
                "CPU KV sparse overlap path returned no attention output; check the FlashAttention block-sparse backend build and overlap flags"
            )
        x = self.o(x.flatten(2))
        if write_clean_kv:
            update_gpu_kv_cache_frames(
                kv_cache,
                start_frame=local_start_frame,
                end_frame=local_end_frame,
                source_start_index=local_start_index,
                current_start=int(current_start),
                frame_seqlen=frame_seqlen,
                num_frame_per_block=int(num_frame_per_block),
                frame_k=k.detach(),
                frame_v=v.detach(),
            )
        kv_cache["global_end_index"].fill_(int(current_end))
        kv_cache["local_end_index"].fill_(local_end_index)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-06,
        long_video_gen=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, long_video_gen=long_video_gen
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        layer_idx=None,
        kv_sparse_config=None,
        num_frame_per_block: int = 1,
        kv_sparse_timestep=None,
        next_kv_cache=None,
        kv_cache_update: bool | None = None,
    ):
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        self_attn_result = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            ).flatten(1, 2),
            grid_sizes=grid_sizes,
            freqs=freqs,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=cache_start,
            layer_idx=layer_idx,
            kv_sparse_config=kv_sparse_config,
            num_frame_per_block=num_frame_per_block,
            kv_sparse_timestep=kv_sparse_timestep,
            next_kv_cache=next_kv_cache,
            kv_cache_update=kv_cache_update,
        )
        y = self_attn_result
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                crossattn_cache=crossattn_cache,
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4])
                    + e[3]
                ).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalWanModel(DenseCausalWanModel):
    attention_block_cls = CausalWanAttentionBlock
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-06,
        long_video_gen=False,
    ):
        self.long_video_gen = bool(long_video_gen)
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.kv_sparse_config = None

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        kv_cache_update: bool | None = None,
    ):
        x, e, e0, seq_lens, grid_sizes, context = self._prepare_inputs(
            x, t, context, seq_len, clip_fea, y
        )
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=None,
            block_mask=self.block_mask,
            kv_sparse_config=getattr(self, "kv_sparse_config", None),
            num_frame_per_block=int(getattr(self, "num_frame_per_block", 1)),
            kv_sparse_timestep=t,
            kv_cache_update=kv_cache_update,
        )
        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "layer_idx": block_index,
                    "next_kv_cache": (
                        kv_cache[block_index + 1]
                        if kv_cache is not None and block_index + 1 < len(kv_cache)
                        else None
                    ),
                }
            )
            x = block(x, **kwargs)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)

    def _attention_block_kwargs(self):
        return {"long_video_gen": self.long_video_gen}
