# Derived from Krea realtime-video; see NOTICE.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import functools
from .attention import baseline_attention as attention
import math
from .baseline_model import (
    WanModel,
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
from .model import causal_rope_apply, CausalHead


def rope_params_riflex(max_seq_len, dim, theta=10000, k=0, L_test=None):
    assert dim % 2 == 0
    omega = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    if k is not None:
        print("Doing riflex w/ ltest", L_test)
        omega[k - 1] = 0.9 * 2 * torch.pi / L_test
    freqs = torch.outer(torch.arange(max_seq_len), omega)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@functools.lru_cache(maxsize=32)
def get_block_mask(
    device: str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    num_frame_per_block=3,
    local_attn_size=-1,
):
    total_length = num_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length
    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    frame_indices = torch.arange(
        start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device
    )
    for tmp in frame_indices:
        ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
            tmp + frame_seqlen * num_frame_per_block
        )

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        else:
            return (kv_idx < ends[q_idx]) & (
                kv_idx >= ends[q_idx] - local_attn_size * frame_seqlen
            ) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    return block_mask


class CausalWanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-06):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        self.fused_projections = False
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @torch.no_grad()
    def fuse_projections(self):
        if self.fused_projections:
            return
        concatenated_weights = torch.cat(
            [self.q.weight.data, self.k.weight.data, self.v.weight.data]
        )
        concatenated_bias = torch.cat([self.q.bias.data, self.k.bias.data, self.v.bias.data])
        out_features, in_features = concatenated_weights.shape
        with torch.device("meta"):
            self.to_qkv = torch.nn.Linear(in_features, out_features, bias=True)
        self.to_qkv.load_state_dict(
            {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
        )
        self.fused_projections = True

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        b, s, n, d = (*x.shape[:2], self.num_heads, self.head_dim)
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            if self.fused_projections:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q = self.norm_q(q).view(b, s, n, d)
                k = self.norm_k(k).view(b, s, n, d)
                v = v.view(b, s, n, d)
            else:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
            return (q, k, v)

        q, k, v = qkv_fn(x)
        if kv_cache is None or block_mask is not None:
            is_tf = False
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
            local_end_index = roped_key.shape[1]
            kv_cache["k"][:, :local_end_index] = roped_key
            kv_cache["v"][:, :local_end_index] = v
            kv_cache["global_end_index"] = local_end_index
            kv_cache["local_end_index"] = local_end_index
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [
                    roped_query,
                    torch.zeros(
                        [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )
            padded_roped_key = torch.cat(
                [
                    roped_key,
                    torch.zeros(
                        [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                        device=k.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )
            padded_v = torch.cat(
                [
                    v,
                    torch.zeros(
                        [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1).contiguous(),
                key=padded_roped_key.transpose(2, 1).contiguous(),
                value=padded_v.transpose(2, 1).contiguous(),
                block_mask=block_mask,
                kernel_options={"BLOCKS_ARE_CONTIGUOUS": True},
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = 1560
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame
            ).type_as(v)
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if (
                self.local_attn_size != -1
                and current_end > kv_cache["global_end_index"]
                and (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size)
            ):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                local_end_index = (
                    kv_cache["local_end_index"]
                    + current_end
                    - kv_cache["global_end_index"]
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = (
                    kv_cache["local_end_index"] + current_end - kv_cache["global_end_index"]
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][
                    :, max(0, local_end_index - self.max_attention_size) : local_end_index
                ],
                kv_cache["v"][
                    :, max(0, local_end_index - self.max_attention_size) : local_end_index
                ],
            )
            kv_cache["global_end_index"] = current_end
            kv_cache["local_end_index"] = local_end_index
        x = x.flatten(2)
        x = self.o(x)
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
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps
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
    ):
        num_frames, frame_seqlen = (e.shape[1], x.shape[1] // e.shape[1])
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
            ).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
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


class CausalWanModel(WanModel):
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
    ):
        ModelMixin.__init__(self)
        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                self.attention_block_cls(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    **self._attention_block_kwargs(),
                )
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )
        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)
        self.init_weights()
        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        block_mask = get_block_mask(
            str(device), num_frames, frame_seqlen, num_frame_per_block, local_attn_size
        )
        return block_mask

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
        )
        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                }
            )
            x = block(x, **kwargs)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)

    def _attention_block_kwargs(self):
        return {}

    def _prepare_inputs(self, x, t, context, seq_len, clip_fea=None, y=None):
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
            )
        )
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
        return (x, e, e0, seq_lens, grid_sizes, context)
