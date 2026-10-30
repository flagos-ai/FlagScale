# Adapted from Krea realtime-video; see ../NOTICE and ../licenses/CC-BY-NC-SA-4.0.md.
"""Dense attention specialization used only by the AR efficiency baseline."""
import torch
from . import model as shared
from .model import (
    WanRMSNorm, WanLayerNorm, Head, MLPProj, RegisterTokens,
    rope_apply, rope_params, sinusoidal_embedding_1d,
    WanI2VCrossAttention, WanGanCrossAttention, GanAttentionBlock,
)
from .attention import (
    baseline_flash_attention as flash_attention,
    FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE,
)

from .sage import sageattn_func, SAGEATTN_AVAILABLE

class WanSelfAttention(shared.WanSelfAttention):
    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if SAGEATTN_AVAILABLE:
            # print("Using sageattention in crossattn")
            og_dtype = q.dtype
            q = q.transpose(1, 2).to(og_dtype)
            k = k.transpose(1, 2).to(og_dtype)
            v = v.transpose(1, 2).to(og_dtype)
            x = sageattn_func(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
            )
            x = x.transpose(1, 2).contiguous().to(og_dtype)
        else:
            x = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanT2VCrossAttention(shared.WanT2VCrossAttention):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        if SAGEATTN_AVAILABLE:
            # print("Using sageattention in crossattn")
            og_dtype = q.dtype
            q = q.transpose(1, 2).to(og_dtype)
            k = k.transpose(1, 2).to(og_dtype)
            v = v.transpose(1, 2).to(og_dtype)
            x = sageattn_func(
                q=q,
                k=k,
                v=v,
            )
            x = x.transpose(1, 2).contiguous().to(og_dtype)
        elif FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
            x = flash_attention(q, k, v, k_lens=context_lens)
        else:
            q = q.transpose(1, 2).to(og_dtype)
            k = k.transpose(1, 2).to(og_dtype)
            v = v.transpose(1, 2).to(og_dtype)

            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2).contiguous()

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class WanAttentionBlock(shared.WanAttentionBlock):
    self_attention_cls = WanSelfAttention
    cross_attention_classes = WAN_CROSSATTENTION_CLASSES

class WanModel(shared.WanModel):
    attention_block_cls = WanAttentionBlock

    def _forward_classify(
        self,
        x,
        t,
        context,
        seq_len,
        register_tokens,
        cls_pred_branch,
        clip_fea=None,
        y=None,
    ):
        r"""
        Feature extraction through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of video features with original input shapes [C_block, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, c=self.dim // 4)
        return torch.stack(x)
