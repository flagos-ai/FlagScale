# Adapted from Krea realtime-video; see NOTICE and licenses/CC-BY-NC-SA-4.0.md.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import logging

import torch

from .wan.scheduler import FlowMatchScheduler

LOGGER = logging.getLogger(__name__)


class CausalInferencePipeline(torch.nn.Module):
    """Dense KV storage used by the migrated Krea GenerationSession."""

    def __init__(self, generator, text_encoder, vae):
        super().__init__()
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = generator.get_scheduler()
        self.num_transformer_blocks = len(generator.model.blocks)
        self.num_frame_per_block = 3
        self.frame_seq_length = 1560
        self.local_attn_size = generator.model.local_attn_size
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.generator.model.num_frame_per_block = 3

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        num_heads = self.generator.model.config.num_heads
        dim = self.generator.model.config.dim
        k_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]
        v_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]

        if (
            self.kv_cache1
            and list(self.kv_cache1[0]["k"].shape) == k_shape
            and list(self.kv_cache1[0]["v"].shape) == v_shape
        ):
            LOGGER.debug("Reinitializing KV cache in place")
            for i in range(self.num_transformer_blocks):
                self.kv_cache1[i]["k"].zero_()
                self.kv_cache1[i]["v"].zero_()
                self.kv_cache1[i]["global_end_index"] = 0
                self.kv_cache1[i]["local_end_index"] = 0
        else:
            LOGGER.debug("Initializing KV cache with shape %s", k_shape)
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append(
                    {
                        "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                        "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                        "global_end_index": 0,
                        "local_end_index": 0,
                    }
                )
            self.k_shape = k_shape
            self.v_shape = v_shape
            self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        num_heads = self.generator.model.config.num_heads
        dim = self.generator.model.config.dim
        k_shape = [batch_size, 512, num_heads, dim // num_heads]
        v_shape = [batch_size, 512, num_heads, dim // num_heads]
        if (
            self.crossattn_cache
            and list(self.crossattn_cache[0]["k"].shape) == k_shape
            and list(self.crossattn_cache[0]["v"].shape) == v_shape
        ):
            LOGGER.debug("Reinitializing cross-attention cache in place")
            for i in range(self.num_transformer_blocks):
                self.crossattn_cache[i]["k"].zero_()
                self.crossattn_cache[i]["v"].zero_()
                self.crossattn_cache[i]["is_init"] = False
        else:
            for _ in range(self.num_transformer_blocks):
                crossattn_cache.append(
                    {
                        "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                        "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                        "is_init": False,
                    }
                )
            self.crossattn_cache = crossattn_cache


def get_denoising_schedule(timesteps, denoising_strength, steps=5):
    indices = torch.linspace(
        denoising_strength * 1000, 0, steps, dtype=torch.float32, device=timesteps.device
    ).to(torch.long)
    return timesteps[1000 - indices]


def encode_video_latent(
    vae, encode_vae_cache, *, frames, height, width, stream=False, dtype=torch.float16
):
    """Upstream tensor-input VAE path, used to reconstruct the boundary latent."""
    frames = (
        torch.nn.functional.interpolate(
            frames.cuda(), size=(height // 8 * 8, width // 8 * 8), mode="bicubic"
        )
        .transpose(0, 1)
        .to(dtype)
    )
    latents, cache = vae(frames.unsqueeze(0), encode_vae_cache, stream=stream)
    return latents.squeeze(0).to(dtype), cache


@dataclass
class GenerateParams:
    prompt: str
    width: int = 832
    height: int = 480
    seed: int = 42
    keep_first_frame: bool = False
    kv_cache_num_frames: int = 3
    num_blocks: int = 9
    num_denoising_steps: int = 5
    timestep_shift: float = 5.0


class GenerationSession:
    """Native Krea text-to-video session, with request-local streaming VAE state.

    Kept separate from HorizonKV: each block resets dense KV and recomputes the
    clean context at window-relative positions before five denoising forwards.
    """

    @torch.inference_mode()
    def __init__(self, params, *, models, frame_callback):
        self.params = params
        self.models = models
        self.frame_callback = frame_callback
        self.gpu = torch.cuda.current_device()
        self.block_idx = self.current_start_frame = self.total_frames_sent = 0
        self.num_blocks = params.num_blocks
        self.num_frame_per_block = 3
        self.kv_cache_num_frames = params.kv_cache_num_frames
        self.current_prompt_embeds = None
        self.disposed = threading.Event()
        self.frame_context_cache = deque(maxlen=1 + (params.kv_cache_num_frames - 1) * 4)
        self.decode_vae_cache = [None] * 55
        self.rnd = torch.Generator(self.gpu).manual_seed(params.seed)
        shape = [1, params.num_blocks * 3, 16, params.height // 8, params.width // 8]
        self.all_latents = torch.zeros(shape, device=self.gpu, dtype=torch.bfloat16).contiguous()
        self.noise = torch.randn(
            shape, device=self.gpu, dtype=torch.bfloat16, generator=self.rnd
        ).contiguous()
        self.init_models(models, params)
        self.denoising_step_list = get_denoising_schedule(
            self.zero_padded_timesteps, 1.0, steps=params.num_denoising_steps
        )

    def dispose(self):
        self.disposed.set()

    def init_models(self, models, params):
        attn_size = self.params.kv_cache_num_frames + models.pipeline.num_frame_per_block
        models.pipeline.local_attn_size = attn_size
        for block in models.pipeline.generator.model.blocks:
            block.self_attn.local_attn_size = -1
        models.pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=self.gpu)
        models.pipeline._initialize_crossattn_cache(
            batch_size=1, dtype=torch.bfloat16, device=self.gpu
        )
        models.pipeline.generator.model.block_mask = None

        # this prevents a cuda sync
        models.pipeline.scheduler = FlowMatchScheduler(
            shift=params.timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        models.pipeline.scheduler.set_timesteps(1000, training=True)

        st = models.pipeline.scheduler.timesteps
        self.zero_padded_timesteps = torch.cat(
            (st.cpu(), torch.tensor([0], dtype=torch.float32))
        ).to(torch.cuda.current_device())

    def get_clean_context_frames(self, models):
        current_kv_cache_num_frames = (
            self.kv_cache_num_frames
            if self.kv_cache_num_frames is not None
            else self.params.kv_cache_num_frames
        )
        clean_context_frames = self.all_latents[:, : self.current_start_frame]
        if (
            self.params.keep_first_frame
            or (self.block_idx - 1) * models.pipeline.num_frame_per_block
            < current_kv_cache_num_frames
        ):
            if current_kv_cache_num_frames == 1:
                clean_context_frames = clean_context_frames[:, :1]
            else:
                clean_context_frames = torch.cat(
                    (
                        clean_context_frames[:, :1],
                        clean_context_frames[:, 1:][:, -current_kv_cache_num_frames + 1 :],
                    ),
                    dim=1,
                )
        else:
            clean_context_frames = clean_context_frames[:, 1:][
                :, -current_kv_cache_num_frames + 1 :
            ]
            first_frame_latent = encode_video_latent(
                models.vae_encoder,
                [None] * 55,
                frames=self.frame_context_cache[0][0].half(),
                height=480,
                width=832,
                stream=False,
            )[0].transpose(0, 1)[None]
            clean_context_frames = torch.cat((first_frame_latent, clean_context_frames), dim=1).to(
                self.all_latents
            )
        return clean_context_frames

    def recompute_kv_cache(self, models):
        if self.block_idx == 0:
            models.pipeline._initialize_kv_cache(
                batch_size=1, dtype=torch.bfloat16, device=self.gpu
            )
            return self.current_start_frame

        for block in models.pipeline.generator.model.blocks:
            block.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block

        current_kv_cache_num_frames = self.params.kv_cache_num_frames

        model_input_start_frame = min(self.current_start_frame, current_kv_cache_num_frames)

        clean_context_frames = self.get_clean_context_frames(models)

        models.pipeline._initialize_kv_cache(
            batch_size=clean_context_frames.shape[0],
            dtype=clean_context_frames.dtype,
            device=clean_context_frames.device,
        )

        block_mask = models.pipeline.generator.model._prepare_blockwise_causal_attn_mask(
            device=str(clean_context_frames.device),
            num_frames=clean_context_frames.shape[1],
            frame_seqlen=models.pipeline.frame_seq_length,
            num_frame_per_block=models.pipeline.num_frame_per_block,
            local_attn_size=-1,
        )

        context_timestep = (
            torch.ones(
                [clean_context_frames.shape[0], clean_context_frames.shape[1]],
                device=clean_context_frames.device,
                dtype=torch.int64,
            )
            * 0
        )
        models.pipeline.generator.model.block_mask = block_mask
        models.transformer(
            noisy_image_or_video=clean_context_frames,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=models.pipeline.kv_cache1,
            crossattn_cache=models.pipeline.crossattn_cache,
            current_start=model_input_start_frame * models.pipeline.frame_seq_length,
        )
        models.pipeline.generator.model.block_mask = None
        return model_input_start_frame

    @torch.inference_mode()
    def generate_block_internal(self, models):
        idx = self.block_idx
        if idx >= self.num_blocks:
            return None

        if self.current_prompt_embeds is None:
            self.conditional_dict = models.text_encoder(text_prompts=[self.params.prompt])
            for key, value in self.conditional_dict.items():
                self.conditional_dict[key] = value.to(dtype=torch.bfloat16).contiguous()
            self.current_prompt_embeds = self.conditional_dict["prompt_embeds"]

        model_input_start_frame = self.recompute_kv_cache(models)
        assert model_input_start_frame is not None
        frame_ids: list[str | None] = []

        noisy_input = self.noise[
            :,
            self.current_start_frame : self.current_start_frame
            + models.pipeline.num_frame_per_block,
        ]

        # This is set from config
        for index, current_timestep in enumerate(self.denoising_step_list):
            if self.disposed.is_set():
                return
            # Normal initialize timestamp stuff
            timestep = (
                torch.ones(
                    [1, models.pipeline.num_frame_per_block],
                    device=self.noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            self.conditional_dict["prompt_embeds"] = self.current_prompt_embeds

            _, denoised_pred = models.transformer(
                noisy_image_or_video=noisy_input,
                conditional_dict=self.conditional_dict,
                timestep=timestep,
                kv_cache=models.pipeline.kv_cache1,
                crossattn_cache=models.pipeline.crossattn_cache,
                current_start=model_input_start_frame * models.pipeline.frame_seq_length,
            )
            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = models.pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn(
                        *denoised_pred.flatten(0, 1).shape,
                        generator=self.rnd,
                        device=denoised_pred.device,
                        dtype=torch.bfloat16,
                    ),
                    next_timestep
                    * torch.ones(
                        [1 * models.pipeline.num_frame_per_block],
                        device=self.noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        self.all_latents[
            :,
            self.current_start_frame : self.current_start_frame
            + models.pipeline.num_frame_per_block,
        ] = denoised_pred

        pixels, self.decode_vae_cache = models.vae_decoder(
            denoised_pred.half(), *self.decode_vae_cache
        )

        self.frame_context_cache.extend(pixels.split(1, dim=1))
        if idx == 0:
            pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block

        event = torch.cuda.Event()
        event.record()
        self.frame_callback(pixels, frame_ids, event)

        self.current_start_frame += models.pipeline.num_frame_per_block
        self.total_frames_sent += pixels.shape[1]
        self.block_idx += 1

        return pixels
