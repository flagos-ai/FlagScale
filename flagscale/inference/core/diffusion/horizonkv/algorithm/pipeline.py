# Modified for the HorizonKV inference artifact; see NOTICE.
"""Autoregressive denoising with clean KV commits between generated blocks."""
import gc
import math
import time
from typing import List
import torch
import torch.distributed as dist
from flagscale.inference.core.diffusion.horizonkv.wan.backbone import (
    WanDiffusionWrapper,
    WanTextEncoder,
    WanVAEWrapper,
)
from flagscale.inference.core.diffusion.horizonkv.algorithm.config import attach_kv_sparse_to_model
from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import (
    build_kv_cache_state,
    reset_kv_cache_state,
)


class CausalInferencePipeline(torch.nn.Module):

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        self.generator = (
            WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
            if generator is None
            else generator
        )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560
        self.kv_cache1 = None
        self.last_profile = None
        self._kv_cache_signature = None
        self.crossattn_cache = None
        self._crossattn_cache_signature = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        _kv_sparse = str(getattr(args, "kv_sparse_config", "") or "").strip()
        if _kv_sparse:
            if bool(getattr(args.model_kwargs, "use_infinite_attention", False)):
                raise NotImplementedError(
                    "kv_sparse currently only supports the Self-Forcing sparse causal path (set model_kwargs.use_infinite_attention=false)."
                )
            attach_kv_sparse_to_model(self.generator.model, _kv_sparse)

    def _configure_frame_seq_length(self, latent_height: int, latent_width: int) -> None:
        patch_size = tuple(getattr(self.generator.model, "patch_size", (1, 2, 2)))
        patch_h, patch_w = (int(patch_size[1]), int(patch_size[2]))
        if latent_height % patch_h != 0 or latent_width % patch_w != 0:
            raise ValueError(
                f"Latent height/width must be divisible by transformer patch size {patch_h}x{patch_w}, got {latent_height}x{latent_width}"
            )
        frame_seq_length = latent_height // patch_h * (latent_width // patch_w)
        if frame_seq_length != self.frame_seq_length:
            if self.kv_cache1 is not None:
                self._release_cache_before_rebuild()
            self.frame_seq_length = frame_seq_length
            self.kv_cache1 = None
            self.crossattn_cache = None
            self._kv_cache_signature = None
            self._crossattn_cache_signature = None

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        decode: bool = True,
    ) -> torch.Tensor:
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        self.last_profile = None
        self._configure_frame_seq_length(height, width)
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            init_wall_start = time.perf_counter()
            init_start.record()
        kv_cache_size = num_output_frames * self.frame_seq_length
        self._prepare_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            frame_capacity=num_output_frames,
            kv_cache_size_override=kv_cache_size,
        )
        self._prepare_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)
        if profile:
            init_end.record()
            torch.cuda.synchronize()
            init_wall_time = (time.perf_counter() - init_wall_start) * 1000.0
            init_cuda_time = init_start.elapsed_time(init_end)
            diffusion_wall_start = time.perf_counter()
            diffusion_start.record()
        current_num_frames = self.num_frame_per_block
        for _ in range(num_blocks):
            noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.int64
                    )
                    * current_timestep
                )
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    kv_cache_update=False,
                )
                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                kv_cache_update=True,
            )
            current_start_frame += current_num_frames
        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_wall_time = (time.perf_counter() - diffusion_wall_start) * 1000.0
            diffusion_cuda_time = diffusion_start.elapsed_time(diffusion_end)
        vae_time = 0.0
        if decode:
            if profile:
                torch.cuda.synchronize()
                vae_wall_start = time.perf_counter()
            if getattr(self.args.model_kwargs, "use_infinite_attention", False):
                video = self.vae.decode_to_pixel_chunk(output, use_cache=False)
            else:
                video = self.vae.decode_to_pixel(output, use_cache=False)
            video.mul_(0.5).add_(0.5).clamp_(0, 1)
            if profile:
                torch.cuda.synchronize()
                vae_time = (time.perf_counter() - vae_wall_start) * 1000.0
        else:
            video = None
        if profile:
            self.last_profile = {
                "initialization_ms": float(init_wall_time),
                "initialization_cuda_ms": float(init_cuda_time),
                "diffusion_ms": float(diffusion_wall_time),
                "diffusion_cuda_ms": float(diffusion_cuda_time),
                "vae_ms": float(vae_time),
                "num_latent_frames": int(num_output_frames),
                "latent_height": int(height),
                "latent_width": int(width),
                "frame_seq_length": int(self.frame_seq_length),
                "num_frame_per_block": int(self.num_frame_per_block),
                "decode": bool(decode),
            }
        if return_latents:
            return (video, output)
        else:
            return video

    def _release_cache_before_rebuild(self) -> None:
        old_kv_cache = self.kv_cache1
        if old_kv_cache:
            for cache in old_kv_cache:
                reset_kv_cache_state(cache)
        self.kv_cache1 = None
        self.crossattn_cache = None
        self._kv_cache_signature = None
        self._crossattn_cache_signature = None
        del old_kv_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_kv_cache(
        self, batch_size, dtype, device, frame_capacity, kv_cache_size_override
    ) -> None:
        kv_sparse_config = getattr(self.generator.model, "kv_sparse_config", None)
        num_heads = int(self.generator.model.num_heads)
        model_dim = int(self.generator.model.dim)
        head_dim = model_dim // max(1, num_heads)
        gpu_cache_frames = int(getattr(kv_sparse_config, "gpu_cache_frames", 0) or 0)
        max_kv_frames = int(getattr(kv_sparse_config, "max_kv_frames", 0) or 0)
        signature = (
            int(batch_size),
            dtype,
            torch.device(device),
            int(self.frame_seq_length),
            int(frame_capacity or 0),
            int(kv_cache_size_override or 0),
            True,
            int(num_heads),
            int(head_dim),
            int(gpu_cache_frames),
            int(max_kv_frames),
            "kv_frame",
        )
        if self.kv_cache1 is not None and self._kv_cache_signature == signature:
            for cache in self.kv_cache1:
                reset_kv_cache_state(cache)
            return
        if self.kv_cache1 is not None:
            self._release_cache_before_rebuild()
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            frame_capacity=frame_capacity,
            kv_cache_size_override=kv_cache_size_override,
        )
        self._kv_cache_signature = signature

    def _prepare_crossattn_cache(self, batch_size, dtype, device) -> None:
        signature = (int(batch_size), dtype, torch.device(device), int(self.frame_seq_length))
        if self.crossattn_cache is not None and self._crossattn_cache_signature == signature:
            for cache in self.crossattn_cache:
                cache["is_init"] = False
            return
        if self.crossattn_cache is not None:
            self.crossattn_cache = None
            self._crossattn_cache_signature = None
            gc.collect()
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._crossattn_cache_signature = signature

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        frame_capacity: int | None = None,
        kv_cache_size_override: int | None = None,
    ):
        kv_cache1 = []
        kv_sparse_config = getattr(self.generator.model, "kv_sparse_config", None)
        num_heads = int(self.generator.model.num_heads)
        head_dim = int(self.generator.model.dim) // num_heads
        if frame_capacity is None:
            if kv_cache_size_override is None:
                raise ValueError("CPU KV offload requires frame_capacity")
            frame_capacity = math.ceil(int(kv_cache_size_override) / int(self.frame_seq_length))
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                build_kv_cache_state(
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                    frame_capacity=int(frame_capacity),
                    frame_seqlen=self.frame_seq_length,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kv_sparse_config=kv_sparse_config,
                )
            )
        self.kv_cache1 = kv_cache1
        return

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        model = self.generator.model
        shape = (batch_size, model.text_len, model.num_heads, model.dim // model.num_heads)
        self.crossattn_cache = [
            {
                "k": torch.zeros(shape, dtype=dtype, device=device),
                "v": torch.zeros(shape, dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        if local_attn_size_value == -1:
            target_size = 32760
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
        for module in self.generator.model.modules():
            if hasattr(module, "max_attention_size"):
                module.max_attention_size = target_size
