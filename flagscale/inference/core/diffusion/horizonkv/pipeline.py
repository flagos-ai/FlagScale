# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagScale protocol adapter for the native Wan/HorizonKV pipeline."""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from .model_spec import denoising_steps, model_config, validate_krea_checkpoint
from .runtime import HorizonKVRuntime, check_environment


@dataclass
class HorizonKVOutput:
    frames: list
    latents: torch.Tensor | None
    metadata: dict


def _concrete_device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


@contextmanager
def generator_scope(generator, device):
    """Bridge native randn_like to a request Generator, restoring global RNG state."""
    device = _concrete_device(device)
    generator = (
        generator if generator is not None else torch.Generator(device=device).manual_seed(42)
    )
    if _concrete_device(generator.device) != device:
        raise ValueError(f"Generator device {generator.device} must match pipeline device {device}")
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        if device.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), device)
        else:
            torch.set_rng_state(generator.get_state())
        try:
            yield generator
        finally:
            state = (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
            )
            generator.set_state(state)


class HorizonKVPipeline(nn.Module):
    def __init__(self, native_pipeline, runtime):
        super().__init__()
        self.native_pipeline = native_pipeline
        self.runtime = runtime
        self.transformer._flagscale_horizonkv_runtime = runtime

    @property
    def transformer(self):
        return self.native_pipeline.generator.model

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        checkpoint,
        torch_dtype=torch.bfloat16,
        horizonkv=None,
        model_variant="krea-14b",
    ):
        if torch_dtype != torch.bfloat16:
            raise ValueError("The HorizonKV integration supports torch_dtype=bfloat16")
        geometry = model_config(model_variant)
        runtime_options = dict(horizonkv or {})
        if model_variant == "krea-14b":
            for key, value in dict(cache=15, window=180, fixed=False).items():
                runtime_options.setdefault(key, value)
        runtime = HorizonKVRuntime(
            assets_root=pretrained_model_name_or_path,
            checkpoint=checkpoint,
            **runtime_options,
        )
        runtime.plan["model_variant"] = model_variant
        runtime.plan["model_geometry"] = geometry
        runtime.plan["sampling_profile"] = (
            "krea-five-step-global-prerope-v1"
            if model_variant == "krea-14b"
            else "self-forcing-four-step"
        )
        try:
            with runtime.activated():
                check_environment(require_cuda=True)
                cls._validate_assets(runtime, model_variant)
                from .native import (
                    LoadedWanDiffusionWrapper,
                    SizedCausalInferencePipeline,
                    load_krea_model,
                )

                plan = runtime.plan
                args = OmegaConf.create(
                    dict(
                        denoising_step_list=denoising_steps(model_variant),
                        warp_denoising_step=True,
                        num_frame_per_block=3,
                        context_noise=0,
                        kv_sparse_config="",
                        model_kwargs=dict(
                            model_name=(
                                "Wan2.1-T2V-14B"
                                if model_variant == "krea-14b"
                                else "Wan2.1-T2V-1.3B"
                            ),
                            timestep_shift=5.0,
                            local_attn_size=plan["local_attn_size"],
                            sink_size=plan["sink"],
                            long_video_gen=plan["long_video_gen"],
                        ),
                    )
                )
                generator = None
                if model_variant == "krea-14b":
                    generator = LoadedWanDiffusionWrapper(
                        load_krea_model(
                            plan["checkpoint"],
                            dtype=torch_dtype,
                            local_attn_size=plan["local_attn_size"],
                            sink_size=plan["sink"],
                            long_video_gen=plan["long_video_gen"],
                        )
                    )
                native = SizedCausalInferencePipeline(
                    args,
                    device=torch.device("cuda"),
                    generator=generator,
                )
                if model_variant == "self-forcing-1.3b":
                    state = torch.load(plan["checkpoint"], map_location="cpu", weights_only=True)
                    if "generator_ema" not in state:
                        raise ValueError("Expected a distilled checkpoint containing generator_ema")
                    weights = {
                        k.replace("_fsdp_wrapped_module.", "").replace(
                            "_checkpoint_wrapped_module.", ""
                        ): v
                        for k, v in state["generator_ema"].items()
                    }
                    native.generator.load_state_dict(weights, strict=True)
                native.eval().requires_grad_(False).to(dtype=torch_dtype)
                return cls(native, runtime)
        except BaseException:
            runtime.close()
            raise

    @staticmethod
    def _validate_assets(runtime, model_variant="self-forcing-1.3b"):
        model_config(model_variant)
        root = Path(runtime.env["HORIZONKV_ROOT"]) / "wan_models" / "Wan2.1-T2V-1.3B"
        for path in (
            Path(runtime.plan["checkpoint"]),
            root / "Wan2.1_VAE.pth",
            root / "models_t5_umt5-xxl-enc-bf16.pth",
            root / "google/umt5-xxl",
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing native Wan asset: {path}")
        if model_variant == "krea-14b":
            return validate_krea_checkpoint(runtime.plan["checkpoint"])
        for path in (root / "config.json", root / "diffusion_pytorch_model.safetensors"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing native Wan asset: {path}")
        config = json.loads((root / "config.json").read_text())
        if (config.get("num_layers"), config.get("num_heads"), config.get("dim")) != (
            30,
            12,
            1536,
        ):
            raise ValueError("self-forcing-1.3b requires the native 30-layer / 12-head backbone")

    def reset_request(self):
        """Drain native transfers before reusing storage; clear conditioning and VAE state."""
        from flagscale.inference.core.diffusion.horizonkv.algorithm.oae import reset_kv_cache_state

        for cache in self.native_pipeline.kv_cache1 or []:
            reset_kv_cache_state(cache)
        for cache in self.native_pipeline.crossattn_cache or []:
            cache["is_init"] = False
        self.native_pipeline.vae.model.clear_cache()

    def to(self, *args, **kwargs):
        if self.native_pipeline.kv_cache1 is not None:
            with self.runtime.activated():
                self.native_pipeline._release_cache_before_rebuild()
        return super().to(*args, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        prompt,
        *,
        latent_frames=24,
        height=480,
        width=832,
        generator=None,
        return_dict=True,
        output_type="np",
        profile=True,
        return_latents=False,
    ):
        if isinstance(prompt, list | tuple):
            if len(prompt) != 1:
                raise ValueError(
                    "HorizonKV supports one prompt per request; use generate.prompts for sequential runs"
                )
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if type(latent_frames) is not int or latent_frames <= 0 or latent_frames % 3:
            raise ValueError("latent_frames must be a positive multiple of 3")
        if any(type(value) is not int or value <= 0 or value % 16 for value in (height, width)):
            raise ValueError("height and width must be positive multiples of 16")
        if output_type not in ("np", "latent"):
            raise ValueError("output_type must be np or latent")
        if not self.runtime.installed:
            raise RuntimeError(
                "Configure HorizonKVTransformation to initialize the causal model backend"
            )
        if self.device.type != "cuda":
            raise ValueError("HorizonKV generation requires a CUDA pipeline device")
        with (
            self.runtime.activated(),
            torch.cuda.device(self.device),
            generator_scope(generator, self.device) as rng,
        ):
            noise = torch.randn(
                (1, latent_frames, 16, height // 8, width // 8),
                device=self.device,
                dtype=torch.bfloat16,
            )
            is_krea = self.runtime.plan["model_variant"] == "krea-14b"
            try:
                video, latents = self.native_pipeline.inference(
                    noise=noise,
                    text_prompts=[prompt],
                    return_latents=True,
                    decode=output_type != "latent",
                    profile=profile,
                )
                metadata = dict(self.native_pipeline.last_profile or {})
                first_cache = self.native_pipeline.kv_cache1[0]
                metadata.update(
                    seed=rng.initial_seed(),
                    latent_frames=latent_frames,
                    pixel_frames=4 * latent_frames - 3,
                    fps=24 if is_krea else 16,
                    model_variant=self.runtime.plan["model_variant"],
                    model_geometry=self.runtime.plan["model_geometry"],
                    sampling_profile=self.runtime.plan["sampling_profile"],
                    denoising_timesteps=self.native_pipeline.denoising_step_list.tolist(),
                    cache_layers=len(self.native_pipeline.kv_cache1),
                    cache_heads=self.transformer.num_heads,
                    cache_mode=first_cache.get("cache_mode", "dense"),
                    host_frame_capacity=(
                        first_cache.get("frame_capacity")
                        if first_cache.get("cpu_offload_kv")
                        else None
                    ),
                    gpu_cache_frames=first_cache.get("gpu_cache_frames"),
                    denoising_forwards=len(self.native_pipeline.denoising_step_list)
                    * (latent_frames // 3),
                    clean_current_commit_forwards=latent_frames // 3,
                    historical_recompute_forwards=0,
                    text_encoder_device=str(
                        next(self.native_pipeline.text_encoder.parameters()).device
                    ),
                    text_encoder_offload=False,
                    release_kv_before_decode=False,
                    vae_decode_path="native",
                )
                frames = (
                    []
                    if video is None
                    else [video[0].permute(0, 2, 3, 1).float().cpu().contiguous().numpy()]
                )
                output = HorizonKVOutput(
                    frames,
                    latents.detach().cpu() if return_latents or output_type == "latent" else None,
                    metadata,
                )
            finally:
                self.reset_request()
        return output if return_dict else (output.frames, output.latents)

    def close(self):
        with self.runtime.activated():
            self.reset_request()
            self.native_pipeline._release_cache_before_rebuild()
        self.runtime.close()
