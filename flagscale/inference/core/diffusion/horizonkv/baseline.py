# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagScale adapter for the locally migrated Krea dense GenerationSession.

Model and session sources are bundled here; see NOTICE.
"""

import hashlib
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


__all__ = ["ARDiffusionOutput", "ARDiffusionRuntime", "ARDiffusionPipeline"]


@dataclass
class ARDiffusionOutput:
    """Result returned by the dense AR+Diffusion baseline."""

    frames: list
    latents: torch.Tensor | None
    metadata: dict


class ARDiffusionRuntime:
    """Local Krea implementation provenance and baseline configuration."""

    def __init__(self, assets, checkpoint, context_latent_frames=6):
        if (
            type(context_latent_frames) is not int
            or context_latent_frames <= 0
            or context_latent_frames % 3
        ):
            raise ValueError("Krea context_latent_frames must be a positive multiple of 3")
        self.context_latent_frames = context_latent_frames
        self.assets = Path(assets).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()

    def snapshot(self):
        root = Path(__file__).resolve().parent
        files = [
            root / name
            for name in ("baseline.py", "baseline_session.py", "streaming_vae.py", "model_spec.py")
        ]
        files += sorted((root / "wan").glob("*.py"))
        return dict(
            sampling_profile=(
                "krea-default-session-five-step-v1"
                if self.context_latent_frames == 3
                else f"krea-session-context{self.context_latent_frames}-five-step-v1"
            ),
            implementation="flagscale-local-causal-inference",
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
            checkpoint=str(self.checkpoint),
            num_denoising_steps=5,
            timestep_shift=5.0,
            context_latent_frames=self.context_latent_frames,
            kv_capacity_frames=self.context_latent_frames + 3,
            boundary_pixel_cache_frames=1 + (self.context_latent_frames - 1) * 4,
            context_selection="reencoded_boundary_plus_recent_latents",
            keep_first_frame=False,
            historical_recomputation=True,
            vae_dtype="float16",
            upstream_initial_pixel_crop=3,
            compatibility=[
                "strict local meta weight loading",
                "pth T5 weights",
                "FA2 import fix",
                "eager streaming VAE",
            ],
        )

    def close(self):
        pass


def load_local_transformer(checkpoint):
    """Construct the original upstream class without first loading teacher weights."""
    from .wan.scheduler import FlowMatchScheduler
    from .wan.backbone import DenseWanDiffusionWrapper as WanDiffusionWrapper
    from .wan.baseline_causal_model import CausalWanModel
    from .wan.loading import load_krea_model

    model = load_krea_model(
        checkpoint,
        CausalWanModel,
        dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
    )
    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.uniform_timestep = False
    wrapper.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    wrapper.scheduler.set_timesteps(1000, training=True)
    wrapper.seq_len = 32760
    wrapper.post_init()
    for block in model.blocks:
        block.self_attn.fuse_projections()
    return wrapper.eval().requires_grad_(False).cuda()


def load_local_text_encoder(assets):
    from accelerate import init_empty_weights
    from .wan.backbone import WanTextEncoder
    from .wan.t5 import umt5_xxl
    from .wan.tokenizers import HuggingfaceTokenizer

    root = assets / "wan_models/Wan2.1-T2V-1.3B"
    encoder = WanTextEncoder.__new__(WanTextEncoder)
    nn.Module.__init__(encoder)
    with init_empty_weights():
        encoder.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.bfloat16,
            device=torch.device("meta"),
        )
    weights = torch.load(
        root / "models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu", weights_only=True
    )
    encoder.text_encoder.load_state_dict(weights, strict=True, assign=True)
    encoder.tokenizer = HuggingfaceTokenizer(
        name=str(root / "google/umt5-xxl"), seq_len=512, clean="whitespace"
    )
    return encoder.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)


def load_local_streaming_vae(assets):
    """Preserve the original FP16 block-3 decoder and boundary encoder."""
    from .streaming_vae import VAEEncoderWrapper, VAEDecoderWrapper
    from .wan.baseline_vae import WanVAE

    path = assets / "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    vae = WanVAE(vae_pth=str(path), dtype=torch.float16)
    encoder = VAEEncoderWrapper(vae)
    decoder = VAEDecoderWrapper()
    state = torch.load(path, map_location="cpu", weights_only=True)
    decoder_state = {
        key: value for key, value in state.items() if "decoder." in key or "conv2" in key
    }
    keys = decoder.load_state_dict(decoder_state, strict=False)
    if set(keys.missing_keys) != {"mean", "std"} or keys.unexpected_keys:
        raise ValueError(f"Streaming VAE checkpoint mismatch: {keys}")
    encoder.eval().to(dtype=torch.float16).requires_grad_(False).to(torch.cuda.current_device())
    decoder.eval().to(dtype=torch.float16).requires_grad_(False).to(torch.cuda.current_device())
    return encoder, decoder


class ARDiffusionPipeline(nn.Module):
    """FlagScale wrapper around GenerationSession's dense AR+Diffusion loop."""

    def __init__(self, models, runtime):
        super().__init__()
        self.native_pipeline = models.pipeline
        self.vae_encoder = models.vae_encoder
        self.models, self.runtime = models, runtime

    @property
    def transformer(self):
        return self.models.transformer.model

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
        context_latent_frames=6,
    ):
        if torch_dtype != torch.bfloat16:
            raise ValueError("AR+Diffusion baseline requires BF16 DiT/T5 and FP16 streaming VAE")
        runtime = ARDiffusionRuntime(
            pretrained_model_name_or_path, checkpoint, context_latent_frames
        )
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("AR+Diffusion baseline requires CUDA")
            from .baseline_session import CausalInferencePipeline

            transformer = load_local_transformer(runtime.checkpoint)
            text_encoder = load_local_text_encoder(runtime.assets)
            vae_encoder, vae_decoder = load_local_streaming_vae(runtime.assets)
            native = CausalInferencePipeline(transformer, text_encoder, vae_decoder)
            models = SimpleNamespace(
                text_encoder=text_encoder,
                transformer=transformer,
                pipeline=native,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
            )
            torch.cuda.empty_cache()
            return cls(models, runtime)
        except BaseException:
            runtime.close()
            raise

    @torch.inference_mode()
    def forward(
        self,
        prompt,
        *,
        latent_frames=120,
        height=480,
        width=832,
        generator=None,
        return_dict=True,
        output_type="np",
        return_latents=False,
        **kwargs,
    ):
        if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Krea supports one nonempty prompt per request")
        if type(latent_frames) is not int or latent_frames < 3 or latent_frames % 3:
            raise ValueError("latent_frames must be a positive multiple of 3")
        # Upstream boundary reencoding hardcodes this size.
        if (height, width) != (480, 832):
            raise ValueError("The original Krea default session requires 832x480")
        if output_type not in ("np", "latent"):
            raise ValueError("output_type must be np or latent")
        seed = generator.initial_seed() if generator is not None else 42
        from .baseline_session import GenerateParams, GenerationSession

        params = GenerateParams(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            num_blocks=latent_frames // 3,
            num_denoising_steps=5,
            kv_cache_num_frames=self.runtime.context_latent_frames,
            keep_first_frame=False,
            timestep_shift=5.0,
        )
        chunks = []

        def collect(pixels, frame_ids, event):
            event.synchronize()
            if output_type == "np":
                chunks.append(
                    pixels[0].float().cpu().permute(0, 2, 3, 1).mul(0.5).add(0.5).clamp_(0, 1)
                )

        with torch.cuda.device(self.device):
            session = GenerationSession(params, models=self.models, frame_callback=collect)
            try:
                for _ in range(params.num_blocks):
                    session.generate_block_internal(self.models)
                # Check the real upstream allocation, not only the requested setting.
                actual_kv_capacity = (
                    self.native_pipeline.kv_cache1[0]["k"].shape[1]
                    // self.native_pipeline.frame_seq_length
                )
                if actual_kv_capacity != params.kv_cache_num_frames + 3:
                    raise RuntimeError("Krea KV allocation did not match requested context")
                metadata = dict(
                    self.runtime.snapshot(),
                    seed=seed,
                    latent_frames=latent_frames,
                    pixel_frames=session.total_frames_sent,
                    fps=24,
                    denoising_timesteps=[float(t) for t in session.denoising_step_list],
                    denoising_forwards=5 * params.num_blocks,
                    historical_recompute_forwards=params.num_blocks - 1,
                    historical_recompute_latent_frames=sum(
                        min(3 * block, params.kv_cache_num_frames)
                        for block in range(1, params.num_blocks)
                    ),
                    boundary_reencode_forwards=max(
                        0, params.num_blocks - 1 - params.kv_cache_num_frames // 3
                    ),
                    actual_kv_capacity_frames=actual_kv_capacity,
                    actual_boundary_pixel_cache_frames=session.frame_context_cache.maxlen,
                )
                output = ARDiffusionOutput(
                    [torch.cat(chunks).numpy()] if chunks else [],
                    (
                        session.all_latents.cpu()
                        if return_latents or output_type == "latent"
                        else None
                    ),
                    metadata,
                )
            finally:
                session.dispose()
                self.transformer.block_mask = None
                del session
        return output if return_dict else (output.frames, output.latents)

    def close(self):
        self.native_pipeline.kv_cache1 = None
        self.native_pipeline.crossattn_cache = None
        self.runtime.close()
