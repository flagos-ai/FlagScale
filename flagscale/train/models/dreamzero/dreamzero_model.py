# Copyright (c) 2025, FlagScale Authors. All rights reserved.
"""
DreamZero Model Module for FlagScale native training backend.

DreamZero is a World Action Model (WAM) - a Wan2.1 video DiT repurposed as a
zero-shot robot policy. Architecture:
  - IdentityBackbone (no LLM backbone)
  - WANPolicyHead: CausalWanModel (DiT) + VAE + T5 + CLIP + action/state encoders

Training forward:
  1. Encode video frames -> latents (VAE, frozen)
  2. Encode text -> prompt embeddings (T5, frozen)
  3. Encode first frame -> CLIP features (CLIP, frozen)
  4. Add noise to latents and actions (flow matching scheduler)
  5. DiT predicts noise (trainable, with LoRA)
  6. Compute MSE loss (dynamics + action)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

logger = logging.getLogger(__name__)


@dataclass
class DreamZeroConfig(PretrainedConfig):
    """DreamZero model configuration."""
    model_type = "dreamzero"

    # DiT backbone (14B default from DreamZero-AgiBot)
    dim: int = 5120
    ffn_dim: int = 13824
    num_heads: int = 40
    num_layers: int = 40
    freq_dim: int = 256
    in_dim: int = 36
    out_dim: int = 16
    frame_seqlen: int = 220
    max_chunk_size: int = 4
    num_frame_per_block: int = 2
    num_action_per_block: int = 48
    num_state_per_block: int = 1
    eps: float = 1e-6
    model_subtype: str = "i2v"

    # Action head
    action_dim: int = 32
    action_horizon: int = 48
    max_state_dim: int = 64
    max_action_dim: int = 32
    hidden_size: int = 64
    input_embedding_dim: int = 1536

    # LoRA
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"

    # Training
    num_frames: int = 33
    use_gradient_checkpointing: bool = True
    train_architecture: str = "full"
    tune_diffusion_model: bool = True
    tune_projector: bool = True
    use_vlln: bool = True

    # Noise scheduling
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # REPA
    repa_layer: int = 8
    repa_coeff: float = 1.0

    # VL self-attention
    vl_num_layers: int = 4
    vl_num_heads: int = 24
    vl_head_dim: int = 64
    vl_dropout: float = 0.2

    # Compute
    compute_dtype: str = "bfloat16"

    # Paths (set at runtime)
    model_path: str = None
    tokenizer_path: str = None
    dit_version: str = None  # Path to Wan2.1-I2V-14B-480P dir (DIT pretrained weights)
    text_encoder_pretrained_path: str = None
    image_encoder_pretrained_path: str = None
    vae_pretrained_path: str = None

    # Embodiment
    embodiment_tag: str = "agibot"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(PreTrainedModel):
    """DreamZero policy model for FlagScale native backend (FSDP2).

    Wraps the original DreamZero VLA (IdentityBackbone + WANPolicyHead).
    All weights live under action_head.* (2146 keys total).
    """

    supports_gradient_checkpointing = True
    config_class = DreamZeroConfig

    # FSDP2 no-split modules
    _no_split_modules = [
        "T5SelfAttention",
        "AttentionBlock",
        "CausalWanModel",
        "CausalWanAttentionBlock",
    ]

    def __init__(self, config: DreamZeroConfig):
        super().__init__(config)
        self.config = config
        self._model_loaded = False

    @classmethod
    def from_pretrained_dreamzero(cls, config: DreamZeroConfig) -> "DreamZeroPolicy":
        """Load DreamZero for FSDP2 training.

        Supports two training modes:
        - train_architecture="full": Full fine-tuning of DIT + action/state projectors.
          Loads ALL weights from DreamZero-AgiBot checkpoint directly. No LoRA.
          VAE/T5/CLIP frozen, DIT + encoders/decoders fully trainable.
        - train_architecture="lora": LoRA on DIT with Wan2.1 base weights.
          Loads Wan2.1 DIT from pretrained, skips DIT keys from checkpoint,
          then injects LoRA. (Matches reference accidental behavior.)

        Full fine-tuning is preferred for FSDP2 since FSDP2 handles memory sharding
        natively and avoids the LoRA-on-fine-tuned-weights NaN issue.
        """
        from flagscale.train.models.dreamzero.base_vla import VLA, VLAConfig
        import json
        import gc

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Step 1: Build model config. Use skip_component_loading=False so __init__
        # loads all pretrained weights (T5, CLIP, VAE, Wan2.1 DIT).
        # Use defer_lora_injection=True so we can load non-DIT weights from checkpoint
        # before injecting LoRA (unlike the reference which relies on PEFT key mismatch).
        config_json = model_path / "config.json"
        if not config_json.exists():
            raise FileNotFoundError(
                f"config.json not found at {model_path}. "
                "DreamZero checkpoint must contain config.json."
            )
        with open(config_json) as f:
            vla_dict = json.load(f)

        # skip_component_loading controls whether __init__ loads T5/CLIP/VAE/DIT individually.
        # For full fine-tuning: skip it (we load everything from checkpoint afterward).
        # For LoRA mode: load components so DIT gets Wan2.1 base weights.
        use_lora = config.train_architecture == "lora"
        load_components_separately = use_lora and config.dit_version is not None
        if "action_head_cfg" in vla_dict and "config" in vla_dict["action_head_cfg"]:
            vla_dict["action_head_cfg"]["config"]["defer_lora_injection"] = True
            vla_dict["action_head_cfg"]["config"]["skip_component_loading"] = not load_components_separately

        # Inject pretrained paths for component loading (LoRA mode only)
        if load_components_separately and "action_head_cfg" in vla_dict:
            ah_cfg = vla_dict["action_head_cfg"].get("config", {})
            # DIT pretrained path (Wan2.1 weights)
            if "diffusion_model_cfg" in ah_cfg:
                ah_cfg["diffusion_model_cfg"]["diffusion_model_pretrained_path"] = config.dit_version
            # Text encoder
            if config.text_encoder_pretrained_path and "text_encoder_cfg" in ah_cfg:
                ah_cfg["text_encoder_cfg"]["text_encoder_pretrained_path"] = config.text_encoder_pretrained_path
            # Image encoder
            if config.image_encoder_pretrained_path and "image_encoder_cfg" in ah_cfg:
                ah_cfg["image_encoder_cfg"]["image_encoder_pretrained_path"] = config.image_encoder_pretrained_path
            # VAE
            if config.vae_pretrained_path and "vae_cfg" in ah_cfg:
                ah_cfg["vae_cfg"]["vae_pretrained_path"] = config.vae_pretrained_path

        # Instantiate VLA
        vla_config = VLAConfig(**vla_dict)
        vla_model = VLA(vla_config)

        if load_components_separately:
            logger.info("VLA instantiated with Wan2.1 DIT from pretrained (LoRA mode)")
        else:
            logger.info("VLA instantiated (full fine-tuning mode, loading all from checkpoint)")

        # Step 2: Load DreamZero-AgiBot checkpoint.
        # If use_component_loading=True: filter out DIT keys (DIT keeps Wan2.1 base weights).
        # If use_component_loading=False: load ALL keys from checkpoint.
        from safetensors.torch import load_file
        import os

        safetensors_index_path = os.path.join(str(model_path), "model.safetensors.index.json")
        safetensors_path = os.path.join(str(model_path), "model.safetensors")

        # In LoRA mode: filter DIT + encoder/decoder keys (DIT keeps Wan2.1, encoders start random)
        # In full mode: load everything from checkpoint
        dit_prefixes = (
            "action_head.model.blocks.",
            "action_head.model.head.",
            "action_head.model.img_emb.",
            "action_head.model.patch_embedding.",
            "action_head.model.text_embedding.",
            "action_head.model.time_embedding.",
            "action_head.model.time_projection.",
            "action_head.model.state_encoder.",
            "action_head.model.action_encoder.",
            "action_head.model.action_decoder.",
        )

        def maybe_filter(state_dict):
            """Filter DIT keys in LoRA mode; pass everything in full fine-tuning mode."""
            if not load_components_separately:
                return state_dict, 0
            filtered = {}
            skipped = 0
            for k, v in state_dict.items():
                if any(k.startswith(p) for p in dit_prefixes):
                    skipped += 1
                else:
                    filtered[k] = v
            return filtered, skipped

        if os.path.exists(safetensors_index_path):
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            total_skipped = 0
            for shard_file in sorted(set(index["weight_map"].values())):
                shard_path = os.path.join(str(model_path), shard_file)
                logger.info(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                shard_state_dict, skipped = maybe_filter(shard_state_dict)
                total_skipped += skipped
                if shard_state_dict:
                    vla_model.load_state_dict(shard_state_dict, strict=False)
                del shard_state_dict
                gc.collect()
            if total_skipped:
                logger.info(f"Skipped {total_skipped} DIT keys (keeping Wan2.1 base weights)")
        elif os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            state_dict, skipped = maybe_filter(state_dict)
            if skipped:
                logger.info(f"Skipped {skipped} DIT keys (keeping Wan2.1 base weights)")
            vla_model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
        else:
            raise FileNotFoundError(
                f"No weights at '{model_path}'. "
                "Expected 'model.safetensors' or 'model.safetensors.index.json'."
            )
        logger.info("DreamZero-AgiBot checkpoint loaded")

        # Step 3: Transfer action_head to our policy wrapper
        policy = cls(config)
        policy.action_head = vla_model.action_head
        policy._model_loaded = True
        del vla_model
        gc.collect()

        # Override frame_seqlen on all attention blocks to match our resolution.
        # With num_views=2, tiled to 352x640 → VAE 8x → 44x80 → patch (1,2,2) → 22x40 = 880
        correct_frame_seqlen = config.frame_seqlen
        if hasattr(policy.action_head, 'model'):
            for module in policy.action_head.model.modules():
                if hasattr(module, 'frame_seqlen'):
                    module.frame_seqlen = correct_frame_seqlen
            logger.info(f"Override frame_seqlen={correct_frame_seqlen} on all attention blocks")

        # Override action_horizon and num_action_per_block from our config.
        correct_action_horizon = config.action_horizon
        correct_num_action_per_block = config.num_action_per_block
        if hasattr(policy.action_head, 'model'):
            dit = policy.action_head.model
            if hasattr(dit, 'num_action_per_block'):
                dit.num_action_per_block = correct_num_action_per_block
            if hasattr(dit, 'action_horizon'):
                dit.action_horizon = correct_action_horizon
            for module in dit.modules():
                if hasattr(module, 'num_action_per_block'):
                    module.num_action_per_block = correct_num_action_per_block
        if hasattr(policy.action_head, 'action_horizon'):
            policy.action_head.action_horizon = correct_action_horizon
        logger.info(
            f"Override action params: action_horizon={correct_action_horizon}, "
            f"num_action_per_block={correct_num_action_per_block}"
        )

        # Step 4: Inject LoRA ONCE (same as reference base.py:734)
        if config.train_architecture == "lora":
            policy.action_head.lora_rank = config.lora_rank
            policy.action_head.lora_alpha = config.lora_alpha
            policy.action_head.lora_target_modules = config.lora_target_modules
            policy.action_head.init_lora_weights = getattr(config, "init_lora_weights", "kaiming")
            policy.action_head.train_architecture = "lora"
            # Reset RNG to fixed state before LoRA init for reproducibility.
            # This ensures lora_A (Kaiming init) is identical regardless of
            # code path between set_seed() and here (which differs between
            # FSDP2 and DeepSpeed/HF Trainer pipelines).
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(1234)
            policy.action_head.inject_lora_after_loading()
            torch.random.set_rng_state(rng_state)
            logger.info(
                f"LoRA injected: rank={config.lora_rank}, alpha={config.lora_alpha}, "
                f"targets={config.lora_target_modules}"
            )
        else:
            policy._apply_freeze_config()

        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy.parameters())
        logger.info(
            f"DreamZero loaded. Total: {total/1e9:.2f}B, "
            f"Trainable: {trainable/1e9:.2f}B ({100*trainable/total:.1f}%)"
        )

        return policy

    def _apply_freeze_config(self):
        """Freeze VAE, T5 text encoder, CLIP image encoder. Keep DiT + action head trainable."""
        if not hasattr(self, "action_head"):
            return

        # Freeze VAE
        if hasattr(self.action_head, "vae"):
            for p in self.action_head.vae.parameters():
                p.requires_grad = False
            self.action_head.vae.eval()
            logger.info("Froze VAE encoder")

        # Freeze T5 text encoder
        if hasattr(self.action_head, "text_encoder"):
            for p in self.action_head.text_encoder.parameters():
                p.requires_grad = False
            self.action_head.text_encoder.eval()
            logger.info("Froze T5 text encoder")

        # Freeze CLIP image encoder
        if hasattr(self.action_head, "image_encoder"):
            for p in self.action_head.image_encoder.parameters():
                p.requires_grad = False
            self.action_head.image_encoder.eval()
            logger.info("Froze CLIP image encoder")

        # If LoRA mode, freeze DiT base weights and only train LoRA
        if self.config.train_architecture == "lora":
            for name, p in self.action_head.model.named_parameters():
                if "lora" not in name.lower():
                    p.requires_grad = False
            logger.info("LoRA mode: froze DiT base weights, training LoRA only")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the DiT backbone."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", False)

        if hasattr(self, "action_head") and hasattr(self.action_head, "model"):
            diffusion_model = self.action_head.model
            if hasattr(diffusion_model, "gradient_checkpointing"):
                diffusion_model.gradient_checkpointing = True
            setattr(diffusion_model, "gradient_checkpointing_use_reentrant", use_reentrant)
            logger.info(f"Enabled gradient checkpointing (use_reentrant={use_reentrant})")

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            batch: Dict from get_batch containing:
                - images: (B, T, H, W, C) uint8 video frames
                - action: (B, action_horizon, action_dim) normalized actions [-1, 1]
                - state: (B, state_dim) robot proprioceptive state
                - text: (B, seq_len) tokenized text
                - text_attention_mask: (B, seq_len) attention mask
                - embodiment_id: (B,) embodiment category IDs
                - has_real_action: (B,) bool mask for valid actions
                - action_mask: (B, action_horizon, action_dim) action validity mask

        Returns:
            Dict with 'loss', 'dynamics_loss', 'action_loss'
        """
        # The action_head.forward handles the full training pipeline:
        # encode video, add noise, predict, compute loss
        backbone_output = BatchFeature(data={})  # Identity backbone
        action_input = BatchFeature(data=batch)
        output = self.action_head(backbone_output, action_input)

        if hasattr(output, "data"):
            return dict(output.data)
        return dict(output)

    def set_frozen_modules_to_eval(self):
        """Set frozen modules to eval mode (called before each forward)."""
        if hasattr(self, "action_head"):
            if hasattr(self.action_head, "set_frozen_modules_to_eval_mode"):
                self.action_head.set_frozen_modules_to_eval_mode()
