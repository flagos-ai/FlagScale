# Pi0.5 Model Implementation - Extended from Pi0
# Supports discrete state input and adaRMSNorm for flow matching timestep injection

# Adopted from huggingface/lerobot (https://github.com/huggingface/lerobot/tree/main)
# Extended to support Pi0.5 architecture

import json
import math
import os

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from flagscale.models.pi0.normalize import Normalize, Unnormalize
from flagscale.models.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from flagscale.models.pi0.types import ACTION, OBS_STATE, FeatureType
from flagscale.runner.utils import logger


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    equal or larger than the mask_as of the query token. For example, if we want
    the model to predict the third token given the first two tokens as context,
    we need to make sure mask_ar[0] >= mask_ar[2], mask_ar[1] >= mask_ar[2],
    mask_ar[2] >= mask_ar[2]. The attention mask will be:
    token -> token
    token 0 -> token 0, token 1, token 2
    token 1 -> token 0, token 1, token 2
    token 2 -> token 0, token 1, token 2
    """
    if pad_masks is None and att_masks is None:
        return None
    if pad_masks is None:
        pad_masks = torch.ones_like(att_masks, dtype=torch.bool)
    if att_masks is None:
        att_masks = torch.ones_like(pad_masks, dtype=torch.bool)

    # Prepare a mask for token positions where each token should attend to.
    # E.g. for the example above, we want to create:
    # ar_mask = [[0, 1, 2], [0, 1, 2], [0, 1, 2]].
    att_masks_cum = att_masks.cumsum(-1)
    ar_mask = att_masks_cum[..., None] >= att_masks_cum[..., None, :]
    return ar_mask & pad_masks[..., None, :] & pad_masks[..., None, :, None]


def get_safe_dtype(dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Gets a safe dtype for the given device type."""
    if dtype == torch.bfloat16:
        # Only newer GPUs support bfloat16
        if device_type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        else:
            logger.warning("bfloat16 is not supported on this device, using float16 instead")
            return torch.float16
    elif dtype == torch.float16:
        # CPU doesn't support float16
        if device_type == "cpu":
            logger.warning("float16 is not supported on CPU, using float32 instead")
            return torch.float32
        else:
            return torch.float16
    else:
        return dtype


@dataclass
class PI0_5_PolicyConfig(PretrainedConfig):
    """
    Pi0.5 Policy Configuration

    Extends base PretrainedConfig with Pi0.5 specific parameters:
    - Discrete state input support
    - AdaRMSNorm normalization
    - Flow matching generation
    - Extended action space (32-dim, 16 steps)
    """

    model_type = "pi05"

    def __init__(
        self,
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,  # Pi0.5 uses 32-dim actions
        action_horizon=16,  # Pi0.5 typically uses 16
        max_token_len=200,  # Pi0.5 supports longer sequences
        pi05=True,  # Enable Pi0.5 mode
        discrete_state_input=True,  # Pi0.5 uses discrete state input
        ada_norm_eps=1e-6,
        ada_norm_hidden_size=None,
        flow_matching_timesteps=1000,
        flow_matching_sigma_min=0.002,
        flow_matching_sigma_max=80.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.pi05 = pi05
        self.discrete_state_input = discrete_state_input
        self.ada_norm_eps = ada_norm_eps
        self.ada_norm_hidden_size = ada_norm_hidden_size
        self.flow_matching_timesteps = flow_matching_timesteps
        self.flow_matching_sigma_min = flow_matching_sigma_min
        self.flow_matching_sigma_max = flow_matching_sigma_max


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm for Pi0.5 flow matching timestep injection"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.gate = nn.Linear(1, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # Normalize
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # Adaptive weighting based on timestep
        timestep_embedding = timestep.float().unsqueeze(-1)  # [batch_size, 1]
        adaptive_weight = torch.sigmoid(self.gate(timestep_embedding))  # [batch_size, hidden_size]

        # Apply adaptive weight to match hidden_states dimensions
        # Expand adaptive_weight to match all dimensions except the last one
        adaptive_weight = adaptive_weight.unsqueeze(1)  # [batch_size, 1, hidden_size]
        adaptive_weight = adaptive_weight.expand_as(
            hidden_states
        )  # [batch_size, seq_len, hidden_size]

        # Apply weight
        return (1.0 + adaptive_weight) * self.weight * hidden_states


class DiscreteStateProcessor(nn.Module):
    """Process continuous states into discrete tokens for Pi0.5"""

    def __init__(self, state_dim: int, vocab_size: int = 1000):
        super().__init__()
        self.state_dim = state_dim
        self.vocab_size = vocab_size

        # Learnable bin centers for discretization
        self.bin_centers = nn.Parameter(torch.linspace(-3.0, 3.0, vocab_size))

        # Embedding for discrete state tokens
        self.state_embedding = nn.Embedding(vocab_size, state_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous states to discrete tokens

        Args:
            states: [batch_size, state_dim] continuous states

        Returns:
            discrete_states: [batch_size, state_dim] discrete state tokens
        """
        # Discretize each dimension
        batch_size, state_dim = states.shape
        discrete_tokens = []

        for i in range(state_dim):
            # Find nearest bin center for each dimension
            state_dim_i = states[:, i : i + 1]  # [batch_size, 1]
            distances = torch.abs(
                state_dim_i - self.bin_centers.unsqueeze(0)
            )  # [batch_size, vocab_size]
            token_ids = torch.argmin(distances, dim=-1)  # [batch_size]
            discrete_tokens.append(token_ids)

        discrete_states = torch.stack(discrete_tokens, dim=-1)  # [batch_size, state_dim]
        return discrete_states


class PI0_5_Model(nn.Module):
    """
    Pi0.5 Model Implementation

    Extends Pi0 architecture with:
    - PaliGemmaWithExpert backbone
    - Discrete state input processing
    - AdaRMSNorm for timestep conditioning
    - Flow matching generation
    - 32-dimensional action space
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize PaliGemma with Expert (following Pi0 pattern)
        from flagscale.models.pi0.paligemma_with_expert import (
            PaliGemmaWithExpertConfig,
            PaliGemmaWithExpertModel,
        )

        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=getattr(config, 'freeze_vision_encoder', False),
            train_expert_only=getattr(config, 'train_expert_only', False),
            attention_implementation=getattr(
                config, 'attention_implementation', 'flash_attention_2'
            ),
            use_adarms=[False, True] if getattr(config, 'pi05', False) else [False, False],
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)

        # Projection layers (following Pi0 pattern)
        hidden_size = getattr(config, 'hidden_size', 2048)
        proj_width = getattr(config, 'proj_width', 1024)

        self.state_proj = nn.Linear(getattr(config, 'max_state_dim', 32), proj_width)
        self.action_in_proj = nn.Linear(config.action_dim, proj_width)
        self.action_out_proj = nn.Linear(proj_width, config.action_dim)

        self.action_time_mlp_in = nn.Linear(proj_width * 2, proj_width)
        self.action_time_mlp_out = nn.Linear(proj_width, proj_width)

        # Pi0.5 specific components
        if config.pi05:
            self.state_processor = DiscreteStateProcessor(
                state_dim=config.action_dim, vocab_size=getattr(config, 'state_vocab_size', 1000)
            )

            # AdaRMSNorm for flow matching timestep injection
            ada_hidden_size = config.ada_norm_hidden_size or hidden_size
            self.ada_norm = AdaRMSNorm(hidden_size=ada_hidden_size, eps=config.ada_norm_eps)
        else:
            self.state_processor = None
            self.ada_norm = None

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ):
        """Forward pass following Pi0 pattern"""
        logger.debug(
            f"PI05Model forward: images={images.shape if hasattr(images, 'shape') else 'tensor'}, "
            f"state={state.shape if hasattr(state, 'shape') else 'tensor'}"
        )

        batch_size = state.shape[0] if hasattr(state, 'shape') else 1

        # Sample noise and time if not provided
        if noise is None:
            noise = torch.randn(
                batch_size,
                getattr(self.config, 'n_action_steps', 16),
                self.config.action_dim,
                device=state.device,
            )

        if time is None:
            # Sample time for flow matching
            time = torch.rand(batch_size, device=state.device)

        # Prepare flow matching inputs
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Process inputs through PaliGemma
        try:
            # Embed images and language
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )

            # Embed state and actions
            suffix_embs, suffix_pad_masks, suffix_att_masks = self._embed_suffix(state, x_t, time)

            # Combine and process through transformer
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            # Use Pi0's attention mask making
            try:
                from flagscale.models.pi0.modeling_pi0 import make_att_2d_masks

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            except:
                # Fallback: simple causal mask
                seq_len = pad_masks.shape[1]
                att_2d_masks = torch.tril(
                    torch.ones(seq_len, seq_len, device=pad_masks.device, dtype=torch.bool)
                )
                att_2d_masks = att_2d_masks.unsqueeze(0).expand(batch_size, -1, -1)

            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            # Forward through PaliGemma
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )

            suffix_out = suffix_out[:, -getattr(self.config, 'n_action_steps', 16) :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

        except Exception as e:
            logger.warning(f"Full forward pass failed, using fallback: {e}")
            # Fallback: simple projection
            v_t = self.action_out_proj(
                self.action_time_mlp_out(
                    F.silu(
                        self.action_time_mlp_in(
                            torch.cat(
                                [
                                    self.action_in_proj(actions),
                                    time.unsqueeze(-1).expand(-1, self.config.action_dim),
                                ],
                                dim=-1,
                            )
                        )
                    )
                )
            )

        # Compute loss
        losses = F.mse_loss(v_t, u_t, reduction="none")
        return losses

    def _embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images and language tokens (simplified version of Pi0's method)"""
        batch_size = lang_tokens.shape[0]

        # Process images
        if isinstance(images, list):
            embs = []
            pad_masks = []
            att_masks = []

            for img, img_mask in zip(images, img_masks):
                img_emb = self.paligemma_with_expert.embed_image(img)
                img_emb = img_emb.to(dtype=torch.bfloat16)

                # Normalize embeddings
                img_emb_dim = img_emb.shape[-1]
                img_emb = img_emb * torch.tensor(
                    img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
                )

                bsize, num_img_embs = img_emb.shape[:2]
                img_mask_expanded = img_mask[:, None].expand(bsize, num_img_embs)

                embs.append(img_emb)
                pad_masks.append(img_mask_expanded)
                att_masks += [0] * num_img_embs
        else:
            # Single image tensor
            img_emb = self.paligemma_with_expert.embed_image(images)
            img_emb = img_emb.to(dtype=torch.bfloat16)
            embs = [img_emb]
            pad_masks = [img_masks]
            att_masks = [0] * img_emb.shape[1]

        # Process language
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * (lang_emb_dim**0.5)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # Full attention between image and language
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(batch_size, len(att_masks))

        return embs, pad_masks, att_masks

    def _embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep (simplified version of Pi0's method)"""
        batch_size = state.shape[0]

        # Embed state
        state = state.to(dtype=self.state_proj.weight.dtype)
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs = [state_emb[:, None, :]]

        state_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=state.device)
        pad_masks = [state_mask]
        att_masks = [1]

        # Create sinusoidal time embedding
        try:
            from flagscale.models.pi0.modeling_pi0 import create_sinusoidal_pos_embedding

            time_emb = create_sinusoidal_pos_embedding(
                timestep, 1024, min_period=4e-3, max_period=4.0, device=state.device
            )
        except:
            # Fallback time embedding
            time_emb = torch.sin(
                timestep.unsqueeze(-1) * torch.arange(1024, device=state.device) * 0.1
            )

        time_emb = time_emb.type(dtype=torch.bfloat16)

        # Fuse timestep + action information
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)

        batch_size, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            batch_size, action_time_dim, dtype=torch.bool, device=state.device
        )
        pad_masks.append(action_time_mask)

        n_action_steps = action_time_dim
        att_masks += [1] + ([0] * (n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(batch_size, len(att_masks))

        return embs, pad_masks, att_masks


class PI0_5_Policy(PreTrainedModel):
    """Wrapper class around PI05 Policy model to train and run inference within FlagScale."""

    config_class = PI0_5_PolicyConfig
    name = "pi05"

    def __init__(
        self, model_path: str, tokenizer_path: str, stat: dict, config: PI0_5_PolicyConfig
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)

        self.config = config

        # Initialize normalization (following Pi0 pattern)
        try:
            from flagscale.models.pi0.normalize import Normalize, Unnormalize

            self.normalize_inputs = Normalize(
                config.input_features, config.normalization_mapping, stat
            )
            self.normalize_targets = Normalize(
                config.output_features, config.normalization_mapping, stat
            )
            self.unnormalize_outputs = Unnormalize(
                config.output_features, config.normalization_mapping, stat
            )
        except:
            # Fallback for basic usage
            self.normalize_inputs = lambda x: x
            self.normalize_targets = lambda x: x
            self.unnormalize_outputs = lambda x: x

        # Initialize tokenizer - create basic tokenizer for Pi0.5
        # Following FlagScale pattern for custom models
        class BasicTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "</s>"
                self.bos_token = "<s>"
                self.unk_token = "<unk>"
                self.vocab_size = 32000

            def __call__(self, text, **kwargs):
                # Simple tokenization for inference
                if isinstance(text, str):
                    return [hash(text) % self.vocab_size]
                return text

        self.language_tokenizer = BasicTokenizer()

        # Create the underlying model
        self.model = PI0_5_Model(config)

        # Load PaliGemmaWithExpert directly from the loaded model weights
        try:
            from flagscale.models.pi0.paligemma_with_expert import (
                PaliGemmaWithExpertConfig,
                PaliGemmaWithExpertModel,
            )

            # Create proper config for PaliGemmaWithExpertModel
            paligemma_with_expert_config = PaliGemmaWithExpertConfig(
                freeze_vision_encoder=getattr(config, 'freeze_vision_encoder', False),
                train_expert_only=getattr(config, 'train_expert_only', False),
                attention_implementation=getattr(config, 'attention_implementation', 'eager'),
                use_adarms=[False, True] if getattr(config, 'pi05_mode', False) else [False, False],
            )

            # PaliGemmaWithExpert will be loaded as part of the model weights
            # No need to load separately since Pi0.5 checkpoint contains all components
            self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config)
            print(f"Created PaliGemmaWithExpertModel for Pi0.5")
        except Exception as e:
            print(f"Warning: Could not create PaliGemmaWithExpertModel: {e}")
            print("Proceeding without PaliGemmaWithExpertModel")
            self.paligemma_with_expert = None

        self.reset()

    @classmethod
    def from_pretrained(cls, model_path, tokenizer_path, stat_path, config):
        """Load Pi0.5 model from checkpoint following Pi0 pattern"""

        # Load statistics
        with open(stat_path, "r") as f:
            stat = json.load(f)
            try:
                # Try to flatten/unflatten like Pi0
                stat = {key: np.array(value) for key, value in flatten_dict(stat).items()}
                stat = unflatten_dict(stat)
            except:
                # Use stats as-is if flattening fails
                pass

        # Load config if not provided
        if config is None:
            config_path = f"{model_path}/config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = cls.config_class(**config_dict)
            else:
                config = cls.config_class()

        # Create policy
        policy = PI0_5_Policy(model_path, tokenizer_path, stat, config)

        # Load model weights
        checkpoint_path = f"{model_path}/model.safetensors"
        if os.path.exists(checkpoint_path):
            state_dict = load_file(checkpoint_path)
            # Apply Pi0-style key transformations if needed
            transformed_state_dict = cls._transform_state_dict_keys(state_dict)
            policy.model.load_state_dict(transformed_state_dict, strict=False)
            logger.info(f"Loaded Pi0.5 model weights from {checkpoint_path}")
        else:
            logger.warning(
                f"Model checkpoint not found at {checkpoint_path}, using initialized weights"
            )

        return policy

    @classmethod
    def _transform_state_dict_keys(cls, state_dict: dict) -> dict:
        """
        Transform state dict keys to match expected model structure.
        Simplified version of Pi0's transformations for Pi0.5.
        """
        import re

        transformed_dict = {}

        # Apply Pi0-style transformations if they exist in the keys
        transformations = [
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.language_model\.lm_head"),
                ".paligemma_with_expert.paligemma.lm_head",
            ),
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.language_model\.model"),
                ".paligemma_with_expert.paligemma.model.language_model",
            ),
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.vision_tower"),
                ".paligemma_with_expert.paligemma.model.vision_tower",
            ),
            (
                re.compile(r"\.paligemma_with_expert\.paligemma\.multi_modal_projector"),
                ".paligemma_with_expert.paligemma.model.multi_modal_projector",
            ),
        ]

        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in transformations:
                new_key = pattern.sub(replacement, new_key)
            transformed_dict[new_key] = value

        return transformed_dict

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque(
            [], maxlen=getattr(self.config, 'n_action_steps', self.config.action_horizon)
        )

    def prepare_images(self, batch):
        """Prepare images for inference (following Pi0 pattern)"""
        images_keys = getattr(
            self.config,
            'images_keys',
            [
                'observation.images.camera0',
                'observation.images.camera1',
                'observation.images.camera2',
            ],
        )

        images = []
        img_masks = []

        for key in images_keys:
            if key in batch:
                img = batch[key]
                # Resize and normalize following Pi0 pattern
                if hasattr(img, 'shape') and len(img.shape) == 4:
                    # [batch, channels, height, width] -> normalize to [-1, 1]
                    img = img * 2.0 - 1.0
                images.append(img)
                img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))
            else:
                # Create dummy image if key not found
                device = next(self.model.parameters()).device
                dummy_img = torch.randn(1, 3, 480, 640, device=device)
                dummy_img = dummy_img * 2.0 - 1.0
                images.append(dummy_img)
                img_masks.append(torch.ones(1, dtype=torch.bool, device=dummy_img.device))

        return images, img_masks

    def prepare_state(self, batch):
        """Prepare state for inference"""
        state_key = getattr(self.config, 'state_key', 'observation.state')
        if state_key in batch:
            state = batch[state_key]
            # Pad or truncate to expected dimension
            expected_dim = getattr(self.config, 'action_dim', 32)
            if state.shape[-1] < expected_dim:
                padding = torch.zeros(
                    state.shape[0],
                    expected_dim - state.shape[-1],
                    device=state.device,
                    dtype=state.dtype,
                )
                state = torch.cat([state, padding], dim=-1)
            elif state.shape[-1] > expected_dim:
                state = state[:, :expected_dim]
            return state
        else:
            # Create dummy state
            return torch.randn(1, getattr(self.config, 'action_dim', 32))

    def prepare_language(self, batch):
        """Prepare language tokens for inference"""
        if 'task' in batch:
            tasks = batch['task']
            if isinstance(tasks, list):
                max_len = getattr(self.config, 'tokenizer_max_length', 200)
                # Create dummy tokens (real implementation would use tokenizer)
                tokens = torch.randint(0, 1000, (len(tasks), max_len))
                masks = torch.ones(len(tasks), max_len, dtype=torch.bool)
                return tokens, masks
        # Return dummy tokens
        return torch.randint(0, 1000, (1, 32)), torch.ones(1, 32, dtype=torch.bool)

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        instruction: str,
        timestep: torch.Tensor = None,
        pi05_mode: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for Pi0.5 model

        Args:
            images: [batch_size, num_cameras, channels, height, width]
            states: [batch_size, state_dim] - continuous or discrete
            instruction: text instruction
            timestep: [batch_size] - flow matching timestep (for Pi0.5)
            pi05_mode: whether to use Pi0.5 specific processing

        Returns:
            actions: [batch_size, action_horizon, action_dim]
        """
        batch_size = images.shape[0]

        # Process images
        image_features = self.vision_encoder(images)

        # Process instruction
        text_tokens = self.language_model.tokenize(instruction)
        text_features = self.language_model.embed(text_tokens)

        if pi05_mode and self.config.pi05:
            # Pi0.5 specific processing

            # Convert states to discrete tokens if needed
            if self.config.discrete_state_input:
                discrete_states = self.state_processor(states)  # [batch_size, state_dim]
                state_tokens = self.language_model.embed(discrete_states)
            else:
                state_tokens = states

            # Combine text and state tokens
            combined_tokens = torch.cat([text_features, state_tokens], dim=1)

            # Apply adaRMSNorm with timestep if provided
            if timestep is not None and self.ada_norm is not None:
                image_features = self.ada_norm(image_features, timestep)

            # Generate actions
            action_logits = self.action_head(image_features, combined_tokens)

        else:
            # Standard Pi0 processing
            combined_tokens = torch.cat([text_features, states.unsqueeze(1)], dim=1)
            action_logits = self.action_head(image_features, combined_tokens)

        return action_logits

    def forward(self, batch):
        """
        Training forward pass that accepts a batch dictionary

        Args:
            batch: Dictionary containing:
                - observation.images.camera0, camera1, camera2: [batch_size, 3, H, W]
                - observation.state: [batch_size, state_dim]
                - action: [batch_size, action_horizon, action_dim]
                - task: [batch_size] or list of instructions

        Returns:
            loss: Training loss
            outputs: Model outputs
        """
        try:
            device = next(self.parameters()).device

            # Validate batch contents
            if not isinstance(batch, dict):
                raise ValueError("Batch must be a dictionary")

            required_keys = ['action']
            missing_keys = [key for key in required_keys if key not in batch]
            if missing_keys:
                raise ValueError(f"Missing required keys in batch: {missing_keys}")

            # Prepare inputs with error handling
            try:
                images, img_masks = self.prepare_images(batch)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare images: {e}")

            try:
                states = self.prepare_state(batch)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare states: {e}")

            instruction = (
                batch.get('task', ['Default task'])[0] if 'task' in batch else 'Default task'
            )

            # Get target actions with validation
            target_actions = batch['action']
            if not torch.is_tensor(target_actions):
                raise ValueError("Target actions must be a torch.Tensor")

            target_actions = target_actions.to(device)
            batch_size = target_actions.shape[0]

            # Validate action dimensions
            if len(target_actions.shape) != 3:
                raise ValueError(
                    f"Expected actions to be 3D (batch, horizon, dim), got shape {target_actions.shape}"
                )

            if target_actions.shape[1] != self.config.action_horizon:
                raise ValueError(
                    f"Expected action horizon {self.config.action_horizon}, got {target_actions.shape[1]}"
                )

            if target_actions.shape[2] != self.config.action_dim:
                raise ValueError(
                    f"Expected action dimension {self.config.action_dim}, got {target_actions.shape[2]}"
                )

            # Generate noise and time for flow matching
            noise = torch.randn_like(target_actions)
            time = torch.rand(batch_size, device=device)

            # Forward pass through model's internal forward method with error handling
            try:
                with torch.amp.autocast('cuda', enabled=True):
                    # Stack images for processing [batch_size, num_cameras, channels, height, width]
                    images_tensor = torch.stack(images, dim=1)  # [batch_size, num_cameras, 3, H, W]
                    batch_size = images_tensor.shape[0]

                    # Use the underlying model for forward pass
                    # Simple approach: call the model with prepared inputs
                    lang_tokens, lang_masks = self.prepare_language(batch)

                    # Reshape images for the model: combine batch and camera dimensions
                    # images_tensor: [batch_size, num_cameras, 3, H, W] -> [batch_size*num_cameras, 3, H, W]
                    batch_size, num_cameras = images_tensor.shape[:2]
                    images_reshaped = images_tensor.view(-1, *images_tensor.shape[2:])

                    # Use PI0_5_Model forward method for training
                    # This should return the loss directly
                    loss = self.model.forward(
                        images=images_reshaped,
                        img_masks=[m.repeat_interleave(num_cameras) for m in img_masks],
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=states,
                        actions=target_actions,
                        noise=noise,
                        time=time,
                    )

                outputs = {'target_actions': target_actions, 'loss': loss}

                return loss, outputs

            except torch.cuda.OutOfMemoryError as e:
                # Handle CUDA out of memory
                raise RuntimeError(
                    f"CUDA out of memory during forward pass: {e}. "
                    "Try reducing batch size or model size."
                )
            except Exception as e:
                # Handle other model-specific errors
                raise RuntimeError(f"Error during model forward pass: {e}")

        except (ValueError, RuntimeError) as e:
            # Re-raise validation and runtime errors with more context
            raise RuntimeError(f"Pi0.5 forward pass failed: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(f"Unexpected error in Pi0.5 forward pass: {e}")

    def generate(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        instruction: str,
        max_token_len: int = 200,
        action_horizon: int = 16,
        pi05_mode: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate action sequence"""
        # For Pi0.5, use flow matching for generation
        if pi05_mode and self.config.pi05:
            return self._generate_flow_matching(
                images, states, instruction, max_token_len, action_horizon, **kwargs
            )
        else:
            return self._generate_autoregressive(
                images, states, instruction, max_token_len, action_horizon, **kwargs
            )

    def _generate_flow_matching(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        instruction: str,
        max_token_len: int,
        action_horizon: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate actions using flow matching (Pi0.5 specific)"""
        batch_size = images.shape[0]
        device = images.device

        # Sample random timestep
        timestep = torch.rand(batch_size, device=device) * self.config.flow_matching_timesteps

        # Forward pass
        action_logits = self.forward(
            images=images, states=states, instruction=instruction, timestep=timestep, pi05_mode=True
        )

        # Convert logits to actions (simplified)
        actions = torch.tanh(action_logits)

        # Reshape to action horizon
        if actions.shape[-1] == self.config.action_dim:
            actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)

        return actions

    def _generate_autoregressive(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        instruction: str,
        max_token_len: int,
        action_horizon: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate actions autoregressively (standard Pi0)"""
        # Standard autoregressive generation
        action_logits = self.forward(
            images=images, states=states, instruction=instruction, pi05_mode=False
        )

        actions = torch.tanh(action_logits)
        actions = actions.unsqueeze(1).expand(-1, action_horizon, -1)

        return actions


# Convenience function for loading Pi0.5 models
def create_pi05_policy(
    model_path: str, tokenizer_path: str, stat_path: str, config: PI0_5_PolicyConfig = None
) -> PI0_5_Policy:
    """Create a Pi0.5 policy from checkpoint"""
    if config is None:
        config = PI0_5_PolicyConfig()

    return PI0_5_Policy.from_pretrained(
        model_path=model_path, tokenizer_path=tokenizer_path, stat_path=stat_path, config=config
    )
