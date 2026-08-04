# Copyright (c) 2025, FlagScale Authors. All rights reserved.
"""
DreamZero pretraining script for Megatron-LM-FL backend.

Uses Megatron's pretrain() loop with:
- model_provider: wraps DreamZero VLA model in MegatronModule
- forward_step: computes flow-matching loss
- get_batch: loads video/action data, broadcasts across TP ranks

Architecture: DreamZero is a 14B video DiT (diffusion transformer) for robot policy.
- Frozen: SigLIP vision encoder, T5 text encoder, VAE
- Trainable: LoRA on DiT attention/FFN layers
- Loss: flow-matching MSE on denoised action sequences
"""

import sys
import os
import torch
import torch.nn.functional as F
from functools import partial
from pathlib import Path

# Add Megatron-LM-FL to path
MEGATRON_PATH = "/public-mixed/liyuzhuo/Megatron-LM-FL"
if MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)

from megatron.training import get_args, pretrain
from megatron.training.utils import get_batch_on_this_tp_rank
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.huggingface.module import HuggingFaceModule
from megatron.core.tensor_parallel.data import broadcast_data


class DreamZeroMegatronModule(HuggingFaceModule):
    """Wraps DreamZero VLA model as a MegatronModule.
    
    Uses HuggingFaceModule pattern:
    - All params get average_gradients_across_tp_domain=True for gradient sync
    - Model loaded from pretrained checkpoint
    - LoRA injected for trainable parameters
    - Frozen encoders remain in eval mode
    """

    def __init__(self, config):
        # HuggingFaceModule.__init__ expects a config with hf_config
        # We bypass it and call MegatronModule.__init__ directly
        from megatron.core.models.common.base_module import MegatronModule
        MegatronModule.__init__(self, config=config)
        
        args = get_args()
        self.model = self._build_dreamzero_model(args)
        
        # Set gradient sync across TP domain for all params
        for param in self.parameters():
            param.average_gradients_across_tp_domain = True

    def _build_dreamzero_model(self, args):
        """Load DreamZero model from checkpoint with LoRA."""
        from flagscale.train.models.dreamzero.dreamzero_model import (
            from_pretrained_dreamzero,
            DreamZeroConfig,
        )
        
        # Build config from args
        model_config = DreamZeroConfig(
            model_path=args.dreamzero_model_path,
            action_dim=args.dreamzero_action_dim,
            state_dim=args.dreamzero_state_dim,
            frame_seqlen=args.dreamzero_frame_seqlen,
            lora_rank=args.dreamzero_lora_rank,
            lora_alpha=args.dreamzero_lora_alpha,
            lora_targets=getattr(args, 'dreamzero_lora_targets', 
                                 ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']),
        )
        
        # Load model (handles LoRA injection internally)
        model = from_pretrained_dreamzero(model_config)
        
        # Freeze everything except LoRA
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return model

    def set_input_tensor(self, input_tensor):
        """Required for PP compatibility (not used without PP)."""
        pass

    def forward(self, video_latents, robot_states, actions, 
                timesteps=None, text_embeds=None, **kwargs):
        """Forward pass computing flow-matching loss.
        
        Args:
            video_latents: (B, T, C, H, W) encoded video frames
            robot_states: (B, T, state_dim) proprioceptive states
            actions: (B, T, action_dim) ground truth actions
            timesteps: (B,) diffusion timesteps (if None, sampled randomly)
            text_embeds: (B, L, D) pre-encoded text embeddings
            
        Returns:
            loss: scalar flow-matching loss
        """
        # Set frozen modules to eval mode
        self.model.set_frozen_modules_to_eval()
        
        # Forward through DreamZero model
        loss = self.model(
            video_latents=video_latents,
            robot_states=robot_states,
            actions=actions,
            timesteps=timesteps,
            text_embeds=text_embeds,
        )
        return loss


def model_provider(pre_process=True, post_process=True):
    """Build DreamZero model for Megatron training loop.
    
    Args:
        pre_process: True for first PP stage (we only use 1 stage)
        post_process: True for last PP stage
        
    Returns:
        DreamZeroMegatronModule instance
    """
    args = get_args()
    
    # We don't use TransformerConfig since DreamZero isn't a standard transformer
    # Pass None as config - HuggingFaceModule pattern handles this
    model = DreamZeroMegatronModule(config=None)
    
    return model


def get_batch(data_iterator):
    """Get training batch, broadcast across TP ranks.
    
    Only TP rank 0 reads from data_iterator.
    All tensors are broadcast to other TP ranks.
    
    Returns dict with:
        video_latents: (B, T, C, H, W) 
        robot_states: (B, T, state_dim)
        actions: (B, T, action_dim)
    """
    args = get_args()
    
    # Only TP rank 0 reads data
    if mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        # Data comes from DreamZeroDataset collator
        video_latents = data['video_latents']  # (B, T, C, H, W)
        robot_states = data['robot_states']    # (B, T, state_dim)
        actions = data['actions']              # (B, T, action_dim)
    else:
        video_latents = None
        robot_states = None
        actions = None
    
    # Broadcast across TP group
    keys = ['video_latents', 'robot_states', 'actions']
    data_dict = {
        'video_latents': video_latents,
        'robot_states': robot_states,
        'actions': actions,
    }
    
    # broadcast_data handles None on non-rank-0 TP processes
    data_b = broadcast_data(keys, data_dict, torch.bfloat16)
    
    return data_b['video_latents'], data_b['robot_states'], data_b['actions']


def loss_func(output_tensor):
    """Process loss returned by model forward.
    
    DreamZero forward already computes MSE flow-matching loss.
    We just report it.
    
    Args:
        output_tensor: scalar loss from model.forward()
        
    Returns:
        (loss, num_tokens_for_averaging, report_dict)
    """
    loss = output_tensor.mean()  # ensure scalar
    
    # Report loss for logging
    report = {'flow_matching_loss': loss.clone().detach()}
    
    # num_tokens = 1 since loss is already averaged
    return loss, torch.tensor(1, dtype=torch.long, device=loss.device), report


def forward_step(data_iterator, model):
    """Forward step for Megatron training loop.
    
    Args:
        data_iterator: yields batches from DreamZeroDataset
        model: DreamZeroMegatronModule
        
    Returns:
        (output_tensor, loss_func)
    """
    # Get batch (handles TP broadcast)
    video_latents, robot_states, actions = get_batch(data_iterator)
    
    # Forward pass - returns flow-matching loss
    output_tensor = model(
        video_latents=video_latents,
        robot_states=robot_states,
        actions=actions,
    )
    
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/valid/test datasets.
    
    Args:
        train_val_test_num_samples: tuple of (train_samples, valid_samples, test_samples)
        
    Returns:
        (train_dataset, valid_dataset, test_dataset)
    """
    from flagscale.train.datasets.dreamzero import DreamZeroDataset
    
    args = get_args()
    
    train_dataset = DreamZeroDataset(
        data_path=args.data_path[0] if isinstance(args.data_path, list) else args.data_path,
        seq_length=args.dreamzero_frame_seqlen,
        action_dim=args.dreamzero_action_dim,
        state_dim=args.dreamzero_state_dim,
    )
    
    # No validation/test for now
    return train_dataset, None, None


def add_dreamzero_args(parser):
    """Add DreamZero-specific arguments to Megatron args."""
    group = parser.add_argument_group(title='DreamZero')
    
    group.add_argument('--dreamzero-model-path', type=str, required=True,
                       help='Path to DreamZero pretrained checkpoint')
    group.add_argument('--dreamzero-action-dim', type=int, default=7,
                       help='Action dimension')
    group.add_argument('--dreamzero-state-dim', type=int, default=8,
                       help='Robot state dimension')
    group.add_argument('--dreamzero-frame-seqlen', type=int, default=220,
                       help='Number of frames in sequence')
    group.add_argument('--dreamzero-lora-rank', type=int, default=128,
                       help='LoRA rank')
    group.add_argument('--dreamzero-lora-alpha', type=int, default=256,
                       help='LoRA alpha')
    group.add_argument('--dreamzero-lora-targets', nargs='+',
                       default=['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2'],
                       help='LoRA target modules')
    
    return parser


if __name__ == "__main__":
    # Initialize Megatron and run pretraining
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_dreamzero_args,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'vocab_size': 1,  # Not used for DreamZero (continuous actions)
        },
    )
