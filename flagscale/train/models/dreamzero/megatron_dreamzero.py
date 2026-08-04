# Copyright (c) 2025, FlagScale Authors. All rights reserved.
"""
DreamZero Megatron-native model for distributed training with TP/DP.

This module wraps the DreamZero DiT model (CausalWanModel) for use with
Megatron's pretrain() loop. The frozen encoders (SigLIP, T5) and VAE are
wrapped as-is (no TP sharding — they're frozen and relatively small).
The DiT backbone gets LoRA injection for training.

Architecture:
- Frozen: SigLIP vision encoder, T5 text encoder, WanVAE
- Trainable: DiT (CausalWanModel) via LoRA on attention Q/K/V/O + FFN

TP strategy: All TP ranks hold full replicated weights (via HuggingFace-style
wrapping with gradient averaging). This is pragmatic for LoRA training where
only ~3% of params are trainable. Full native TP on DiT can be a future
optimization.
"""

import os
import sys
import math
import logging
from functools import partial
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def get_megatron_imports():
    """Lazy import Megatron modules to avoid import errors during testing."""
    from megatron.training import get_args
    from megatron.core import mpu, tensor_parallel
    from megatron.core.models.common.embeddings.language_model_embedding import (
        LanguageModelEmbedding,
    )
    return get_args, mpu, tensor_parallel


# ---------------------------------------------------------------------------
# Model Provider
# ---------------------------------------------------------------------------

def model_provider(pre_process=True, post_process=True):
    """Build the DreamZero model for Megatron training.
    
    Called by Megatron's pretrain() to construct the model.
    For single-stage (no PP), both pre_process and post_process are True.
    
    Returns:
        MegatronDreamZero model instance
    """
    from megatron.training import get_args
    args = get_args()
    
    model = MegatronDreamZero(
        checkpoint_path=args.load,
        pre_process=pre_process,
        post_process=post_process,
        lora_rank=getattr(args, 'lora_rank', 128),
        lora_alpha=getattr(args, 'lora_alpha', 256),
        frame_seqlen=getattr(args, 'frame_seqlen', 220),
    )
    return model


# ---------------------------------------------------------------------------
# get_batch: Data loading with TP broadcast
# ---------------------------------------------------------------------------

def get_batch(data_iterator):
    """Get a batch of data and broadcast across TP ranks.
    
    Only TP rank 0 reads from data_iterator. All tensors are then
    broadcast to other TP ranks to ensure identical inputs.
    
    Returns dict with keys:
        - video_latents: [B, C, T, H, W] VAE-encoded video
        - robot_states: [B, T_action, state_dim]
        - actions: [B, T_action, action_dim] ground truth actions
        - task_embeddings: [B, seq_len, embed_dim] T5 text embeddings
        - image_embeddings: [B, num_patches, embed_dim] SigLIP image embeddings
    """
    from megatron.training import get_args
    from megatron.core import mpu
    from megatron.core.tensor_parallel.data import broadcast_data
    
    args = get_args()
    
    # Keys and their dtypes for broadcast
    keys_dtypes = {
        'video_latents': torch.float32,
        'robot_states': torch.float32,
        'actions': torch.float32,
        'task_embeddings': torch.float32,
        'image_embeddings': torch.float32,
    }
    
    # Only TP rank 0 reads data
    if mpu.get_tensor_model_parallel_rank() == 0:
        assert data_iterator is not None
        data = next(data_iterator)
        # Move to GPU
        batch = {k: data[k].cuda(non_blocking=True) for k in keys_dtypes if k in data}
    else:
        batch = {k: None for k in keys_dtypes}
    
    # Broadcast all tensors from TP rank 0 to all TP ranks
    batch = broadcast_data(list(keys_dtypes.keys()), batch, 
                           torch.float32)
    
    return batch


# ---------------------------------------------------------------------------
# forward_step + loss_func
# ---------------------------------------------------------------------------

def loss_func(output_tensor):
    """Compute loss from model output.
    
    For DreamZero, the model forward() already computes the flow-matching
    MSE loss. We just need to extract and report it.
    
    Args:
        output_tensor: scalar loss from model.forward()
    
    Returns:
        (loss, num_tokens_for_averaging, report_dict)
    """
    loss = output_tensor.mean()  # already scalar, but ensure
    
    # Report dict for logging
    report = {
        'flow_loss': loss.clone().detach(),
    }
    
    # num_tokens = 1 since loss is already averaged
    return loss, 1, report


def forward_step(data_iterator, model):
    """Forward step for Megatron training loop.
    
    Args:
        data_iterator: yields batches of training data
        model: MegatronDreamZero instance
    
    Returns:
        (output_tensor, loss_func_partial)
    """
    batch = get_batch(data_iterator)
    
    # Forward pass — model computes flow-matching loss
    output_tensor = model(
        video_latents=batch['video_latents'],
        robot_states=batch['robot_states'],
        actions=batch['actions'],
        task_embeddings=batch['task_embeddings'],
        image_embeddings=batch['image_embeddings'],
    )
    
    return output_tensor, partial(loss_func)


# ---------------------------------------------------------------------------
# MegatronDreamZero Model
# ---------------------------------------------------------------------------

class MegatronDreamZero(nn.Module):
    """DreamZero model wrapped for Megatron training.
    
    This is a pragmatic approach: we load the full DreamZero model
    (with frozen encoders + LoRA on DiT) and wrap it in the Megatron
    training interface. TP is handled via gradient averaging on replicated
    params rather than weight sharding.
    
    For memory efficiency with 8x80GB GPUs:
    - Frozen params: ~23B (replicated per GPU but no gradients)
    - LoRA params: ~700M trainable (gradients stored)
    - Total per GPU: ~46GB (bf16 weights) + ~2.8GB (LoRA gradients+optimizer)
    - With gradient checkpointing: fits in 80GB
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        pre_process: bool = True,
        post_process: bool = True,
        lora_rank: int = 128,
        lora_alpha: int = 256,
        frame_seqlen: int = 220,
    ):
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.frame_seqlen = frame_seqlen
        
        # Load the DreamZero model from checkpoint
        self._build_model(checkpoint_path)
    
    def _build_model(self, checkpoint_path: str):
        """Load DreamZero VLA model and inject LoRA."""
        from flagscale.train.models.dreamzero.dreamzero_model import (
            from_pretrained_dreamzero,
            DreamZeroConfig,
        )
        
        config = DreamZeroConfig(
            model_path=checkpoint_path,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            frame_seqlen=self.frame_seqlen,
        )
        
        # Load model — this handles:
        # 1. Loading base VLA weights from safetensors
        # 2. Injecting LoRA on DiT attention/FFN
        # 3. Freezing everything except LoRA
        # 4. Setting frame_seqlen on all submodules
        self.policy = from_pretrained_dreamzero(config)
        
        # Set frozen modules to eval mode
        self.policy.set_frozen_modules_to_eval()
        
        # Log param counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MegatronDreamZero: {total/1e9:.2f}B total, "
                    f"{trainable/1e6:.1f}M trainable ({100*trainable/total:.2f}%)")
    
    def set_input_tensor(self, input_tensor):
        """Required by Megatron for PP. No-op for single stage."""
        pass
    
    def forward(
        self,
        video_latents: torch.Tensor,
        robot_states: torch.Tensor,
        actions: torch.Tensor,
        task_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing flow-matching loss.
        
        Args:
            video_latents: [B, C, T, H, W] VAE-encoded video frames
            robot_states: [B, T, state_dim] robot proprioception
            actions: [B, T, action_dim] ground truth actions
            task_embeddings: [B, seq_len, D] T5 text embeddings
            image_embeddings: [B, N, D] SigLIP vision embeddings
            
        Returns:
            Scalar flow-matching MSE loss
        """
        # Set frozen parts to eval (BN, dropout)
        if hasattr(self.policy, 'set_frozen_modules_to_eval'):
            self.policy.set_frozen_modules_to_eval()
        
        # The policy's forward computes flow-matching loss
        loss = self.policy(
            video_latents=video_latents,
            robot_states=robot_states,
            actions=actions,
            task_embeddings=task_embeddings,
            image_embeddings=image_embeddings,
        )
        
        return loss


# ---------------------------------------------------------------------------
# Dataset provider (placeholder — real impl in datasets/dreamzero/)
# ---------------------------------------------------------------------------

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/valid/test datasets for Megatron.
    
    This is called by pretrain() to get datasets. We return the
    DreamZero dataset for train, None for valid/test.
    """
    from megatron.training import get_args
    from flagscale.train.datasets.dreamzero import DreamZeroDataset
    
    args = get_args()
    
    train_ds = DreamZeroDataset(
        data_path=args.data_path[0] if isinstance(args.data_path, list) else args.data_path,
        seq_length=getattr(args, 'frame_seqlen', 220),
    )
    
    return train_ds, None, None
