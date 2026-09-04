# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain BAGEL multimodal model for unified image understanding and generation."""

import os

import torch
from megatron.core import parallel_state
from megatron.core.enums import ModelType

from megatron.training import (
    get_args,
    get_timers,
    pretrain,
    print_rank_0,
)

from flagscale.models.megatron.bagel.models import BagelModel
from flagscale.models.megatron.bagel.model_providers import model_provider_bagel_vlm_t2i

from flagscale.train.megatron.training.multimodal_args import add_multimodal_extra_args
from flagscale.train.datasets.energon.energon_bagel_task_encoder import bagel_vlm_dataloader_provider

from flagscale.logger import logger

_MODEL_PROVIDERS = {
    "bagel_vlm_t2i": model_provider_bagel_vlm_t2i,
}


_DATASET_PROVIDERS = {
    "bagel_vlm_t2i": bagel_vlm_dataloader_provider,
}


def count_parameters(module: torch.nn.Module) -> tuple[int, int, int]:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def print_trainable_parameters(module: torch.nn.Module, module_name: str = "Model", logger=None):
    """打印可训练参数的详细信息"""
    trainable_params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.shape, param.numel()))
    
    if logger:
        logger.info(f"\n{'='*80}")
        logger.info(f"{module_name} 可训练参数详情 (共 {len(trainable_params)} 个参数组):")
        logger.info(f"{'='*80}")
        total_trainable = 0
        for name, shape, numel in trainable_params:
            total_trainable += numel
            logger.info(f"  {name:80s} | Shape: {str(shape):30s} | Params: {numel:>15,} ({numel/1e6:.2f}M)")
        logger.info(f"{'='*80}")
        logger.info(f"总计可训练参数: {total_trainable:,} ({total_trainable/1e9:.2f}B)")
        logger.info(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"{module_name} 可训练参数详情 (共 {len(trainable_params)} 个参数组):")
        print(f"{'='*80}")
        total_trainable = 0
        for name, shape, numel in trainable_params:
            total_trainable += numel
            print(f"  {name:80s} | Shape: {str(shape):30s} | Params: {numel:>15,} ({numel/1e6:.2f}M)")
        print(f"{'='*80}")
        print(f"总计可训练参数: {total_trainable:,} ({total_trainable/1e9:.2f}B)")
        print(f"{'='*80}\n")


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    args.use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')

            filename = f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
            torch.cuda.memory._dump_snapshot(filename)

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    try:
        builder_fn = _MODEL_PROVIDERS[args.model_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported model provider '{args.model_provider}'. "
            f"Available providers: {list(_MODEL_PROVIDERS.keys())}"
        ) from e

    model = builder_fn(pre_process=pre_process, post_process=post_process, pg_collection=pg_collection)
    total_params, total_trainable, total_frozen = count_parameters(model)
    lm_params, lm_trainable, lm_frozen = count_parameters(model.language_model)
    print_rank_0(f"Model parameter count: {total_params / 1e9:.2f}B (Trainable: {total_trainable / 1e9:.2f}B, Frozen: {total_frozen / 1e9:.2f}B)")
    print_rank_0(f"LM parameter count:    {lm_params / 1e9:.2f}B (Trainable: {lm_trainable / 1e9:.2f}B, Frozen: {lm_frozen / 1e9:.2f}B)")
    print_trainable_parameters(model, "完整模型", logger)
    print_trainable_parameters(model.language_model, "语言模型", logger)

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):

    args = get_args()
    try:
        dataset_provider = _DATASET_PROVIDERS[args.dataset_provider]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dataset provider '{args.dataset_provider}'. "
            f"Available providers: {list(_DATASET_PROVIDERS.keys())}"
        ) from e

    return dataset_provider(train_val_test_num_samples)


def _load_alignment_batch(path):
    """Load and validate a CPU-only batch dumped by original BAGEL."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "sequence_length", "sample_lens", "split_lens", "attn_modes",
        "packed_text_ids", "packed_text_indexes", "packed_position_ids",
        "packed_latent_clean", "packed_latent_noise",
        "packed_timesteps_effective",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"alignment batch is missing required fields: {missing}")
    metadata = data.get("_alignment_metadata", {})
    if metadata.get("source") != "original_bagel":
        raise ValueError(f"unexpected alignment batch source: {metadata!r}")
    return data


def get_batch(data_iterator):
    """Generate a batch from data iterator.

    The dataloader yields a dict from BagelPackedBatch.to_dict() with fields:
      - Always present: sequence_length, sample_lens, packed_text_ids,
        packed_text_indexes, packed_position_ids, split_lens, attn_modes
      - VAE (optional): padded_images, patchified_vae_latent_shapes,
        packed_latent_position_ids, packed_vae_token_indexes
      - ViT (optional): packed_vit_tokens, packed_vit_position_ids,
        packed_vit_token_indexes, vit_token_seqlens
      - Diffusion (optional): packed_timesteps, mse_loss_indexes
      - CE loss (optional): packed_label_ids, ce_loss_indexes, ce_loss_weights

    Returns:
        Dictionary with all batch tensors needed for BAGEL forward.
    """
    args = get_args()

    # Currently only support TP=1, PP=1 — no cross-rank broadcast needed.
    assert (getattr(args, 'pipeline_model_parallel_size', 1) == 1)
    assert (getattr(args, 'tensor_model_parallel_size', 1) == 1)

    alignment_batch_path = os.getenv("BAGEL_ALIGN_BATCH")
    if alignment_batch_path:
        data = _load_alignment_batch(alignment_batch_path)
    else:
        data = next(data_iterator)
    # print(f"{data=}")

    # --- Assemble batch dict ---
    batch = {
        # Core text fields (always present)
        'packed_text_ids': data['packed_text_ids'].cuda(non_blocking=True),
        'packed_text_indexes': data['packed_text_indexes'].cuda(non_blocking=True),
        'packed_position_ids': data['packed_position_ids'].cuda(non_blocking=True),
        # Non-tensor metadata (needed for FlexAttention mask construction)
        'sequence_length': data['sequence_length'],
        'sample_lens': data['sample_lens'],
        'split_lens': data['split_lens'],
        'attn_modes': data['attn_modes'],
        'attention_mask': None,
    }

    # --- CE loss fields ---
    if 'packed_label_ids' in data:
        batch['packed_label_ids'] = data['packed_label_ids'].cuda(non_blocking=True)
    if 'ce_loss_indexes' in data:
        batch['ce_loss_indexes'] = data['ce_loss_indexes'].cuda(non_blocking=True)
    if 'ce_loss_weights' in data:
        batch['ce_loss_weights'] = data['ce_loss_weights'].cuda(non_blocking=True)

    # --- VAE image generation fields ---
    if 'padded_images' in data:
        batch['padded_images'] = data['padded_images'].cuda(non_blocking=True)
    if 'padded_latent' in data:
        batch['padded_latent'] = data['padded_latent'].cuda(non_blocking=True)
    if 'packed_latent_clean' in data:
        batch['packed_latent_clean'] = data['packed_latent_clean'].cuda(non_blocking=True)
    if 'packed_latent_noise' in data:
        batch['packed_latent_noise'] = data['packed_latent_noise'].cuda(non_blocking=True)
    if 'packed_timesteps_effective' in data:
        batch['packed_timesteps_effective'] = data['packed_timesteps_effective'].cuda(non_blocking=True)
    if 'patchified_vae_latent_shapes' in data:
        batch['patchified_vae_latent_shapes'] = data['patchified_vae_latent_shapes']
    if 'packed_latent_position_ids' in data:
        batch['packed_latent_position_ids'] = data['packed_latent_position_ids'].cuda(non_blocking=True)
    if 'packed_vae_token_indexes' in data:
        batch['packed_vae_token_indexes'] = data['packed_vae_token_indexes'].cuda(non_blocking=True)

    # --- ViT image understanding fields ---
    if 'packed_vit_tokens' in data:
        batch['packed_vit_tokens'] = data['packed_vit_tokens'].cuda(non_blocking=True)
    if 'packed_vit_position_ids' in data:
        batch['packed_vit_position_ids'] = data['packed_vit_position_ids'].cuda(non_blocking=True)
    if 'packed_vit_token_indexes' in data:
        batch['packed_vit_token_indexes'] = data['packed_vit_token_indexes'].cuda(non_blocking=True)
    if 'vit_token_seqlens' in data:
        batch['vit_token_seqlens'] = data['vit_token_seqlens'].cuda(non_blocking=True)

    # --- Diffusion timesteps ---
    if 'packed_timesteps' in data:
        batch['packed_timesteps'] = data['packed_timesteps'].cuda(non_blocking=True)
    if 'mse_loss_indexes' in data:
        batch['mse_loss_indexes'] = data['mse_loss_indexes'].cuda(non_blocking=True)

    # print(f"{batch=}")
    return batch


def bagel_loss_func(output_tensor, model=None):
    """Loss function for BAGEL that combines CE and MSE losses.

    Args:
        output_tensor: Combined loss tensor from forward_step.
        model: The model (unused).

    Returns:
        Tuple of (loss, num_tokens, report_dict).
    """
    losses = output_tensor.view(-1).float()
    loss = losses.sum()
    num_tokens = torch.ones((), device=loss.device, dtype=torch.int)

    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    return loss, num_tokens, report


def forward_step(data_iterator, model: BagelModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model: BagelModel instance.

    Returns:
        Tuple of the combined loss tensor and Megatron loss callback.
    """
    args = get_args()
    timers = get_timers()

    # Get the batch
    timers('batch-generator', log_level=2).start()
    batch = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Forward pass — batch keys match model forward signature directly
    output = model(
        packed_text_ids=batch['packed_text_ids'],
        packed_text_indexes=batch['packed_text_indexes'],
        packed_position_ids=batch['packed_position_ids'],
        attention_mask=batch.get('attention_mask'),
        sequence_length=batch.get('sequence_length'),
        sample_lens=batch.get('sample_lens'),
        split_lens=batch.get('split_lens'),
        attn_modes=batch.get('attn_modes'),
        # CE loss
        packed_label_ids=batch.get('packed_label_ids'),
        ce_loss_indexes=batch.get('ce_loss_indexes'),
        # ViT understanding
        packed_vit_tokens=batch.get('packed_vit_tokens'),
        packed_vit_token_indexes=batch.get('packed_vit_token_indexes'),
        packed_vit_position_ids=batch.get('packed_vit_position_ids'),
        vit_token_seqlens=batch.get('vit_token_seqlens'),
        # VAE generation
        padded_images=batch.get('padded_images'),
        padded_latent=batch.get('padded_latent'),
        packed_latent_clean=batch.get('packed_latent_clean'),
        packed_latent_noise=batch.get('packed_latent_noise'),
        packed_timesteps_effective=batch.get('packed_timesteps_effective'),
        patchified_vae_latent_shapes=batch.get('patchified_vae_latent_shapes'),
        packed_latent_position_ids=batch.get('packed_latent_position_ids'),
        packed_vae_token_indexes=batch.get('packed_vae_token_indexes'),
        packed_timesteps=batch.get('packed_timesteps'),
        mse_loss_indexes=batch.get('mse_loss_indexes'),
    )

    ce, mse, dummy = output
    device = batch['packed_text_ids'].device
    ce_loss_indexes = batch.get('ce_loss_indexes')
    ce_loss_weights = batch.get('ce_loss_weights')
    local_ce_tokens = (
        ce_loss_indexes.numel()
        if ce is not None and ce_loss_indexes is not None
        else 0
    )
    local_ce_loss_weights = (
        ce_loss_weights.float().sum()
        if getattr(args, 'ce_loss_reweighting', False) and ce_loss_weights is not None and local_ce_tokens > 0
        else 0
    )

    mse_loss_indexes = batch.get('mse_loss_indexes')
    local_mse_tokens = (
        mse_loss_indexes.numel()
        if getattr(args, 'visual_gen', False) and mse is not None and mse_loss_indexes is not None
        else 0
    )

    total_tokens = torch.tensor(
        [local_ce_tokens, local_ce_loss_weights, local_mse_tokens],
        device=device, dtype=torch.float64
    )
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    dp_world_size = torch.distributed.get_world_size(group=dp_group)
    torch.distributed.all_reduce(total_tokens, group=dp_group)
    (
        total_ce_tokens,
        total_ce_loss_weights,
        total_mse_tokens,
    ) = total_tokens

    combined_loss = torch.tensor(0.0, device=device)
    if local_ce_tokens > 0 and total_ce_tokens.item() > 0:
        if getattr(args, 'ce_loss_reweighting', False):
            assert ce_loss_weights is not None, "ce_loss_reweighting=True but ce_loss_weights is missing"
            ce = ce * ce_loss_weights
            ce = ce.sum() * dp_world_size / total_ce_loss_weights.clamp_min(1e-12)
        else:
            ce = ce.sum() * dp_world_size / total_ce_tokens
        combined_loss = combined_loss + ce * getattr(args, 'ce_weight', 1.0)

    if local_mse_tokens > 0 and total_mse_tokens.item() > 0:
        mse = mse.mean(dim=-1).sum() * dp_world_size / total_mse_tokens
        combined_loss = combined_loss + mse * getattr(args, 'mse_weight', 1.0)

    # print(f"{ce=}, {mse=}")

    combined_loss = combined_loss + dummy

    # if torch.distributed.get_rank() == 0:
    #     print(
    #         "[BAGEL_LOSS_GRAPH]",
    #         f"ce_shape={None if ce is None else tuple(ce.shape)}",
    #         f"ce_grad_fn={None if ce is None else ce.grad_fn}",
    #         f"mse_shape={None if mse is None else tuple(mse.shape)}",
    #         f"mse_grad_fn={None if mse is None else mse.grad_fn}",
    #         f"combined_grad_fn={combined_loss.grad_fn}",
    #         flush=True,
    #     )

    return combined_loss.unsqueeze(0), bagel_loss_func


def add_bagel_extra_args(parser):
    """Add BAGEL-specific command-line arguments."""
    parser = add_multimodal_extra_args(parser)
    group = parser.add_argument_group(title='BAGEL model arguments')

    group.add_argument(
        '--dataset-provider', type=str,
        default='mock', help='Dataset provider to choose from [mock, bagel_vlm]'
    )
    group.add_argument(
        '--model-provider', type=str,
        default='mock', help='Model provider to choose from [mock, bagel_vlm]'
    )

    # Modality switches
    group.add_argument(
        '--visual-gen', action='store_true',
        help='Enable visual generation path',
    )
    group.add_argument(
        '--visual-und', action='store_true',
        help='Enable visual understanding path',
    )
    group.add_argument(
        '--interpolate-pos', action='store_true',
        help="Interpolate positional embeddings when image resolution differs from pre-training."
    )

    # VAE latent configuration
    group.add_argument(
        '--latent-patch-size', type=int, default=2,
        help='Spatial size (in VAE pixels) covered by each latent patch.',
    )
    group.add_argument(
        '--max-latent-size', type=int, default=32,
        help='Maximum latent grid size (patches per side) for the VAE latent tensor.',
    )
    group.add_argument(
        '--latent-channel', type=int, default=16,
        help='Number of VAE latent channels',
    )
    group.add_argument(
        '--vae-downsample', type=int, default=16,
        help='VAE spatial downsampling factor',
    )
    group.add_argument(
        '--timestep-shift', type=float, default=1.0,
        help='Flow matching timestep shift parameter',
    )

    # ViT configuration
    group.add_argument(
        '--image-size', type=int, default=70,
        help='Maximum Image size for ViTs position embedding',
    )
    group.add_argument(
        '--vit-max-num-patch-per-side', type=int, default=70,
        help='Maximum ViT patches per side for position embedding',
    )

    # Loss weights
    group.add_argument(
        '--ce-weight', type=float, default=1.0,
        help='Weight for cross-entropy (understanding) loss',
    )
    group.add_argument(
        '--mse-weight', type=float, default=1.0,
        help='Weight for MSE (generation) loss',
    )
    group.add_argument(
        '--ce-loss-reweighting', action='store_true', default=False,
        help='Reweight CE loss using per-token ce_loss_weights.',
    )

    # Random preprocessing controls. Keep these explicit so alignment runs do
    # not silently fall back to BagelDataConfig defaults.
    group.add_argument(
        '--text-cond-dropout-prob', type=float, default=0.0,
        help='Probability of dropping text conditioning during packing.',
    )
    group.add_argument(
        '--vit-cond-dropout-prob', type=float, default=0.0,
        help='Probability of dropping ViT conditioning during packing.',
    )
    group.add_argument(
        '--vae-cond-dropout-prob', type=float, default=0.0,
        help='Probability of dropping VAE conditioning during packing.',
    )


    # Freeze controls
    group.add_argument(
        '--freeze-VAE', action='store_true', default=False,
        help='Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder.'
    )
    group.add_argument(
        '--freeze-text-embed', action='store_true', default=False,
        help='Freeze text embedding and lm_head (embed_tokens, lm_head).'
    )
    group.add_argument(
        '--freeze-connect', action='store_true', default=False,
        help='Freeze vit and vae connector modules (vit connector, vae2llm, llm2vae, time_embedder).'
    )
    group.add_argument(
        '--freeze-und', action='store_true', default=False,
        help='Freeze the visual understanding connector layers by requires_grad=False.'
    )
    group.add_argument(
        '--freeze-gen', action='store_true', default=False,
        help='Freeze the image generation connector layers by requires_grad=False.'
    )

    # for dataset
    group.add_argument(
        '--vit-patch-size', type=int, default=14,
        help='Patch size (pixels) for the Vision Transformer encoder.'
    )
    # for packing data
    group.add_argument(
        '--expected-num-tokens', type=int, default=32768,
        help='Soft target token count per packed batch; yield once reached.',
    )
    group.add_argument(
        '--max-num-tokens', type=int, default=36864,
        help='Hard upper limit on tokens in a packed batch.',
    )
    group.add_argument(
        '--max-num-tokens-per-sample', type=int, default=16384,
        help='Maximum tokens allowed in one raw sample; longer samples are skipped.',
    )
    group.add_argument(
        '--max-buffer-size', type=int, default=50,
        help='Maximum number of overflow samples kept by BAGEL packing.',
    )
    group.add_argument(
        "--prefer-buffer-before",
        type=int,
        default=16384,
        help="Prefer FIFO overflow samples while current packed length is below this threshold.",
    )

    return parser


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_bagel_extra_args,
    )
