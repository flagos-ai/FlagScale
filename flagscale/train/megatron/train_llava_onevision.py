# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain or SFT LLaVA-NeXT model."""
from copy import deepcopy
from functools import partial
import os
import sys
import warnings

import torch

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)

from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from flagscale.models.megatron.llava_onevision.config import (
    get_language_model_config,
    get_vision_model_config,
    get_vision_projection_config,
)
from flagscale.models.megatron.llava_onevision.llava_onevision_model import (
    LLaVAOneVisionModel,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from flagscale.models.megatron.llava_onevision.layer_specs import (
    get_layer_spec,
    get_mlp_module_spec,
    get_layer_spec_te,
)
from megatron.training.utils import average_losses_across_data_parallel_group
from flagscale.models.megatron.llava_onevision.dataloader_provider import (
    train_valid_test_dataloaders_provider,
)
from megatron.training.training import pretrain


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
) -> LLaVAOneVisionModel:
    """Builds the model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable parallel model output.

    Returns:
        model: A multimodal model.
    """
    args = get_args()
    args.use_te = args.transformer_impl == "transformer_engine"
    use_te = args.use_te

    print_rank_0("building a llava onevision model ...")

    num_image_tokens = get_image_token_count()

    old_seq_length = args.seq_length
    args.decoder_seq_length = args.max_position_embeddings
    # This seq_length is used for the vision model.
    args.seq_length = num_image_tokens
    if torch.distributed.get_rank() == 0:
        warnings.warn(
            "Changed decoder_seq_length to num_image_tokens ({num_image_tokens}) + user-specified seq_length ({old_seq_length})."
        )

    base_config = core_transformer_config_from_args(get_args())
    base_config.language_model_type = args.language_model_type

    language_config = deepcopy(base_config)
    language_config = get_language_model_config(language_config)

    if use_te:
        language_transformer_layer_spec = get_layer_spec_te(is_vit=False)
    else:
        language_transformer_layer_spec = get_layer_spec(
            is_vit=False, normalization=args.normalization
        )

    vision_config = deepcopy(base_config)
    vision_config.vision_model_type = args.vision_model_type
    vision_config = get_vision_model_config(
        vision_config, apply_query_key_layer_scaling=args.apply_query_key_layer_scaling
    )

    if use_te:
        vision_transformer_layer_spec = get_layer_spec_te(is_vit=True)
    else:
        vision_transformer_layer_spec = get_layer_spec(
            is_vit=True, normalization="LayerNorm"
        )

    vision_projection_config = deepcopy(base_config)
    vision_projection_config = get_vision_projection_config(
        vision_projection_config, language_config.hidden_size
    )

    if args.encoder_pipeline_model_parallel_size > 0:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "ViT can only live on 1 pipeline stage."
        vision_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        vision_projection_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )
            vision_projection_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )

    vision_projection_layer_spec = get_mlp_module_spec(use_te=use_te).submodules

    model = LLaVAOneVisionModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        vision_projection_type="mlp",
        allow_missing_vision_projection_checkpoint=args.allow_missing_vision_projection_checkpoint,
        parallel_output=parallel_output,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
        add_class_token=not args.disable_vision_class_token,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )

    # Print model for debugging.
    if args.use_te:
        print(f"LLaVA OneVision Model with TE: ", model)
    else:
        print(f"LLaVA OneVision Model without TE: ", model)
    return model


def get_batch(data_iterator):
    """Generate a batch"""

    args = get_args()

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    input_ids = tensor_parallel.broadcast_data(["input_ids"], data, torch.int64)[
        "input_ids"
    ]
    input_ids_shape = tensor_parallel.broadcast_data(
        ["input_ids_shape"], data, torch.int64
    )["input_ids_shape"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    labels_shape = tensor_parallel.broadcast_data(["labels_shape"], data, torch.int64)[
        "labels_shape"
    ]
    images = tensor_parallel.broadcast_data(["images"], data, torch.float32)["images"]
    split_image_sizes = tensor_parallel.broadcast_data(
        ["split_image_sizes"], data, torch.int64
    )["split_image_sizes"]
    image_sizes = tensor_parallel.broadcast_data(["image_sizes"], data, torch.int64)[
        "image_sizes"
    ]
    modalities = tensor_parallel.broadcast_data(["modalities"], data, torch.int64)[
        "modalities"
    ]

    # Convert to list
    # input_ids to list
    input_ids_list = []
    start_idx = 0
    for shape in input_ids_shape:
        num_elements = torch.prod(shape).item()
        sub_tensor = input_ids[start_idx : start_idx + num_elements].reshape(
            shape.tolist()
        )
        input_ids_list.append(sub_tensor)
        start_idx += num_elements
    assert start_idx == input_ids.numel()
    input_ids = input_ids_list

    # labels to list
    labels_list = []
    start_idx = 0
    for shape in labels_shape:
        num_elements = torch.prod(shape).item()
        sub_tensor = labels[start_idx : start_idx + num_elements].reshape(
            shape.tolist()
        )
        labels_list.append(sub_tensor)
        start_idx += num_elements
    assert start_idx == labels.numel()
    labels = labels_list

    # images to list
    images_list = []
    start_idx = 0
    for shape in split_image_sizes:
        num_elements = torch.prod(shape).item()
        sub_tensor = images[start_idx : start_idx + num_elements].reshape(
            shape.tolist()
        )
        images_list.append(sub_tensor)
        start_idx += num_elements
    assert start_idx == images.numel()
    images = images_list

    # image_sizes to list
    image_sizes = list(torch.split(image_sizes, 1, dim=0))
    image_sizes = [size.squeeze(0) for size in image_sizes]
    image_sizes = [[size[0].item(), size[1].item()] for size in image_sizes]

    # modalities to List[str]
    modalities = list(torch.split(modalities, 1, dim=0))
    modalities_list = []
    for modality in modalities:
        if modality.item() == 0:
            modalities_list.append("image")
        elif modality.item() == 1:
            modalities_list.append("video")
        elif modality.item() == 2:
            modalities_list.append("text")
        else:
            raise ValueError(f"Unknown modality {modality.item()}")
    modalities = modalities_list

    torch.cuda.nvtx.range_pop()

    tokenizer = get_tokenizer()
    if not hasattr(tokenizer, "pad_token_id"):
        if hasattr(tokenizer, "pad_id"):
            tokenizer.pad_token_id = tokenizer.pad_id
        else:
            tokenizer.pad_token_id = 0
    if not hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"

    # Padding input_ids and labels
    torch.cuda.nvtx.range_push("pad_sequence_and_attn_mask")
    # Truncation and padding to the max len
    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        tokenizer=tokenizer,
    )
    labels = pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX, tokenizer=tokenizer
    )
    # Attention mask same as LLaVA-NeXT
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    torch.cuda.nvtx.range_pop()

    return input_ids, labels, attention_mask, images, image_sizes, modalities


def pad_sequence(input_ids, batch_first, padding_value, tokenizer):
    # Refer to the padding of LLaVA-NeXT.
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=batch_first, padding_value=padding_value
    )
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids


def get_image_token_count():
    args = get_args()

    add_class_token = not args.disable_vision_class_token

    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    num_image_tokens = num_patches + (1 if add_class_token else 0)

    return num_image_tokens


def loss_func(labels: torch.Tensor, loss_mask: torch.Tensor, logits: torch.Tensor):
    labels = labels.transpose(0, 1).contiguous()  # [b s] => [s b]
    logits = logits.transpose(0, 1).contiguous()  # [b s h] => [s b h]

    shift_logits = logits[:-1, :, :].contiguous()
    shift_labels = labels[1:, ...].contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(
        shift_logits.float(), shift_labels
    )
    losses = losses.transpose(0, 1).contiguous().float()
    if loss_mask is not None:
        loss_mask = loss_mask[..., 1:].contiguous()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / max(1, loss_mask.sum())
    else:
        loss = torch.mean(losses)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model: LLaVAOneVisionModel):
    """Forward training step.

    Args:
        data_iterator (torch.utils.data.dataloader): Input data iterator
        model: Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    input_ids, labels, attention_mask, images, image_sizes, modalities = get_batch(
        data_iterator
    )
    if "text" in modalities and ("image" in modalities or "video" in modalities):
        raise ValueError(
            "Both text and other modalities are present in the same batch."
        )
    timers("batch-generator").stop()

    output_tensor, labels, loss_mask = model(
        input_ids, labels, attention_mask, images, image_sizes, modalities
    )

    return output_tensor, partial(loss_func, labels, loss_mask)


def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title="multimodal arguments")
    group.add_argument(
        "--valid-path",
        nargs="*",
        default=None,
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-weight dataset1-path dataset2-weight "
        "dataset2-path ...",
    )
    group.add_argument("--dataset-config", type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument("--freeze-LM", action="store_true", default=False)
    group.add_argument("--freeze-ViT", action="store_true", default=False)
    group.add_argument("--language-model-type", type=str, required=True)
    group.add_argument(
        "--disable-vision-class-token", action="store_true", default=False
    )
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint",
        action="store_true",
        default=False,
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument(
        "--dataloader-save",
        type=str,
        default=None,
        help="Energon dataloader state save path",
    )

    # LLaVA OneVision specific arguments
    group.add_argument(
        "--interleaved-dataset",
        action="store_true",
        default=False,
        help="Offline dataset with InterleavedSample",
    )
    group.add_argument(
        "--training-dataset-only",
        action="store_true",
        default=False,
        help="Only training dataset",
    )
    group.add_argument("--vision-model-type", default="clip", help="Vision model type")
    group.add_argument(
        "--image-aspect-ratio", type=str, default="square", help="Image aspect ratio"
    )
    group.add_argument(
        "--mm-patch-merge-type",
        type=str,
        default="flat",
        help="Multimodal patch merge type",
    )
    group.add_argument(
        "--image-grid-pinpoints", type=str, default=None, help="Image grid pinpoints"
    )
    group.add_argument(
        "--use-pos-skipping",
        action="store_true",
        default=False,
        help="Use position skipping",
    )
    group.add_argument(
        "--pos-skipping-range", type=int, default=4096, help="Position skipping range"
    )
    group.add_argument(
        "--add-faster-video", default=False, help="Whetehr add fatser video token"
    )
    group.add_argument(
        "--mm-spatial-pool-mode", type=str, default="bilinear", help="Spatial pool mode"
    )
    group.add_argument(
        "--mm-newline-position", type=str, default="grid", help="Newline position."
    )
    return parser


def llava_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=add_multimodal_extra_args,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
