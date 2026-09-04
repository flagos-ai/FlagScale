#!/usr/bin/env python3
"""Convert released BAGEL safetensors to a TP=1 Megatron-FSDP checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _option_value(argv: list[str], name: str) -> str | None:
    for index, argument in enumerate(argv):
        if argument == name and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def _has_option(argv: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in argv)


def inject_checkpoint_config_args(argv: list[str]) -> None:
    """Fill Megatron defaults from the released BAGEL config.json."""

    source_value = _option_value(argv, "--bagel-source-checkpoint")
    if source_value is None:
        return
    source = Path(source_value)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"source checkpoint config does not exist: {config_path}")
    config = json.loads(config_path.read_text())
    llm = config["llm_config"]
    vit = config["vit_config"]
    vae = config["vae_config"]

    values = {
        "--save-interval": 1,
        "--ckpt-format": "fsdp_dtensor",
        "--data-parallel-sharding-strategy": "optim_grads_params",
        "--tensor-model-parallel-size": 1,
        "--pipeline-model-parallel-size": 1,
        "--context-parallel-size": 1,
        "--expert-model-parallel-size": 1,
        "--distributed-backend": "nccl",
        "--megatron-fsdp-main-params-dtype": "fp32",
        "--megatron-fsdp-main-grads-dtype": "bf16",
        "--megatron-fsdp-grad-comm-dtype": "bf16",
        "--language-model-type": "qwen2.5_7B",
        "--vision-model-type": "siglip",
        "--model-provider": "bagel_vlm_t2i",
        "--dataset-provider": "bagel_vlm_t2i",
        "--num-layers": llm["num_hidden_layers"],
        "--hidden-size": llm["hidden_size"],
        "--ffn-hidden-size": llm["intermediate_size"],
        "--num-attention-heads": llm["num_attention_heads"],
        "--num-query-groups": llm["num_key_value_heads"],
        "--seq-length": min(llm["max_position_embeddings"], 30720),
        "--max-position-embeddings": llm["max_position_embeddings"],
        "--position-embedding-type": "rope",
        "--rotary-base": int(llm["rope_theta"]),
        "--normalization": "RMSNorm",
        "--norm-epsilon": llm["rms_norm_eps"],
        "--init-method-std": llm["initializer_range"],
        "--transformer-impl": "transformer_engine",
        "--patch-dim": vit["patch_size"],
        "--image-size": vit["image_size"],
        "--max-latent-size": config["max_latent_size"] * config["latent_patch_size"],
        "--latent-patch-size": config["latent_patch_size"],
        "--latent-channel": vae["z_channels"],
        "--vae-downsample": vae["downsample"],
        "--vit-max-num-patch-per-side": config["vit_max_num_patch_per_side"],
        "--timestep-shift": config["timestep_shift"],
        "--tokenizer-type": "BagelTokenizerFS",
        "--tokenizer-path": source,
        "--tokenizer-prompt-format": "qwen2p5",
        "--vocab-size": llm["vocab_size"],
        "--make-vocab-size-divisible-by": 64,
        "--micro-batch-size": 1,
        "--global-batch-size": int(os.environ.get("WORLD_SIZE", "1")),
        "--train-iters": 1,
        "--lr": 1e-4,
    }
    flags = [
        "--use-megatron-fsdp",
        "--group-query-attention",
        "--swiglu",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--untie-embeddings-and-output-weights",
        "--no-rope-fusion",
        "--no-save-optim",
        "--no-save-rng",
        "--no-gradient-accumulation-fusion",
    ]
    if llm.get("torch_dtype") == "bfloat16" or config.get("torch_dtype") == "bfloat16":
        flags.append("--bf16")
    if llm.get("qk_norm"):
        flags.append("--qk-layernorm")
    if config.get("visual_und"):
        flags.append("--visual-und")
    if config.get("visual_gen"):
        flags.append("--visual-gen")
    if config.get("interpolate_pos"):
        flags.append("--interpolate-pos")

    for name, value in values.items():
        if not _has_option(argv, name):
            argv.extend((name, str(value)))
    for flag in flags:
        if not _has_option(argv, flag):
            argv.append(flag)



def add_conversion_args(parser):
    """Add BAGEL model arguments plus conversion-only controls."""

    from flagscale.train.megatron.train_bagel import add_bagel_extra_args

    parser = add_bagel_extra_args(parser)
    group = parser.add_argument_group(title="BAGEL checkpoint conversion")
    group.add_argument(
        "--bagel-source-checkpoint",
        type=str,
        required=True,
        help="Directory containing ema.safetensors, ae.safetensors, and config.json.",
    )
    group.add_argument(
        "--bagel-conversion-audit-only",
        action="store_true",
        help="Build and audit the FSDP model without injecting or saving weights.",
    )
    return parser


def validate_conversion_args(args) -> tuple[Path, Path | None]:
    """Validate the deliberately narrow TP=1 Megatron-FSDP conversion scope."""

    source = Path(args.bagel_source_checkpoint)
    required = ("ema.safetensors", "ae.safetensors", "config.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source checkpoint is missing required files: {missing}")
    if not args.use_megatron_fsdp:
        raise ValueError("BAGEL conversion requires use_megatron_fsdp=true")
    if args.ckpt_format != "fsdp_dtensor":
        raise ValueError(
            f"BAGEL Megatron-FSDP conversion requires ckpt_format=fsdp_dtensor, got {args.ckpt_format}"
        )
    parallel_sizes = {
        "tensor": args.tensor_model_parallel_size,
        "pipeline": args.pipeline_model_parallel_size,
        "context": args.context_parallel_size,
        "expert": args.expert_model_parallel_size,
    }
    invalid = {name: size for name, size in parallel_sizes.items() if size != 1}
    if invalid:
        raise ValueError(f"BAGEL converter currently supports only TP=PP=CP=EP=1, got {invalid}")
    if getattr(args, "load", None):
        raise ValueError("conversion run must not set checkpoint.load; input comes from safetensors")

    if args.bagel_conversion_audit_only:
        return source, None
    if not args.save:
        raise ValueError("checkpoint.save must specify the new fsdp_dtensor output directory")
    output = Path(args.save)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output path: {output}")
    return source, output


def _state_dict_for_audit(model):
    state_dict_fn = getattr(model, "state_dict_for_save_checkpoint", None)
    if state_dict_fn is None:
        state_dict_fn = model.state_dict
    return state_dict_fn()


def _audit_failure_message(audit) -> str:
    details = {
        "expected_count": audit.expected_count,
        "actual_count": audit.actual_count,
        "matched_count": audit.matched_count,
        "missing": audit.missing[:20],
        "unexpected": audit.unexpected[:20],
        "shape_mismatches": audit.shape_mismatches[:20],
    }
    return f"target BagelModel state dict mismatch: {details}"


def main() -> None:
    inject_checkpoint_config_args(sys.argv)

    import torch
    from megatron.core.enums import ModelType
    from megatron.core.process_groups_config import ProcessGroupCollection

    from flagscale.models.megatron.bagel.model_providers.bagel_vlm import model_provider_bagel_vlm_t2i
    from flagscale.train.megatron.training import get_args, print_rank_0
    from flagscale.train.megatron.training.initialize import initialize_megatron
    from flagscale.train.megatron.training.training import get_model
    from flagscale.train.megatron.training.checkpointing import save_checkpoint

    from tools.checkpoint.bagel.fsdp_inject import (
        inject_checkpoint,
        install_optimized_model_weights,
    )
    from tools.checkpoint.bagel.model_state_audit import audit_model_state_dict

    initialize_megatron(extra_args_provider=add_conversion_args)
    args = get_args()
    source, output = validate_conversion_args(args)

    # Match the released embedding/output tensors rather than re-padding the
    # tokenizer vocabulary (151665 would otherwise become 151680).
    from safetensors import safe_open
    with safe_open(str(source / "ema.safetensors"), framework="pt", device="cpu") as handle:
        args.padded_vocab_size = handle.get_slice(
            "language_model.model.embed_tokens.weight"
        ).get_shape()[0]

    # The converter always produces a model-only, synchronous release checkpoint.
    args.no_save_optim = True
    args.no_save_rng = True
    args.async_save = False

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def conversion_model_provider(**kwargs):
        # get_model() forwards config, but BAGEL builds configs from Megatron args.
        kwargs.pop("config", None)
        return model_provider_bagel_vlm_t2i(**kwargs)

    model = get_model(
        conversion_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=True,
        pg_collection=pg_collection,
    )
    if len(model) != 1:
        raise ValueError(f"TP=PP=1 conversion expects one model chunk, got {len(model)}")

    audit = audit_model_state_dict(source, _state_dict_for_audit(model[0]))
    print_rank_0(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
    if not audit.ok:
        raise RuntimeError(_audit_failure_message(audit))
    if args.bagel_conversion_audit_only:
        print_rank_0("BAGEL conversion audit completed; no checkpoint was written.")
        return

    report = inject_checkpoint(model[0], source)
    install_optimized_model_weights(model[0])
    print_rank_0(
        f"Injected {len(report.copied)} tensors; "
        f"regenerated {len(report.regenerated)} position tensors."
    )

    # A second audit checks names and global shapes after FSDP local-slice writes.
    post_audit = audit_model_state_dict(source, _state_dict_for_audit(model[0]))
    if not post_audit.ok:
        raise RuntimeError("post-injection BagelModel state dict audit failed")

    save_checkpoint(
        iteration=0,
        model=model,
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
        release=False,
        tp_group=pg_collection.tp,
        pp_group=pg_collection.pp,
        dp_cp_group=pg_collection.dp_cp,
    )
    torch.distributed.barrier()
    print_rank_0(f"Saved BAGEL fsdp_dtensor checkpoint to {output}")


if __name__ == "__main__":
    main()
