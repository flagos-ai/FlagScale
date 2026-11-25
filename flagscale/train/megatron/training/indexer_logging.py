
"""Indexer metrics for the FlagScale training loop."""

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossLoggingHelper,
    is_dsa_skip_topk_layer,
)


def _get_indexer_logging_layer_counts(args) -> tuple[int, int | None]:
    """Count tracker slots and indexer modules. Repeated depths add loss to a shared slot."""
    mtp_layers = args.mtp_num_layers or 0
    tracker_layers = args.num_layers + mtp_layers
    ratios = getattr(args, "csa_compress_ratios", None)
    if ratios is not None:
        if getattr(args, "hybrid_layer_pattern", None):
            from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern

            pattern = parse_hybrid_pattern(args.hybrid_layer_pattern)
            tracker_layers = args.num_layers + len(pattern.mtp_pattern or "")
            main_indexers = sum(ratio == 4 for ratio in ratios[: args.num_layers])
            mtp_indexers = sum(ratio == 4 for ratio in ratios[args.num_layers : tracker_layers])
            repeats = (
                1 if getattr(args, "mtp_use_repeated_layer", False) else pattern.mtp_num_depths
            )
            count = main_indexers + mtp_indexers * repeats
        else:
            count = sum(ratio == 4 for ratio in ratios[:tracker_layers])
        return tracker_layers, 0 if getattr(args, "csa_dense_mode", False) else count

    hybrid_pattern = getattr(args, "hybrid_layer_pattern", None)
    if getattr(args, "experimental_attention_variant", None) != "dsa" and not hybrid_pattern:
        return tracker_layers, None

    indexer_types = getattr(args, "indexer_types", None)
    rule = getattr(args, "indexer_type_rule", None)
    if indexer_types is None and rule is not None:
        from megatron.core.transformer.transformer_config import TransformerConfig

        indexer_types = TransformerConfig._parse_indexer_type_rule(rule, args.num_layers)
    topk_freq = getattr(args, "dsa_indexer_topk_freq", 1) or 1
    skip_offset = getattr(args, "dsa_indexer_skip_topk_offset", 0) or 0

    def count_indexers(num_layers, layer_types=None):
        return sum(
            (layer_types is None or layer_types[layer - 1] == "D")
            and (
                indexer_types[layer - 1] not in ("share", "shared")
                if indexer_types is not None
                else not is_dsa_skip_topk_layer(layer, skip_offset, topk_freq)
            )
            for layer in range(1, num_layers + 1)
        )

    if hybrid_pattern:
        from megatron.core.models.hybrid.hybrid_layer_allocation import parse_hybrid_pattern

        pattern = parse_hybrid_pattern(hybrid_pattern)
        main_types = pattern.main_pattern.replace("|", "")
        mtp_types = pattern.mtp_pattern or ""
        tracker_layers = args.num_layers + len(mtp_types)
        repeats = 1 if getattr(args, "mtp_use_repeated_layer", False) else pattern.mtp_num_depths
        return tracker_layers, count_indexers(len(main_types), main_types) + (
            count_indexers(len(mtp_types), mtp_types) * repeats
        )

    # MTP uses its own local layer numbering and can reuse one module at every depth.
    mtp_modules = (
        min(mtp_layers, 1) if getattr(args, "mtp_use_repeated_layer", False) else mtp_layers
    )
    return tracker_layers, count_indexers(args.num_layers) + count_indexers(mtp_modules)


def log_indexer_metrics(
    args, loss_scale, iteration, writer, wandb_writer, total_loss_dict, pg_collection
):
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["pp", "dp", "cp"]
        )
    slots, count = _get_indexer_logging_layer_counts(args)
    DSAIndexerLossLoggingHelper.track_indexer_metrics(
        loss_scale=loss_scale,
        iteration=iteration,
        writer=writer,
        wandb_writer=wandb_writer,
        total_loss_dict=total_loss_dict,
        pg_collection=pg_collection,
        num_layers=slots,
        num_indexer_layers=count,
        preserve_groups=args.cuda_graph_impl != "none",
        calculate_per_token_loss=args.calculate_per_token_loss,
    )
