# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Online SFT dataset built on top of ms-swift components.

This is the Phase-1 port of the ms-swift ``megatron sft`` data pipeline. It reuses
``swift`` as a library so behavior matches the validated training run exactly:

    load_dataset (jsonl + ``#N`` sampling)
      -> template.encode (tokenize + loss_scale)
      -> AddLengthPreprocessor (compute per-sample token lengths)
      -> LazyLLMDataset (lazy re-tokenize, skip over-long samples)
      -> PackingDataset (best-fit-decreasing bin packing to packing_length)

``PackingDataset.__getitem__`` returns a *list* of encoded rows (one packed
sequence). The list is flattened and collated by the ms-swift template's
``data_collator`` inside :mod:`swift_collator`, then converted to FlagScale's
Megatron batch keys.
"""

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from megatron.training import print_rank_0


@dataclass
class SwiftSFTConfig:
    """Configuration for :class:`SwiftSFTDataset`.

    The defaults mirror ``exp/train_dsv4_flash_multinode.sh`` so a FlagScale run
    reproduces the ms-swift behavior out of the box.
    """

    # model / template
    model_dir: str
    model_type: str = "deepseek_v4"
    template: str = "deepseek_v4_flash"
    max_length: int = 65536
    truncation_strategy: str = "delete"  # raise | left | right | delete | split
    loss_scale: str = "default+ignore_empty_think"
    add_non_thinking_prefix: bool = True

    # datasets: entries may carry ms-swift ``#N`` sampling syntax, e.g. "a.jsonl#512"
    dataset: List[str] = field(default_factory=list)
    val_dataset: List[str] = field(default_factory=list)
    split_dataset_ratio: float = 0.0

    # packing / padding_free. ms-swift's megatron default is padding_free=True; the
    # collator in :mod:`swift_collator` only supports that (packed) layout.
    padding_free: bool = True
    packing: bool = True
    packing_length: Optional[int] = None  # defaults to max_length
    packing_strategy: str = "binpack"  # binpack | sequential
    packing_num_proc: int = 1

    # parallelism (affects CP padding inside the template)
    context_parallel_size: int = 1

    # dataset processing
    dataset_num_proc: int = 1
    load_from_cache_file: bool = True
    strict: bool = False
    data_seed: int = 42

    def __post_init__(self):
        if self.packing_length is None:
            self.packing_length = self.max_length


def _build_template(config: SwiftSFTConfig):
    """Build a stand-alone ms-swift template (tokenizer only, no model weights)."""
    from swift.model import get_processor
    from swift.template import get_template

    # ``get_processor`` wraps ``get_model_processor(load_model=False)``, giving us
    # the processor/tokenizer without materializing the (huge) model on this rank
    # -- tokenization only needs the tokenizer.
    processor = get_processor(
        config.model_dir,
        model_type=config.model_type,
        download_model=False,
    )
    # ms-swift arg-level ``delete`` means "drop over-long samples". The template
    # itself only accepts raise/left/right/split, so (matching ms-swift
    # ``template_args.py``) we map ``delete`` -> ``raise`` here; over-long rows
    # then raise ``MaxLengthError`` and are skipped by the LazyLLMDataset layer.
    template_truncation_strategy = config.truncation_strategy
    if template_truncation_strategy == "delete":
        template_truncation_strategy = "raise"
    template = get_template(
        processor,
        max_length=config.max_length,
        template_type=config.template,
        truncation_strategy=template_truncation_strategy,
        loss_scale=config.loss_scale,
        add_non_thinking_prefix=config.add_non_thinking_prefix,
        padding_side="right",
        sequence_parallel_size=config.context_parallel_size,
        # ms-swift's megatron path defaults `padding_free=True`
        # (``megatron/arguments/megatron_args.py``). Without it the collator pads
        # every row to the batch max and emits a [B, T] batch plus a materialized
        # [B, 1, T, T] attention_mask, instead of the [1, T] packed sequence with
        # restarting position_ids that this pipeline's cu_seqlens derivation and
        # FlagScale's varlen attention both require.
        padding_free=config.padding_free,
    )
    template.set_mode("train")
    # Route the collator down the Megatron (padding_free / packed) code path.
    template.use_megatron = True
    return template


class SwiftSFTDataset:
    """Map-style SFT dataset producing packed sequences.

    When ``packing`` is enabled, ``__getitem__`` returns a ``list`` of encoded
    rows (a single packed sequence). Otherwise it returns a single encoded row.
    The associated ``template`` (exposed as an attribute) must be used to build
    the collator; see :func:`~.swift_collator.build_swift_sft_collator`.
    """

    def __init__(self, config: SwiftSFTConfig, dataset_paths: List[str], *, is_train: bool):
        from swift.dataset import AddLengthPreprocessor, LazyLLMDataset, PackingDataset, load_dataset

        self.config = config
        self.is_train = is_train
        self.template = _build_template(config)

        # 1. Online load + `#N` sampling + concat. split_dataset_ratio only makes
        #    sense for the training split (val is passed explicitly).
        split_ratio = config.split_dataset_ratio if is_train else 0.0
        train_dataset, val_dataset = load_dataset(
            dataset_paths,
            split_dataset_ratio=split_ratio,
            seed=config.data_seed,
            num_proc=config.dataset_num_proc,
            load_from_cache_file=config.load_from_cache_file,
            strict=config.strict,
            streaming=False,
        )
        # For an explicit val_dataset (split_ratio == 0) the samples land in the
        # "train" slot; a val split is only returned when split_ratio > 0, which
        # we only use for the training path.
        hf_dataset = train_dataset
        self._raw_len = len(hf_dataset)

        # 2. Compute per-sample token lengths (writes a `lengths` column) so the
        #    packer can bin-pack without tokenizing twice up front.
        add_length = AddLengthPreprocessor(template=self.template)
        batch_size = 1000
        hf_dataset = add_length(
            hf_dataset,
            num_proc=config.dataset_num_proc,
            load_from_cache_file=config.load_from_cache_file,
            strict=config.strict,
            batch_size=batch_size,
        )

        # 3. Lazy re-tokenize on access (drops over-long / malformed rows), then
        #    bin-pack to `packing_length`.
        lazy_dataset = LazyLLMDataset(
            hf_dataset,
            self.template.encode,
            strict=config.strict,
            random_state=config.data_seed,
        )
        if config.packing:
            self.dataset = PackingDataset(
                self.template,
                lazy_dataset,
                num_proc=config.dataset_num_proc,
                packing_length=config.packing_length,
                packing_num_proc=config.packing_num_proc,
                packing_strategy=config.packing_strategy,
                strict=config.strict,
                load_from_cache_file=config.load_from_cache_file,
            )
        else:
            self.dataset = lazy_dataset

        print_rank_0(
            f"> SwiftSFTDataset ({'train' if is_train else 'valid'}): "
            f"{self._raw_len} raw samples -> {len(self.dataset)} "
            f"{'packs' if config.packing else 'samples'} "
            f"(packing_length={config.packing_length})"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx]


def build_swift_sft_datasets(config: SwiftSFTConfig):
    """Build (train, valid) :class:`SwiftSFTDataset` instances.

    ``valid`` is ``None`` when no validation dataset is configured.
    """
    assert config.dataset, "SwiftSFTConfig.dataset must not be empty"
    train = SwiftSFTDataset(config, config.dataset, is_train=True)
    valid = None
    if config.val_dataset:
        valid = SwiftSFTDataset(config, config.val_dataset, is_train=False)
    return train, valid
