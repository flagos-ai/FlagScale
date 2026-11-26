# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Online SFT dataset built on top of ms-swift components.

This is the Phase-1 port of the ms-swift ``megatron sft`` data pipeline. It reuses
``swift`` as a library so behavior matches the validated training run exactly:

    load_dataset (jsonl + ``#N`` sampling)
      -> template.encode (tokenize + loss_scale, global rank 0 only)
      -> shared Arrow encoded cache (all ranks load)
      -> PackingDataset (cached best-fit-decreasing indices)

``PackingDataset.__getitem__`` returns a *list* of encoded rows (one packed
sequence). The list is flattened and collated by the ms-swift template's
``data_collator`` inside :mod:`swift_collator`, then converted to FlagScale's
Megatron batch keys.
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch.distributed as dist

from megatron.training import print_rank_0

from ._utils import disable_safe_ddp_synchronization, get_cache_dir

_ENCODED_CACHE_VERSION = 1
_PACKING_CACHE_VERSION = 2
_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


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
    dataset_num_proc: Optional[int] = 1
    load_from_cache_file: bool = True
    cache_dir: Optional[str] = None
    strict: bool = False
    data_seed: int = 42

    def __post_init__(self):
        if self.packing_length is None:
            self.packing_length = self.max_length


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str, *, content_hash: bool = False) -> dict:
    path = os.path.realpath(path)
    stat = os.stat(path)
    identity = {
        "path": path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        identity["sha256"] = _sha256_file(path)
    return identity


def _dataset_path(entry: str) -> str:
    """Strip ms-swift's optional ``#N`` suffix without misreading '#' in a path."""
    if os.path.exists(entry):
        return entry
    path, sep, sample = entry.rpartition("#")
    if sep and sample.isdigit() and os.path.exists(path):
        return path
    return entry


def _pipeline_code_digest() -> str:
    """Invalidate encoded rows when tokenizer/template/preprocessor code changes."""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.join(data_dir, "dataset"),
        os.path.join(data_dir, "loss_scale"),
        os.path.join(data_dir, "template"),
    ]
    files = [
        os.path.join(data_dir, "swift_processor_shim.py"),
        os.path.join(data_dir, "swift_sft_dataset.py"),
    ]
    for root in roots:
        for current_root, _, names in os.walk(root):
            files.extend(os.path.join(current_root, name) for name in names if name.endswith(".py"))
    digest = hashlib.sha256()
    for path in sorted(files):
        relative_path = os.path.relpath(path, data_dir).replace(os.sep, "/")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _cache_manifest(config: SwiftSFTConfig, dataset_paths: List[str], *, is_train: bool) -> dict:
    model_dir = os.path.realpath(config.model_dir)
    tokenizer_files = []
    for name in _TOKENIZER_FILES:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            # Tokenizer/config files are small enough to content-hash and are the
            # semantic input to template.encode. The large jsonl is identified by
            # path/size/mtime so a cache lookup does not itself scan all samples.
            tokenizer_files.append(_file_identity(path, content_hash=True))
    return {
        "version": _ENCODED_CACHE_VERSION,
        "pipeline_code_sha256": _pipeline_code_digest(),
        "is_train": is_train,
        "datasets": [
            {"entry": entry, "file": _file_identity(_dataset_path(entry))}
            for entry in dataset_paths
        ],
        "model_dir": model_dir,
        "tokenizer_files": tokenizer_files,
        "model_type": config.model_type,
        "template": config.template,
        "max_length": config.max_length,
        "truncation_strategy": config.truncation_strategy,
        "loss_scale": config.loss_scale,
        "add_non_thinking_prefix": config.add_non_thinking_prefix,
        "context_parallel_size": config.context_parallel_size,
        "split_dataset_ratio": config.split_dataset_ratio if is_train else 0.0,
        "data_seed": config.data_seed,
        "strict": config.strict,
    }


def _manifest_key(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_ready(path: str, expected_key: str) -> bool:
    manifest_path = os.path.join(path, "flagscale_cache.json")
    success_path = os.path.join(path, "_SUCCESS")
    if not os.path.isfile(manifest_path) or not os.path.isfile(success_path):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f).get("cache_key") == expected_key
    except (OSError, ValueError):
        return False


def _atomic_save_dataset(dataset, cache_path: str, manifest: dict, cache_key: str) -> None:
    """Publish a datasets ``save_to_disk`` directory only after it is complete."""
    parent = os.path.dirname(cache_path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = tempfile.mkdtemp(prefix=f".{cache_key}.", dir=parent)
    # save_to_disk expects to create its destination itself.
    os.rmdir(tmp_path)
    try:
        dataset.save_to_disk(tmp_path)
        with open(os.path.join(tmp_path, "flagscale_cache.json"), "w", encoding="utf-8") as f:
            json.dump({"cache_key": cache_key, "manifest": manifest}, f, sort_keys=True, indent=2)
        with open(os.path.join(tmp_path, "_SUCCESS"), "w", encoding="utf-8") as f:
            f.write("ok\n")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)


def _broadcast_cache_build_error(error: Optional[str]) -> None:
    """Synchronize a rank-0 cache build without unmatched barriers."""
    if not (dist.is_available() and dist.is_initialized()):
        if error is not None:
            raise RuntimeError(error)
        return
    status = [error]
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


def _build_template(config: SwiftSFTConfig):
    """Build a stand-alone template (tokenizer only, no model weights)."""
    from .swift_processor_shim import build_processor
    from .template import get_template

    # Only the tokenizer plus a handful of config-derived fields are needed to
    # encode; see swift_processor_shim for why swift.model is not used here.
    processor = build_processor(config.model_dir, config.model_type, config.template)
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
        from datasets import load_from_disk

        from .dataset import EncodePreprocessor, PackingDataset, load_dataset

        self.config = config
        self.is_train = is_train
        self.template = _build_template(config)

        cache_root = config.cache_dir or os.path.join(get_cache_dir(), "datasets", "flagscale_sft")
        manifest = _cache_manifest(config, dataset_paths, is_train=is_train)
        encoded_key = _manifest_key(manifest)
        encoded_cache_path = os.path.join(cache_root, "encoded", encoded_key)

        def build_encoded_dataset():
            # Online load + `#N` sampling + concat. split_dataset_ratio only
            # makes sense for the training split (val is passed explicitly).
            split_ratio = config.split_dataset_ratio if is_train else 0.0
            # This function is called by global rank 0 only. Disable the older,
            # nested dataset barriers; the outer status broadcast below is the
            # single synchronization point for all ranks.
            with disable_safe_ddp_synchronization():
                train_dataset, _ = load_dataset(
                    dataset_paths,
                    split_dataset_ratio=split_ratio,
                    seed=config.data_seed,
                    num_proc=config.dataset_num_proc,
                    load_from_cache_file=config.load_from_cache_file,
                    strict=config.strict,
                    streaming=False,
                )
                raw_len = len(train_dataset)
                # Persist the complete encoded rows. The former AddLength + Lazy
                # pipeline encoded every row once for lengths and again on access.
                # Keeping input_ids/labels/loss_scale makes later experiments and
                # every training epoch tokenizer-free.
                encoded = EncodePreprocessor(template=self.template)(
                    train_dataset,
                    num_proc=config.dataset_num_proc,
                    load_from_cache_file=False,
                    strict=config.strict,
                    batch_size=1000,
                )
            manifest["raw_len"] = raw_len
            manifest["encoded_len"] = len(encoded)
            _atomic_save_dataset(encoded, encoded_cache_path, manifest, encoded_key)

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if config.load_from_cache_file:
            error = None
            if rank == 0 and not _cache_ready(encoded_cache_path, encoded_key):
                try:
                    from filelock import FileLock

                    lock_path = os.path.join(cache_root, "locks", f"encoded-{encoded_key}.lock")
                    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
                    with FileLock(lock_path):
                        # Another experiment may have populated this shared key
                        # while this rank was waiting for the lock.
                        if not _cache_ready(encoded_cache_path, encoded_key):
                            print_rank_0(
                                f"> building shared Swift SFT encoded cache: {encoded_cache_path}"
                            )
                            build_encoded_dataset()
                except Exception as exc:  # propagate instead of hanging peers in a barrier
                    error = f"Failed to build Swift SFT encoded cache {encoded_cache_path}: {exc!r}"
            _broadcast_cache_build_error(error)
            if not _cache_ready(encoded_cache_path, encoded_key):
                raise RuntimeError(
                    f"Swift SFT encoded cache is not visible on rank {rank}: {encoded_cache_path}. "
                    "Use a cache directory shared by every training node."
                )
            hf_dataset = load_from_disk(encoded_cache_path)
            cache_manifest_file = os.path.join(encoded_cache_path, "flagscale_cache.json")
            with open(cache_manifest_file, encoding="utf-8") as f:
                saved_manifest = json.load(f)["manifest"]
            self._raw_len = saved_manifest.get("raw_len", len(hf_dataset))
            if rank == 0:
                print_rank_0(f"> loaded Swift SFT encoded cache: {encoded_cache_path}")
        else:
            # Cache-disabled mode preserves the old per-rank construction
            # behavior, but still avoids the second lazy encode within a run.
            split_ratio = config.split_dataset_ratio if is_train else 0.0
            with disable_safe_ddp_synchronization():
                train_dataset, _ = load_dataset(
                    dataset_paths,
                    split_dataset_ratio=split_ratio,
                    seed=config.data_seed,
                    num_proc=config.dataset_num_proc,
                    load_from_cache_file=False,
                    strict=config.strict,
                    streaming=False,
                )
                self._raw_len = len(train_dataset)
                hf_dataset = EncodePreprocessor(template=self.template)(
                    train_dataset,
                    num_proc=config.dataset_num_proc,
                    load_from_cache_file=False,
                    strict=config.strict,
                    batch_size=1000,
                )

        packing_manifest = {
            "version": _PACKING_CACHE_VERSION,
            "encoded_key": encoded_key,
            "dataset_len": len(hf_dataset),
            "packing_length": config.packing_length,
            "packing_strategy": config.packing_strategy,
            # Packing splits the input into one independent chunk per process,
            # so changing this value can change pack boundaries.
            "packing_num_proc": config.packing_num_proc,
            # ms-swift's _handle_megatron_cp pads every encoded row before
            # packing_row concatenates the rows. Packing must therefore budget
            # the same physical lengths or a raw-length-valid pack can cross
            # packing_length and be rounded up to an entire extra pack.
            "packing_alignment": 2 * config.context_parallel_size,
        }
        packing_key = _manifest_key(packing_manifest)
        packing_cache_file = os.path.join(cache_root, "packing", f"{packing_key}.pkl")

        if config.packing:
            self.dataset = PackingDataset(
                self.template,
                hf_dataset,
                num_proc=config.dataset_num_proc,
                packing_length=config.packing_length,
                packing_num_proc=config.packing_num_proc,
                packing_strategy=config.packing_strategy,
                packing_alignment=packing_manifest["packing_alignment"],
                strict=config.strict,
                load_from_cache_file=config.load_from_cache_file,
                packing_cache_file=packing_cache_file,
                packing_cache_key=packing_key,
            )
        else:
            self.dataset = hf_dataset

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
