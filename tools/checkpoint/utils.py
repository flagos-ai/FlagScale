# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import psutil
import torch


@contextmanager
def suspend_nn_inits():
    """
    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None
    saved_inits = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
        torch.nn.init.xavier_uniform_,
    )  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = (
        torch.nn.init.xavier_uniform_
    ) = skip  # replacing
    try:
        yield
    finally:
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
            torch.nn.init.xavier_uniform_,
        ) = saved_inits  # restoring


def validate_args(args):
    pass


def padding_vocab_size(orig_word_embed, md, args, attr_name="padded_vocab_size"):
    vocab_size_attr = eval(f"args.{attr_name}")
    if md.true_vocab_size is not None:
        orig_vocab_size = orig_word_embed.shape[0]
        # Cut out extra padding we don't need
        if orig_vocab_size > vocab_size_attr:
            full_word_embed = orig_word_embed[0:vocab_size_attr, :]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < vocab_size_attr:
            padding_size = vocab_size_attr - orig_vocab_size

            full_word_embed = torch.cat(
                (orig_word_embed, orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1))
            )

        # Same size!
        else:
            full_word_embed = orig_word_embed
        print(f"> padding vocab_size from {orig_vocab_size} to {vocab_size_attr}")
    else:
        print(
            "Original vocab size not specified, leaving embedding table as-is. "
            "If you've changed the tensor parallel size this could cause problems."
        )
        setattr(args, attr_name, orig_word_embed.shape[0])
        full_word_embed = orig_word_embed
    return full_word_embed


def print_memory_usage(key, rank, num_ranks):
    """Print memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(
        f"> memory usage: '{key}', "
        f"rank {rank} / {num_ranks}, "
        f"mem {mem_info.rss / 1024**3:.1f}/{100 * mem_info.rss / process.memory_percent() / 1024**3:.1f} gb."
    )


def get_expert_tensor_parallel_size(args):
    return (
        getattr(args, "expert_tensor_parallel_size", None)
        or getattr(args, "tensor_model_parallel_size", 1)
        or 1
    )


def get_mcore_model_parallel_size(args):
    tp_size = getattr(args, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    ep_size = getattr(args, "expert_model_parallel_size", 1) or 1
    etp_size = get_expert_tensor_parallel_size(args)
    dense_parallel_size = tp_size * cp_size
    expert_parallel_size = ep_size * etp_size
    return max(dense_parallel_size, expert_parallel_size)


def validate_mcore_parallel_size(args):
    tp_size = getattr(args, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    ep_size = getattr(args, "expert_model_parallel_size", 1) or 1
    etp_size = get_expert_tensor_parallel_size(args)
    dense_parallel_size = tp_size * cp_size
    expert_parallel_size = ep_size * etp_size
    mcore_model_parallel_size = get_mcore_model_parallel_size(args)

    if mcore_model_parallel_size % dense_parallel_size != 0:
        raise ValueError(
            f"Unsupported MCore parallel sizes: max(tp*cp={dense_parallel_size}, "
            f"ep*etp={expert_parallel_size}) is not divisible by tp*cp."
        )
    if mcore_model_parallel_size % expert_parallel_size != 0:
        raise ValueError(
            f"Unsupported MCore parallel sizes: max(tp*cp={dense_parallel_size}, "
            f"ep*etp={expert_parallel_size}) is not divisible by ep*etp."
        )

    seen_checkpoint_names = set()
    for rank_id in range(mcore_model_parallel_size):
        checkpoint_rank = (
            get_tensor_model_parallel_rank(rank_id, args),
            get_expert_model_parallel_rank(rank_id, args),
        )
        if checkpoint_rank in seen_checkpoint_names:
            raise ValueError(
                "Unsupported MCore legacy torch layout: multiple physical ranks map to "
                f"checkpoint tensor/expert rank {checkpoint_rank}. "
                "Use expert_tensor_parallel_size <= tensor_model_parallel_size."
            )
        seen_checkpoint_names.add(checkpoint_rank)


def _get_parallel_order(args, sizes):
    order = (getattr(args, "order", None) or "tp-cp-ep-dp-pp").lower().split("-")
    for name, size in sizes.items():
        if name not in order and size != 1:
            raise ValueError(
                f"The checkpoint converter cannot map {name} size {size} because "
                f"parallel order {getattr(args, 'order', None)} does not include {name}."
            )
        if name not in order:
            order.append(name)
    return order


def _rank_coordinates(rank_id, sizes, order):
    stride = 1
    coords = {}
    for name in order:
        size = sizes[name]
        coords[name] = (rank_id // stride) % size
        stride *= size
    return coords


def _dense_rank_coordinates(rank_id, args):
    tp_size = getattr(args, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    model_parallel_size = get_mcore_model_parallel_size(args)
    dense_parallel_size = tp_size * cp_size
    sizes = {
        "tp": tp_size,
        "cp": cp_size,
        "ep": 1,
        "dp": model_parallel_size // dense_parallel_size,
        "pp": 1,
    }
    return _rank_coordinates(rank_id, sizes, _get_parallel_order(args, sizes))


def _expert_rank_coordinates(rank_id, args):
    ep_size = getattr(args, "expert_model_parallel_size", 1) or 1
    etp_size = get_expert_tensor_parallel_size(args)
    model_parallel_size = get_mcore_model_parallel_size(args)
    expert_parallel_size = ep_size * etp_size
    sizes = {
        "tp": etp_size,
        "cp": 1,
        "ep": ep_size,
        "dp": model_parallel_size // expert_parallel_size,
        "pp": 1,
    }
    return _rank_coordinates(rank_id, sizes, _get_parallel_order(args, sizes))


def get_tensor_model_parallel_rank(rank_id, args):
    return _dense_rank_coordinates(rank_id, args)["tp"]


def get_expert_model_parallel_rank(rank_id, args):
    return _expert_rank_coordinates(rank_id, args)["ep"]


def get_expert_tensor_parallel_rank(rank_id, args):
    return _expert_rank_coordinates(rank_id, args)["tp"]


def get_tensor_parallel_models(models, args):
    tp_size = getattr(args, "tensor_model_parallel_size", 1) or 1
    tp_models = [None] * tp_size
    for rank_id, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(rank_id, args)
        if tp_models[tp_rank] is None:
            tp_models[tp_rank] = model

    if any(model is None for model in tp_models):
        missing = [str(rank) for rank, model in enumerate(tp_models) if model is None]
        raise RuntimeError(f"Missing tensor-parallel model ranks: {', '.join(missing)}")
    return tp_models


def get_expert_tensor_parallel_models(models, args, ep_rank):
    etp_groups = get_expert_tensor_parallel_model_groups(models, args, ep_rank)
    return [group[0] for group in etp_groups]


def get_expert_tensor_parallel_model_groups(models, args, ep_rank):
    etp_size = get_expert_tensor_parallel_size(args)
    etp_groups = [[] for _ in range(etp_size)]
    for rank_id, model in enumerate(models):
        if get_expert_model_parallel_rank(rank_id, args) != ep_rank:
            continue
        etp_rank = get_expert_tensor_parallel_rank(rank_id, args)
        etp_groups[etp_rank].append(model)

    if any(not group for group in etp_groups):
        missing = [str(rank) for rank, group in enumerate(etp_groups) if not group]
        raise RuntimeError(
            f"Missing expert tensor-parallel model ranks for ep rank {ep_rank}: "
            f"{', '.join(missing)}"
        )
    return etp_groups


class _ConverterFakeProcessGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def set_rank(self, rank):
        self._rank = rank

    def set_size(self, size):
        self._size = size
