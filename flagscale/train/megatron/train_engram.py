# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT Engram."""

from flagscale.models.megatron.engram.engram_builder import engram_builder
from train_gpt import main
from model_specific_batch import get_batch_for_model_with_all_stage_tokens


def get_batch(data_iterator, vp_stage=None, dualpipev_stage=None):
    return get_batch_for_model_with_all_stage_tokens(data_iterator, vp_stage, dualpipev_stage)


if __name__ == "__main__":
    main(engram_builder, get_batch_func=get_batch, build_dataset_on_all_pipeline_stages=True)
