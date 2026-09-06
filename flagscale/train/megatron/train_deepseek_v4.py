# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT DeepSeek-V4."""

from flagscale.models.megatron.deepseek_v4.deepseek_builder import deepseek_builder
from train_gpt import main
from model_specific_batch import get_batch_for_model_with_all_stage_tokens


def get_batch(data_iterator, vp_stage=None, dualpipev_stage=None):
    return get_batch_for_model_with_all_stage_tokens(data_iterator, vp_stage, dualpipev_stage)


if __name__ == "__main__":
    main(deepseek_builder, get_batch_func=get_batch, build_dataset_on_all_pipeline_stages=True)
