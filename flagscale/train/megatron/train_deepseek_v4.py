# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT DeepSeek-V4."""

from flagscale.models.megatron.deepseek_v4.deepseek_builder import deepseek_builder
from train_gpt import main


if __name__ == "__main__":
    main(deepseek_builder)
