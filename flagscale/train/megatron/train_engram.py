# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT Engram."""

from flagscale.models.megatron.engram.engram_builder import engram_builder
from train_gpt import main


if __name__ == "__main__":
    main(engram_builder)
