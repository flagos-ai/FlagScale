"""PEFT (Parameter-Efficient Fine-Tuning) module.

This module provides implementations of various PEFT methods for fine-tuning
large language models efficiently.
"""

from megatron.training.peft.peft import PEFT, AdapterWrapper
from megatron.training.peft.lora import LoRA

__all__ = ['PEFT', 'AdapterWrapper', 'LoRA']

