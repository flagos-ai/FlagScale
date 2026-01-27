import torch
from typing import List, Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from flagscale.logger import logger

# 尝试导入必要的库，处理不同版本的路径差异
try:
    from llmcompressor.modifiers.quantization import QuantizationModifier
except ImportError:
    QuantizationModifier = None

try:
    # 优先尝试从 transformers 导入 oneshot
    from llmcompressor.transformers import oneshot
except ImportError:
    try:
        # 备选：尝试从根目录或其他路径导入
        from llmcompressor import oneshot
    except ImportError:
        oneshot = None

try:
    from llmcompressor.modifiers import ScheduledModifierManager
except ImportError:
    ScheduledModifierManager = None


class LLMCompressorAdapter:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        if hasattr(config, "compress") and hasattr(config.compress, "compress_args"):
             self.compress_args = config.compress.compress_args
        else:
             self.compress_args = config.get("compress_args", config)

    def run(self):
        logger.info("Starting LLMCompressor Adapter...")
        
        if QuantizationModifier is None:
            raise ImportError("Could not import QuantizationModifier from llmcompressor.modifiers.quantization")

        targets = self.compress_args.get("targets", ["Linear"])
        if hasattr(targets, "to_container"): 
            targets = targets.to_container()
        
        ignore_layers = self.compress_args.get("ignore", [])
        if hasattr(ignore_layers, "to_container"): 
            ignore_layers = ignore_layers.to_container()

        quant_config_list = self.compress_args.get("quantization", [])
        if not quant_config_list:
            logger.warning("No quantization config found.")
            return

        q_cfg = quant_config_list[0]
        scheme_name = q_cfg.get("scheme", "W8A16")

        logger.info(f"Applying Scheme: {scheme_name}")
        
        # 初始化 Modifier
        # 注意：scheme 必须是字符串
        modifier = QuantizationModifier(
            targets=targets,
            ignore=ignore_layers,
            scheme=scheme_name
        )

        logger.info("Applying quantization modifier...")

        # 策略 1: 使用 oneshot (首选)
        if oneshot is not None:
            logger.info("Using 'oneshot' API.")
            oneshot(
                model=self.model,
                recipe=modifier,
            )
        
        # 策略 2: 使用 ScheduledModifierManager (备选)
        elif ScheduledModifierManager is not None:
            logger.info("Using 'ScheduledModifierManager' API.")
            manager = ScheduledModifierManager([modifier])
            manager.apply(self.model)
            
        # 策略 3: 如果 Modifier 有 apply 方法 (旧版)
        elif hasattr(modifier, "apply"):
            logger.info("Using 'modifier.apply' directly.")
            modifier.apply(self.model)
            
        else:
            raise ImportError(
                "Could not find a valid method to apply the quantization. "
                "'oneshot' not found in llmcompressor.transformers, "
                "and ScheduledModifierManager not available."
            )
        
        logger.info("Quantization complete.")

    def save(self, save_dir: str):
        logger.info(f"Saving quantized model to {save_dir}...")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved successfully.")

