import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, AutoModelForVision2Seq

# 引入 llmcompressor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier

# 引入我们刚才定义的 pipeline，确保它被注册
# 注意路径：根据你的实际 python path，可能需要调整 import
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipelines import mix_precision_pipeline

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="compress")
def main(cfg: DictConfig):
    print("Starting Mixed Precision Quantization (FlagScale Integrated)")
    
    model_path = cfg.model.model_path
    save_dir = cfg.model.save_dir
    
    # 1. 加载多模态模型 (适配 RoboBrain)
    print(f"Loading Model from: {model_path}")
    try:
        # 尝试加载 Processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载模型 (使用 Vision2Seq)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    # 2. 构建 Recipe
    # 我们需要同时定义 Quantization (8bit) 和 QuIP (4bit) 的配置，
    # 具体的选择逻辑交由 mix_precision_pipeline 处理。
    
    # 获取配置中的参数
    #targets = cfg.compress_args.targets # e.g., ["Linear"]
    #ignore = cfg.compress_args.ignore   # e.g., ["lm_head"]
    targets = cfg.compress.compress_args.targets
    ignore = cfg.compress.compress_args.ignore    

    recipe = [
        # 定义 QuIP 修改器 (作为候选项)
        QuIPModifier(
            targets=targets,
            ignore=ignore,
            scheme="W4A16", 
            targets_to_quantize_regex=None 
        ),
        # 定义标准量化修改器 (作为候选项)
        QuantizationModifier(
            targets=targets,
            ignore=ignore,
            scheme="W8A16",
            targets_to_quantize_regex=None
        ),
    ]

    print("Recipe constructed. Starting OneShot with 'mix_precision_search' pipeline...")

    # 3. 执行 OneShot
    # 关键点：dataset="mix_precision_search" 会触发我们在 pipeline.py 中注册的类
    oneshot(
        model=model,
        dataset="mix_precision_search",  # 对应 @CalibrationPipeline.register 的名字
        recipe=recipe,
        output_dir=save_dir,
        num_calibration_samples=0,       # Data-free 不需要样本
        max_seq_length=2048,
        pad_to_max_length=False,
    )

    print(f"Mixed Precision Quantization Complete! Saved to: {save_dir}")
    
    # 保存 Processor
    if processor:
        processor.save_pretrained(save_dir)

if __name__ == "__main__":
    main()

