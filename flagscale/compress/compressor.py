import argparse
import os
import yaml
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForVision2Seq, AutoModelForImageTextToText
# 确保 adapter 在路径中
import sys
sys.path.append(os.getcwd()) 

from flagscale.compress.adapter import LLMCompressorAdapter
from flagscale.logger import logger

_g_ignore_fields = ["experiment", "action", "job"]

def prepare_config(config_path):
    with open(config_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    for key in _g_ignore_fields:
        if key in yaml_dict:
            yaml_dict.pop(key)

    config = OmegaConf.create(yaml_dict)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    cfg = prepare_config(args.config_path)
    
    if hasattr(cfg, "compress"):
        root_cfg = cfg.compress
        model_args = root_cfg.model
    else:
        model_args = cfg.get("model", cfg)
        root_cfg = cfg

    model_path = model_args.get("model_path")
    model_cls_name = model_args.get("model_cls", "AutoModelForCausalLM")
    
    save_dir = "output_model"
    if hasattr(root_cfg, "system"):
        save_dir = root_cfg.system.get("save_dir", "output_model")
    
    if not os.path.isabs(save_dir) and hasattr(cfg, "experiment"):
         save_dir = os.path.join(cfg.experiment.exp_dir, save_dir)

    logger.info(f"Loading model: {model_path} ({model_cls_name})")

    if model_cls_name == "AutoModelForVision2Seq":
        try:
            ModelClass = AutoModelForImageTextToText
        except AttributeError:
            ModelClass = AutoModelForVision2Seq
    elif model_cls_name == "AutoModelForImageTextToText":
        ModelClass = AutoModelForImageTextToText
    else:
        ModelClass = AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = ModelClass.from_pretrained(
        model_path, 
        torch_dtype="auto", 
        device_map="auto", 
        trust_remote_code=True
    )

    adapter = LLMCompressorAdapter(model, tokenizer, cfg)
    adapter.run()

    adapter.save(save_dir)

if __name__ == "__main__":
    main()

