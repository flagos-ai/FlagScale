import argparse
import os
import sys
import yaml
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForVision2Seq
from torch.utils.data import DataLoader

# 1. Import adapter module first
import flagscale.compress.adapter
from flagscale.compress.adapter import LLMCompressorAdapter

# --- Monkey Patch Start (critical fix) ---
# Since adapter.py calls oneshot internally, patch the reference inside the adapter module directly
# This ensures our wrapper is called regardless of the original import source
if hasattr(flagscale.compress.adapter, "oneshot"):
    print(">> [Patch] Found 'oneshot' in adapter, applying fix...")
    _real_oneshot = flagscale.compress.adapter.oneshot

    def _patched_oneshot(**kwargs):
        # Intercept and remove parameters that cause errors
        if "num_calibration_batches" in kwargs:
            print(">> [Patch] Removing unsupported 'num_calibration_batches' argument")
            del kwargs["num_calibration_batches"]
        # Call the original function
        return _real_oneshot(**kwargs)
    
    # Replace the oneshot reference in the adapter module with our patched version
    flagscale.compress.adapter.oneshot = _patched_oneshot
else:
    print(">> [Warning] Could not find 'oneshot' in flagscale.compress.adapter. Patch may not work.")
# --- Monkey Patch End ---

def load_calibration_dataset(cfg, tokenizer):
    if not cfg.data.get("path"):
        return None
    return None

def prepare_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def compress(cfg):
    if "compress" in cfg:
        cfg = cfg.compress

    model_path = cfg.model.model_path
    save_dir = cfg.system.save_dir

    tokenizer = None
    if cfg.data.get("tokenizer_args"):
        tokenizer_args = cfg.data.tokenizer_args
        t_path = tokenizer_args.get("tokenizer_path", model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            t_path,
            use_fast=tokenizer_args.get("use_fast", True),
            trust_remote_code=tokenizer_args.get("trust_remote_code", True)
        )

    model_cls_str = cfg.model.get("model_cls", "AutoModelForCausalLM")
    model_cls = globals().get(model_cls_str)
    if model_cls is None:
        try:
            model_cls = eval(model_cls_str)
        except:
            model_cls = AutoModelForCausalLM

    # Fix float16 issue
    dtype_str = cfg.model.get("torch_dtype", "float16")
    if isinstance(dtype_str, str):
        dtype_str = dtype_str.replace("torch.", "")
        torch_dtype = getattr(torch, dtype_str)
    else:
        torch_dtype = dtype_str

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=cfg.model.get("device_map", "auto")
    )

    dataset = load_calibration_dataset(cfg, tokenizer)

    compress_args = OmegaConf.to_container(cfg.compress_args, resolve=True)

    # Extra safety: also attempt removal before passing to Adapter
    if "num_calibration_batches" in compress_args:
        del compress_args["num_calibration_batches"]

    adapter = LLMCompressorAdapter(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=save_dir,
        num_calibration_steps=cfg.data.get("num_calibration_steps", 128),
        **compress_args
    )

    adapter.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()
    cfg = prepare_config(args.config_path)
    compress(cfg)



