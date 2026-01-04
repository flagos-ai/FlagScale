### 📎 Reference

Mainly based on official [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen3_vl),with necessary modifications for integration into the current training framework.

### Download Model
```bash
mkdir -p /mnt/qwen3-vl-ckpts
cd /mnt/qwen3-vl-ckpts
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
cd Qwen3-VL-8B-Instruct
git lfs pull
```

### Megatron-Core Checkpoint Format Conversion
Run the `hf2mcore_qwen3_vl_convertor.sh` script with the following arguments:
```bash
MODEL_SIZE=$1                 # Model size: 4B / 8B / 32B
SOURCE_CKPT_PATH=$2           # Path to the original Huggingface-style checkpoint
TARGET_CKPT_PATH=$3           # Path to save the converted checkpoint
TP=$4                         # Tensor parallel size
PP=$5                         # Pipeline parallel size
mg2hf=$6                      # Whether to convert Megatron-Core checkpoint back to Huggingface format
PR=$7                         # Precision: fp16 / bf16 / fp32
HF_CKPT_PATH=$8               # Path to HF checkpoint (required when mg2hf=true)
```
Example: convert the HF checkpoint to MCore-Dense format and verify the output:

```bash
cd /workspace/FlagScale/tools/checkpointing/qwen3_vl
export PYTHONPATH=../../../:../../../third_party/Megatron-LM:$PYTHONPATH
bash hf2mcore_qwen3_vl_convertor.sh \
8B \
/mnt/qwen3-vl-ckpts/Qwen3-VL-8B-Instruct \
/mnt/qwen3-vl-ckpts/Qwen3-VL-8B-Instruct-tp2pp2 \
2  \
2  \
false \
bf16
```

To convert the trained MCore checkpoint back to Huggingface format for inference:

```bash
cd /workspace/FlagScale/tools/checkpointing/qwen3_vl
export PYTHONPATH=$PYTHONPATH:../../../:../../../third_party/Megatron-LM
bash hf2mcore_qwen3_vl_convertor.sh \
8B \
/mnt/qwen3-vl-ckpts/Qwen3-VL-8B-Instruct-tp2pp2 \
/mnt/qwen3-vl-ckpts/Qwen3-VL-8B-Instruct-tp2pp2-back \
2  \
2  \
true \
bf16 \
/mnt/qwen3-vl-ckpts/Qwen3-VL-8B-Instruct
```

### Optional: Set Custom Layer Distribution for Best Throughput
When continuing pretraining and using an asymmetric pipeline parallel layout for optimal throughput, you need to manually set the number of transformer layers in the first pipeline stage by exporting the following environment variable (same as in training):
```bash
export MP_PP0_LAYERS=16
```
