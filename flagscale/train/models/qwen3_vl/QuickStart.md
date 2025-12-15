
# 1. Install the FlagScale

## 1.1. Downlowd the source code 

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale
```

## 1.2. Apply the submodule patch code

```bash
python ./tools/patch/unpatch.py --backend=Megatron-LM
python ./tools/patch/unpatch.py --backend=Megatron-Energon
cd ./third_party/Megatron-Energon/
pip install -e .
cp -r src/megatron/energon/ ../Megatron-LM/megatron/
```

You can also refered the readme in `https://github.com/FlagOpen/FlagScale.git`
# 2. Prepare checkpoint

Reference [convert.md](../../../../tools/checkpoint/qwen3_vl/convert.md)
```bash
mkdir -p /mnt/qwen-vl-ckpts
cd /mnt/qwen-vl-ckpts
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
cd Qwen3-VL-8B-Instruct
git lfs pull

cd ./tools/checkpoint/qwen3_vl/
export PYTHONPATH=$PYTHONPATH:../../../:../../../third_party/Megatron-LM/
bash hf2mcore_qwen_vl_convertor.sh 8B \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-tp2 \
2 1 false bf16  \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct
```

# 3. Preprocess dataset

Reference [dataset_preparation.md](../../../../tools/datasets/qwenvl/dataset_preparation.md)

```bash
cd /mnt # custom your path

mkdir llava-datasets
cd llava-datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip

#convert to webdataset format
cd ./tools/datasets/qwenvl/
export PYTHONPATH=$PYTHONPATH:../../../third_party/Megatron-LM/

python convert_custom_dataset_to_wds_chatml_str.py \
    --dataset-root=/mnt/LLaVA-Pretrain \
    --output-root=/mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/ \
    --json=blip_laion_cc_sbu_558k.json \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --vision-root=/mnt/LLaVA-Pretrain \
    --dp-size 1 \
    --num-workers 1
```
The preprocessed dataset will be stored at the output-root path `/mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1`.
The configuration of `data-path` is `/mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1` and the configuration of `vision-path` is `/mnt/LLaVA-Pretrain` in the step 4.

# 4. Add your configuration

Add the data path and checkpoint path in ./examples/qwen3_vl/conf/train/8b.yaml as shown below:

```bash
# dataset
data_path: /mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1
vision_root: /mnt/LLaVA-Pretrain

# ckpt
pretrained_checkpoint: /mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-tp2
tokenizer_path: /mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct
```

Start training.
```bash
python run.py --config-path ./examples/qwen3_vl/conf  --config-name train action=run
```

Stop training.
```bash
python run.py --config-path ./examples/qwen3_vl/conf  --config-name train action=stop
```

# 5. Convert the checkpoint to HuggingFace

Reference [convert.md](../../../../tools/checkpoint/qwen3_vl/convert.md)

``` bash
cd ./tools/checkpoint/qwen3_vl/
bash hf2mcore_qwen_vl_convertor.sh 8B \
./train_qwen3_vl_7b/checkpoints \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-fs2hf-tp2 \
2 1 true bf16  \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct
```
The converved checkpoint is stored in `/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-fs2hf-tp2`

# PS
The path `./` represents the path of `FlagScale` that you download.