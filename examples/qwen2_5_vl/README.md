# QuickStart

## Installation

### Clone Repository

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

### Setup Conda Environment

Create a new conda environment for robotics training:

```sh
conda create -n flagscale-train python=3.12
conda activate flagscale-train
```

Install FlagScale:

```sh
cd FlagScale/
pip install . --verbose
```

Install Megatron and Energon:

```sh
pip install git+https://github.com/NVIDIA/Megatron-Energon.git@ab40226
mkdir -p /tmp
cd /tmp
git clone https://github.com/flagos-ai/Megatron-LM-FL.git
cd Megatron-LM-FL
pip install --no-build-isolation .[mlm,dev]

# add your path of FlagScale and the Megatron in FlagScale to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/xxx/FlagScale:/xxx/FlagScale/flagscale/train/
```

## Train

### Prepare checkpoint

Reference [convert.md](../../../../tools/checkpoint/qwen2_5_vl/convert.md)
```bash
mkdir -p /mnt/qwen2.5-vl-ckpts
cd /mnt/qwen2.5-vl-ckpts
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
cd Qwen2.5-VL-7B-Instruct
git lfs pull

cd ./tools/checkpoint/qwen2_5_vl/
bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-tp2 \
2 1 false bf16  \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct
```

### Preprocess dataset

#### Demo dataset

FlagScale uses WebDataset format and Megatraon.Energon data loader, you need process your data first.

There is a dataset processed: [demo_0913_n2_vlm](https://gitee.com/hchnr/flag-scale/tree/robotics_dataset/demo_0913_n2_vlm/wds-1).

Download demo_0913_n2_vlm:

```sh
mkdir /tmp/datasets
cd /tmp/datasets
git clone https://gitee.com/hchnr/flag-scale.git
cd flag-scale
git checkout robotics_dataset
```

Move .jpg and .npy files from ./demo_0913_n2_vlm/deps to /:

```sh
mkdir -p /share/
cp -r ./demo_0913_n2_vlm/deps/* /
```

If you need to make your own datasets, generate Data in webdataset format (DP=2) to ./demo_0913_n2_vlm/wds-2:

```sh
python tools/datasets/qwenvl/convert.py \
    --dataset-root=./demo_0913_n2_vlm \
    --output-root=./demo_0913_n2_vlm \
    --json=demo_0913_n2.jsonl \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --vision-root='' \
    --shuffle-tars \
    --num-workers=1 \
    --max-samples-per-tar 100000 \
    --dp-size 2
```

#### Formal dataset

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
export PYTHONPATH=$PYTHONPATH:../../../flagscale/train/

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
    --num-workers 20
```

The preprocessed dataset will be stored at the output-root path `/mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1`.
The configuration of `data-path` is `/mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1` and the configuration of `vision-path` is `/mnt/LLaVA-Pretrain` in the step 4.

### Add your configuration

Add the data path and checkpoint path in ./examples/qwen2_5_vl/conf/train/7b.yaml as shown below:

```bash
# Use demo dataset
data_path: /tmp/datasets/flag-scale/demo_0913_n2_vlm/wds-1
vision_root: /

# Or Use formal dataset
data_path: /mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1
vision_root: /mnt/LLaVA-Pretrain

# ckpt
pretrained_checkpoint: /mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-tp2
tokenizer_path: /mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-tp2
```

Start training.
```bash
python run.py --config-path ./examples/qwen2_5_vl/conf  --config-name train action=run
```

Stop training.
```bash
python run.py --config-path ./examples/qwen2_5_vl/conf  --config-name train action=stop
```

### Convert the checkpoint to HuggingFace

Reference [convert.md](../../../../tools/checkpoint/qwen2_5_vl/convert.md)

``` bash
cd ./tools/checkpoint/qwen2_5_vl/
bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
./train_qwen2_5_vl_7b/checkpoints \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-fs2hf-tp2 \
2 1 true bf16  \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct
```
The converted checkpoint is stored in `/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-fs2hf-tp2`

## Evaluation

Our evaluation process leverages the capabilities of [FlagEval](https://flageval.baai.ac.cn/#/home) platform. Currently, it supports both LLM and VLM, but does not support VLA at this time.

More details about [Auto-Evaluation](https://github.com/flageval-baai/Auto-Evaluation/blob/main/README_en.md) tools.

### Start the server

    ```sh
    python run.py --config-path ./examples/robobrain2/conf --config-name serve action=run
    ```

### Start evaluation

    ```sh
    IP=$(ip addr show | grep -E 'inet ([0-9]{1,3}\.){3}[0-9]{1,3}' | grep -v '127.0.0.1' | grep -v '::1' | awk '{print $2}' | cut -d/ -f1 | head -n1)
    MODEL_NAME=$(curl -s http://localhost:9010/v1/models | jq -r '.data[].id')
    curl http://120.92.17.239:5050/evaluation \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
        "eval_infos": [
            {
                "eval_model": "'$MODEL_NAME'",
                "model": "'$MODEL_NAME'",
                "eval_url": "http://'$IP':9010/v1/chat/completions",
                "tokenizer": "BAAI/RoboBrain2.0-3B",
                "base_model_name": "BAAI/RoboBrain2.0-3B",
                "num_concurrent": 4,
                "batch_size": 8
            }
        ],
        "domain": "MM"
    }'
    ```

### Check Progress

   `request_id` is in response of `Start evaluation`.
    ```sh
    curl http://120.92.17.239:5050/evaluation_progress \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
        "request_id": "4c32ee2b-5d21-41c1-beea-3c4f6f8f2c20",
        "domain": "MM"
    }'
    ```

### Check result

    ```sh
    curl -X GET http://120.92.17.239:5050/evaldiffs \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
        "request_id": "4c32ee2b-5d21-41c1-beea-3c4f6f8f2c20"
    }'
    ```

## PS

The path `./` represents the path of `FlagScale` that you download.
