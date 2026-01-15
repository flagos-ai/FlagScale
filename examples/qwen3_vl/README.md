# QuickStart

## Train

### 1. Install the FlagScale

Download the source code。

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale
```

Apply the submodule patch code.

```bash
# install Megatron-Energon
pip install git+https://github.com/NVIDIA/Megatron-Energon.git@ab40226100830f41de38d1f1204d7848b54b1f3e
# install Megatron-LM-FL
git clone https://github.com/flagos-ai/Megatron-LM-FL
cd Megatron-LM-FL
pip install --no-build-isolation .
# update transformers
pip install transformers==4.57.1
```

You can also refer to the readme in `https://github.com/FlagOpen/FlagScale.git`

### 2. Prepare checkpoint

Reference [convert.md](../../../../tools/checkpoint/qwen3_vl/convert.md)
```bash
mkdir -p /mnt/qwen-vl-ckpts
cd /mnt/qwen-vl-ckpts
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
cd Qwen3-VL-8B-Instruct
git lfs pull

cd ./tools/checkpoint/qwen3_vl/
export PYTHONPATH=../../../:$PYTHONPATH
bash hf2mcore_qwen3_vl_convertor.sh 8B \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-tp2 \
2 1 false bf16  \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct
```

### 3. Preprocess dataset


#### Demo dataset

FlagScale uses WebDataset format and Megatraon.Energon data loader, you need to process your data first.

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

# convert to webdataset format
cd ./tools/datasets/qwenvl/
export PYTHONPATH=../../../:$PYTHONPATH

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

### 4. Add your configuration

Add the data path and checkpoint path in ./examples/qwen3_vl/conf/train/8b.yaml as shown below:

```bash
# use demo dataset
data_path: /tmp/datasets/flag-scale/demo_0913_n2_vlm/wds-1
vision_root: /

# or use formal dataset
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

### 5. Convert the checkpoint to HuggingFace

Reference [convert.md](../../../../tools/checkpoint/qwen3_vl/convert.md)

``` bash
cd ./tools/checkpoint/qwen3_vl/
export PYTHONPATH=../../../:$PYTHONPATH

bash hf2mcore_qwen3_vl_convertor.sh 8B \
../../../train_qwen3_vl_8b/checkpoints \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-fs2hf-tp2 \
2 1 true bf16  \
/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct
```

The converted checkpoint is stored in `/mnt/qwen-vl-ckpts/Qwen3-VL-8B-Instruct-fs2hf-tp2`

## Evaluation

Our evaluation process leverages the capabilities of [FlagEval](https://flageval.baai.ac.cn/#/home) platform. Currently, it supports both LLM and VLM, but does not support VLA at this time.

More details about [Auto-Evaluation](https://github.com/flageval-baai/Auto-Evaluation/blob/main/README_en.md) tools.

### 1. Start the server
    More details refer to this [doc](https://github.com/flagos-ai/FlagScale/tree/main/examples/robobrain2_5).

    ```sh
    python run.py --config-path ./examples/robobrain2_5/conf --config-name serve action=run
    ```

### 2. Start evaluation

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
                "tokenizer": "Qwen/Qwen3-VL-4B-Instruct",
                "base_model_name": "Qwen/Qwen3-VL-4B-Instruct",
                "num_concurrent": 4,
                "batch_size": 8
            }
        ],
        "domain": "MM"
    }'
    ```

### 3. Check Progress

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

### 4. Check result

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
