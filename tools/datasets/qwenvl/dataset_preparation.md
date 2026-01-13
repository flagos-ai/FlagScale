# 📎 Reference
Mainly based on [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/multimodal_data_preprocessing/),with necessary modifications for integration into the current training framework.

# Dataset Download & Preprocessing

```bash
cd /mnt

mkdir llava-datasets
cd llava-datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip

#convert to webdataset format:
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
    --max-samples-per-tar 100000000 \
    --dp-size 1 \
    --num-workers 20
```
The preprocessed data will stored at the output-root path `/mnt/LLaVA-Pretrain/blip_laion_cc_sbu_558k/wds-1`.

## Prepare Multimodal Datasets Based on ShareGPT Format

The current Qwen2-VL/Qwen2.5-VL supports training with complex multimodal samples in a specific ShareGPT-like format. Follow the instructions below to convert your datasets into the supported format.

## ShareGPT Data Format
You may need to manually convert your dataset into the ShareGPT format, structured as follows:
```json
[
  {
    "conversations": [
        {
            "from": "human",
            "value": "<image>human instruction<image>"
        },
        {
            "from": "gpt",
            "value": "model response"
        },
        {
            "from": "human",
            "value": "<video><video>human instruction"
        },
        {
            "from": "gpt",
            "value": "model response"
        }
    ],
    "images": [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
    ],
    "videos": [
        "path/to/video1.mp4",
        "path/to/video2.mp4"
    ]
  },
  {
    // another sample ...
  }
]
```
Here,the images and videos lists should contain the raw file paths corresponding to `<image>` and `<video> `tokens in the conversation in order.

### Video Frame Extraction
Before training, you must convert video files into frame images using tools such as DataJuicer.

For example, given path/to/video1.mp4 located at dataset_root/path/to/video1.mp4, the exported frames should be stored under dataset_root/path/to/video1/. Frame filenames should be in sequential order to ensure temporal alignment.

Recommended filename format:
```
00001.jpg # frame 1
00002.jpg # frame 2
...
```

To enable temporal alignment and support dynamic resolution sampling in Qwen2.5-VL, you must save the exported frame rate for each video in a JSON file. For instance, for the video frames saved in `dataset_root/path/to/video1/`, create a `dataset_root/path/to/video1.json` with the following structure:
```
{
    "fps": "2.0" // Exported frame rate
}
```

### Video Frame Extraction(Demo)
We provide a lightweight script to convert LLaVA-format video datasets (e.g., LLaVA-Video-178K) into the ShareGPT format for small-scale testing. For large-scale datasets, we recommend using a dedicated tool for frame extraction.
```
cd /mnt/llava-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/LLaVA-Video-178K-demo.tar
tar -xvf LLaVA-Video-178K-demo.tar

cd /workspace/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_llava_frame_dataset.py \
    --dataset-root /mnt/llava-datasets/LLaVA-Video-178K \
    --time-interval 0.5 # Save one frame every 0.5 seconds (FPS ≈ 2.0)

```

You can then run `convert_custom_dataset_to_wds_chatml.py` to convert `/mnt/llava-datasets/LLaVA-Video-178K` into the training format.

### Converting LLaVA-style Image Datasets

For LLaVA-style image datasets in .jsonl format, simply run the following script to convert them for use with `convert_custom_dataset_to_wds_chatml.py`:

```
# replace `image` key with `images`
python replace_llava_image_key.py \
    --input-file your_raw_dataset.json_or_jsonl \
    --output-file dataset.json

```

Convert to ChatML Format
Assuming your **ShareGPT-formatted** dataset looks like this:
```
dataset_root/
-   dataset.json
-   ...
```

Run the following to convert the dataset into WebDataset format for training. Output will be stored in `dataset_root/wds`.
```
python tools/datasets/qwenvl/convert_custom_dataset_to_wds_chatml.py \
--dataset-root dataset_root \
--json dataset.json \
--train-split 9 \
--val-split 1 \
--test-split 0
```
