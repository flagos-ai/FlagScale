#  Install FlagScale

Clone FlagScale code from github.

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

If you don't have access to the international internet, import FlagScale project on [gitee](https://gitee.com/), then clone from gitee.

```sh
git clone https://gitee.com/flagopen/FlagScale.git
cd FlagScale/
```

Install train and inference env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md) 

# Download Model
Checkpoint is not publish yet.

Directory structure:
```sh
|-- action_model.pt
|-- backbone
|   |-- added_tokens.json
|   |-- chat_template.jinja
|   |-- config.json
|   |-- generation_config.json
|   |-- merges.txt
|   |-- model-00001-of-00002.safetensors
|   |-- model-00002-of-00002.safetensors
|   |-- model.safetensors.index.json
|   |-- preprocessor_config.json
|   |-- special_tokens_map.json
|   |-- tokenizer.json
|   |-- tokenizer_config.json
|   |-- video_preprocessor_config.json
|   `-- vocab.json
`-- config.yaml
```

# Serving

## Edit Config

```sh
cd FlagScale/
vim examples/robobrain_x0_5/conf/serve/libero_qwengroot.yaml
```

Change 2 fields:
- engine_args.model -> checkpoint path.
- engine_args.framework.qwenvl.base_vlm -> backbone checkpoint path.

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/robobrain_x0_5/conf --config-name serve action=run
```

## Test Server with Client

Download test images:

```sh
cd FlagScale/
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_0_latest.jpg
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_1_latest.jpg
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_2_latest.jpg
```

Run client:

```sh
python examples/robobrain_x0_5/client_libero.py  \
--host 127.0.0.1 \
--port 5001 \
--base-img orbbec_0_latest.jpg \
--left-wrist-img orbbec_1_latest.jpg \
--right-wrist-img orbbec_2_latest.jpg \
--num-steps 20
```
