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

```sh
git lfs install

mkdir -p /models/BAAI/
cd /models/BAAI/
git clone https://huggingface.co/BAAI/RoboBrain-X0-Preview
```

If you don't have access to the international internet, download from modelscope.

```sh
mkdir -p /models/
cd /models/
modelscope download --model BAAI/RoboBrain-X0-Preview --local_dir BAAI/RoboBrain-X0-Preview
```

# Download Tokenzier

```sh
mkdir -p /models/physical-intelligence/
cd /models/physical-intelligence/
git lfs install
git clone https://huggingface.co/physical-intelligence/fast
```

# Serving

## Edit Config

```sh
cd FlagScale/
vim examples/robobrain_x0/conf/serve/robobrain_x0.yaml
```

Change 3 fields:
- engine_args.model_sub_task -> /models/BAAI/RoboBrain-X0-Preview
- engine_args.port -> A port available in your env, for example: 5001
- engine_args.tokenizer_path ->/models/physical-intelligence/fast

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/robobrain_x0/conf --config-name serve action=run
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
python examples/robobrain_x0/client_agilex.py  \
--host 127.0.0.1 \
--port 5001 \
--base-img orbbec_0_latest.jpg \
--left-wrist-img orbbec_1_latest.jpg \
--right-wrist-img orbbec_2_latest.jpg \
--num-steps 20
```
