# Quick Start

vLLM implementation of https://github.com/baaivision/Emu3.5

## Environment Setup

### Install FlagScale
- Build from source code base on vLLM tag/0.11.0
```bash
git clone https://github.com/flagos-ai/FlagScale.git
cd FlgScale
python tools/patch/unpatch.py --backend vllm
cd FlagScale/third_party/vllm
pip install -r requirements/cuda.txt --no-cache-dir
MAX_JOBS=32 pip install --no-build-isolation -v .
```

### Prepare Emu3.5
```bash
pip install flash_attn==2.8.3 --no-build-isolation
cd FalgScale
git clone --no-checkout https://github.com/baaivision/Emu3.5.git tmp_repo
cd tmp_repo
git sparse-checkout init --cone
git sparse-checkout set src assets
git checkout 5d6f548ea63d7540460c3524b7a46cfd3cc67942
mv src assets ../
cd ..
rm -rf tmp_repo
```

## Configuration

- Path: `./examples/emu3.5/conf`

- Experiment configuration: `./examples/emu3.5/conf/image_generation.yaml`

```yaml
defaults:
  - _self_
  - inference: t2i # x2i or t2i

experiment:
  exp_name: emu3p5_one_image_generation
  exp_dir: outputs/${experiment.exp_name}
  vq_model: BAAI/Emu3.5-VisionTokenizer
  model: BAAI/Emu3.5-image
  tokenizer: src/tokenizer_emu3_ibq/ # from Emu3.5 repo, please prepare Emu3.5's src&assers
  cmds:
    before_start: source /root/miniconda3/bin/activate flagscale-inference # a conda env with vllm installed
```

- sampling configuration: `./examples/emu3.5/conf/inference/t2i.yaml`
```yaml
generate:
  task_type: t2i
  ratio: "default"
  image_area: 1048576
  sampling:
    max_tokens: 5120
    detokenize: false
    top_k: 131072
    top_p: 1.0
    temperature: 1.0
    text_top_k: 1024
    text_top_p: 0.9
    text_temperature: 1.0
    image_top_k: 5120
    image_top_p: 1.0
    image_temperature: 1.0
    guidance_scale: 5.0
```

## Run Inference

entrypoint: `./flagscale/inference/inference_emu3p5.py`

```bash
python run.py --config-path examples/emu3.5/conf/ --config-name image_generation.yaml
```

```bash
python run.py --config-path examples/emu3.5/conf/ --config-name interleaved_generation.yaml
```
