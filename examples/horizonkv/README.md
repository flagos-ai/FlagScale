# HorizonKV video generation in FlagScale

FlagScale runs the distilled **Wan2.1-14B autoregressive** model with HorizonKV
through its diffusion inference engine. HorizonKV uses frame retrieval, compacted
RoPE, RASA and OAE to accelerate generation with a bounded GPU KV cache. The
AR+Diffusion pipeline is retained only as an efficiency baseline.

Install dependencies with Python 3.10 and CUDA toolkit 12.4 (`nvcc` on PATH):

```bash
bash examples/horizonkv/setup_env.sh
```

Download the [Krea 14B checkpoint](https://huggingface.co/krea/krea-realtime-video)
to the directory used by the default config (run from the FlagScale root):

```bash
.venvs/horizonkv-worker/bin/python -m huggingface_hub.cli.hf download krea/krea-realtime-video \
  krea-realtime-video-14b.safetensors \
  --local-dir models/horizonkv/checkpoints/krea
```

The VAE, T5 and tokenizer are also required. Download only these shared assets
from Wan2.1; the 1.3B DiT weights are unnecessary:

```bash
.venvs/horizonkv-worker/bin/python -m huggingface_hub.cli.hf download Wan-AI/Wan2.1-T2V-1.3B \
  --include 'Wan2.1_VAE.pth' 'models_t5_umt5-xxl-enc-bf16.pth' 'google/umt5-xxl/*' \
  --local-dir models/horizonkv/wan_models/Wan2.1-T2V-1.3B
```

Run HorizonKV from the FlagScale root:

```bash
.venvs/horizonkv-worker/bin/flagscale run -p examples/horizonkv/conf -n inference -a test
```

To compare HorizonKV with the AR+Diffusion baseline:

```bash
DURATION_SECONDS=30 bash examples/horizonkv/tests/run_efficiency_comparison.sh
```
