# KERV in FlagScale

This example integrates KERV training and inference into FlagScale. KERV
accelerates vision-language-action inference with speculative action
generation, kinematic feedback, and batched tree verification.

FlagScale provides a native, model-independent KERV control path for batched
candidate generation, verification-tree construction, relaxed acceptance,
dynamic threshold adjustment, and Kalman completion. Model definitions,
weights, datasets, and the optimized KERV operators remain in their public
upstream projects and are selected through reproducible FlagScale
configurations.

## Components

| Stage | Entrypoint | Purpose |
|---|---|---|
| Verifier LoRA | OpenVLA `vla-scripts/finetune.py` | Parameter-efficient OpenVLA adaptation |
| Verifier full training | OpenVLA `vla-scripts/train.py` | Full FSDP OpenVLA training |
| Draft data generation | KERV `training/generate_drafter_data.py` | Frozen-verifier supervision |
| Drafter training | KERV `training/train_drafter.py` | One-layer speculative drafter |
| Inference | KERV `run_kerv_libero.py` | LIBERO rollout with KERV runtime optimization |

## Installation

Clone FlagScale, KERV, and the official OpenVLA training repository. The
integration was validated with KERV commit `4abb3df` and OpenVLA commit
`c8f03f4`:

```bash
git clone https://github.com/flagos-ai/FlagScale.git
git clone https://github.com/zhengzihaoPKU/KERV.git
git clone https://github.com/openvla/openvla.git openvla-training

git -C KERV checkout 4abb3df
git -C openvla-training checkout c8f03f4

cd FlagScale
pip install -r ../KERV/requirements-train.txt
pip install -e ../KERV/openvla
pip install -e . --no-deps
pip install "typer>=0.9" hydra-core omegaconf

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
pip install -e ../LIBERO
```

Set the source locations. These variables replace all local absolute paths:

```bash
export KERV_SOURCE_ROOT=/path/to/KERV
export OPENVLA_SOURCE_ROOT=/path/to/openvla-training
export KERV_BASE_CHECKPOINT=/path/to/openvla-libero-goal
export KERV_DRAFT_CHECKPOINT=/path/to/kerv-drafter
export KERV_DATA_ROOT=/path/to/modified_libero_rlds
export HF_TOKEN=hf_your_token
export LIBERO_CONFIG_PATH=/path/to/.libero
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
```

If the container does not expose NVIDIA EGL, use an OSMesa runtime instead:

```bash
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/path/to/osmesa/lib:${LD_LIBRARY_PATH:-}
```

## Train the verifier

LoRA fine-tuning follows the official OpenVLA Hugging Face recipe:

```bash
flagscale train kerv -c examples/kerv/conf/train_verifier_lora.yaml
```

Full training follows the official OpenVLA FSDP recipe and normally requires
eight GPUs. Set `KERV_PRISMATIC_CHECKPOINT` and a registered OpenVLA
`KERV_VLA_CONFIG` before launching:

```bash
export KERV_TRAIN_GPUS=8
flagscale train kerv -c examples/kerv/conf/train_verifier_full.yaml
```

## Train the drafter

Generate supervision with the frozen verifier:

```bash
flagscale train kerv -c examples/kerv/conf/generate_draft_data.yaml
```

For a short pipeline check, set `KERV_MAX_SAMPLES=8`. Then train the one-layer
drafter with the released DeepSpeed ZeRO-2 configuration:

```bash
export KERV_DRAFT_DATA=$PWD/outputs/kerv_draft_data/samples
flagscale train kerv -c examples/kerv/conf/train_drafter.yaml
```

The drafter script writes DeepSpeed checkpoints under `state_<epoch>`. Select
the desired `pytorch_model.bin` and place a copy of `drafter_config.json` beside
it as `config.json` before using that directory as `KERV_DRAFT_CHECKPOINT`.

## Run inference

Configure LIBERO and run one BF16-safe KERV rollout:

```bash
export LIBERO_CONFIG_PATH=$HOME/.libero
export KERV_MAX_TASKS=1
export KERV_TRIALS=1
flagscale inference kerv -c examples/kerv/conf/inference.yaml
```

The default BF16 profile keeps KERV's acceptance rule, Kalman logic, model
precision, and action definition unchanged. It enables the 14 integrated KERV
operators from the checked-out `runtime_opt` package, including static-tree
packing and attention, Verify-Accept control, KV commit, QKV fusion,
Gate-Up-SwiGLU fusion, and RoPE/KV write.

## Native runtime and CI smoke

The model-independent control path is directly importable from
`flagscale.models.kerv`. It has no checkpoint dependency and is covered by the
multi-platform unit-test suite:

```bash
pytest tests/unit_tests/models/kerv -v
```

The CUDA functional test runs real candidate generation, static-tree
construction, verification, and acceptance on the current GPU, then compares
its output with a checked-in golden result:

```bash
flagscale inference kerv \
  -c tests/functional_tests/inference/kerv/conf/native_runtime_smoke.yaml \
  --test

pytest tests/test_utils/runners/check_results.py::test_inference_equal \
  --path tests/functional_tests --task inference --model kerv \
  --case native_runtime_smoke
```

## Configuration check

Configuration composition does not require weights:

```bash
flagscale train kerv -c examples/kerv/conf/train_verifier_lora.yaml --dryrun
flagscale inference kerv -c examples/kerv/conf/inference.yaml --dryrun
```

`--dryrun` checks FlagScale configuration composition. Running the generated
stage with `--dry-run` additionally validates the source checkout, entrypoint,
and rendered KERV arguments without loading model weights.

## Upstream references

- [KERV](https://github.com/zhengzihaoPKU/KERV)
- [OpenVLA](https://github.com/openvla/openvla)
- [SpecVLA](https://github.com/PineTreeWss/SpecVLA)

KERV and OpenVLA include their own license and third-party notices. This
FlagScale integration is released under the FlagScale Apache-2.0 license.
