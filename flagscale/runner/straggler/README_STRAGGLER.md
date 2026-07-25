<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Straggler on Metax C550

## Scope

This note documents the current `straggler` integration for the Metax C550 training path on `main-legacy`.

Important:

- The active branch in use is `main-legacy`.
- The current runnable path is still the legacy training stack.
- The new launcher path `flagscale/runner/launcher/launcher_ssh.py` is not used here.

## Current Code Path

- FlagScale detector logic:
  - `flagscale/runner/straggler/`
- Train-side integration:
  - `flagscale/train/train.py`

This is a FlagScale-side detector layered on top of the existing Megatron training loop. It does not replace Megatron's native `log_straggler`.

## Metax-Specific Notes

- The implementation avoids new CUDA-only assumptions.
- GPU event profiling is not required by default.
- The practical validation path on Metax should use the mini Aquila config that already passed smoke training.

## Known Pitfalls

- Do not validate this first on the original full 7B config. The practical path we already stabilized on Metax is the mini Aquila config.
- Do not reuse old checkpoints while testing. Use a fresh `exp_dir` and force a missing `checkpoint.load`.
- If the straggler keys are not present in YAML, pass them with `++`.

## Smoke Test

Run from the generated Metax build tree:

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=test \
  experiment.exp_dir=/workspace/exp/aquila_straggler_smoke_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_straggler_smoke_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=16 \
  ++train.system.enable_straggler_detection=true \
  ++train.system.straggler_report_interval=2 \
  ++train.system.straggler_threshold=1.5 \
  ++train.system.straggler_warmup_steps=0
```

## Expected Result

- Training starts from random initialization.
- `iteration 1/2` and `iteration 2/2` both complete.
- A straggler report is printed near the end of the short run.
- Report files are written under:

```bash
/workspace/exp/aquila_straggler_smoke_${TS}/logs/straggler
```

## Full Run Example

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  experiment.exp_dir=/workspace/exp/aquila_straggler_run_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_straggler_run_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=1600 \
  ++train.system.enable_straggler_detection=true \
  ++train.system.straggler_report_interval=20 \
  ++train.system.straggler_threshold=1.5 \
  ++train.system.straggler_warmup_steps=10
```
