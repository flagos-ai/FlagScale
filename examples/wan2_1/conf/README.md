# Wan2.1 Inference with AdaSpa
## Overview
This example shows how to run Wan2.1 inference with AdaSpa acceleration in FlagScale.
## Prerequisites
- Linux + NVIDIA GPU (SM80+ recommended)
- CUDA toolkit available (`nvcc -V`)
- Python environment compatible with FlagScale inference
- Install FlagScale and inference dependencies:
  ```bash
  pip install . --verbose
  pip install -r requirements/cuda/inference.txt
  ```

- AdaSpa extra dependency
AdaSpa relies on block sparse attention extension under: `flagscale/inference/core/diffusion/adaspa/third_party/block_sparse_attention`

  ```bash
  cd flagscale/inference/core/diffusion/adaspa/third_party/block_sparse_attention
  pip install -e .
  cd -
  ```
- Prepare model
  Set your local Wan2.1 model path in:

  `examples/wan2_1/conf/inference/1.3b.yaml`
  `examples/wan2_1/conf/inference/1.3b_adaspa.yaml`
- Config selection
  In `examples/wan2_1/conf/inference.yaml`:

  ```bash
  defaults:
    - _self_
    - inference: 1.3b_adaspa (adaspa)
              : 1.3b (taylorseer)  

  ```
- Run inference
  ```bash
  flagscale inference wan2_1 -c examples/wan2_1/conf/inference.yaml
  ```

