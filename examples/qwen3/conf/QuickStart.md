## 1. Heterogeneous Training Environment and Code

### 1.1 Docker Image Paths

- NVIDIA A800: https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/hetero_train/metax/nvidia_metax.tar

- METAX C550: https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/hetero_train/metax/metax_nvidia.tar

You can directly download the images using the `wget` command on Linux.

#### Configure SSH Port for Password-Free Multi-Machine Access

```Plain
#Replace 22 with your custom password-free port
#Execute the following commands inside the Docker container

sed -i 's/^Port .*/Port 22/' /etc/ssh/sshd_config

service ssh restart
```

### 1.2 Install FlagScale

#### 1.2.1 Download the Source Code

```bash
git clone -b main-legacy https://github.com/flagos-ai/FlagScale.git
cd FlagScale/
```

#### 1.2.2 Apply Submodule Patch Code

```bash
# C550
python3 tools/patch/unpatch.py --backend FlagScale Megatron-LM --device-type Metax_C550 --task train --commit 4e1b978fd626e8c23e3f894cc32ae09fe641401e

# A800
git reset --hard 05267318f750f694f61e547fa7a7b95876c72b5e 
python3 tools/patch/unpatch.py --backend Megatron-LM
```

## 2. Start Heterogeneous Training (hetero_train)

### 2.1 Prepare Dataset Demo

We provide a small processed dataset ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) derived from the [Pile](https://pile.eleuther.ai/) dataset.

```bash
mkdir -p /path/to/data && cd /path/to/data
wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx
wget https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin
```

### 2.2 Edit Configuration Files

We use the qwen3-10b model as an example:

#### File Path: examples/qwen3/conf/train_hetero_10b.yaml

```yaml
defaults:
  - _self_
  - train: 10b_hetero

experiment:
  exp_name: Qwen3-10b_muxi
  seed: 42
  save_steps: 1000
  load: null
  exp_dir: ./${experiment.exp_name}
  ckpt_format: torch
  task:
    type: train
    backend: megatron
    entrypoint: flagscale/train/train_gpt.py
  runner:
    backend: torchrun
    per_node_task: false
    no_shared_fs: false
    ssh_port: xxx  Replace with Docker SSH port
    nnodes: 2
    nproc_per_node: 8
    rdzv_backend: static
    hostfile: ./muxi_hostfile  
  cmds:
    before_start: source /root/miniconda3/bin/activate flagscale-train
  envs:
    FLAGCX_ENABLE_TOPO_DETECT: TRUE
    FLAGCX_DEBUG: TRACE
    FLAGCX_IB_HCA: mlx5
    CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
    CUDA_DEVICE_MAX_CONNECTIONS: 1
    NVTE_ALLOW_NONDETERMINISTIC_ALGO: 0
    device_type_specific:
      C550:
        LOGLEVEL: "INFO"
        CUCC_PATH: "/opt/maca/tools/cu-bridge"
        CUDA_PATH: "/opt/maca/tools/cu-bridge"
        DEVINFO_ROOT: "/opt/maca"
        LD_LIBRARY_PATH: "/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/mxdriver/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib"
        MACA_CLANG: "/opt/maca/mxgpu_llvm"
        MACA_CLANG_PATH: "/opt/maca/mxgpu_llvm/bin"
        MACA_PATH: "/opt/maca"
        PATH: "/opt/conda/bin:/opt/conda/condabin:/opt/maca/tools/cu-bridge:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:/opt/conda/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        MCCL_LIMIT_RING_LL_THREADTHRESHOLDS: 1
        SET_DEVICE_NUMA_PREFERRED: 1
        PYTORCH_ENABLE_SAME_RAND_A100: 1
        NVTE_FLASH_ATTN: 1
        NVTE_FUSED_ATTN: 0
        MACA_SMALL_PAGESIZE_ENABLE: 1
        MCCL_MAX_NCHANNELS: 18
        MCCL_P2P_LEVEL: SYS

device_type_specific:
  C550:
    build_dir: /path/to/FlagScale/build/Metax_C550/FlagScale
action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

#### File Path: examples/qwen3/conf/train/10b_hetero.yaml

```yaml
system:
  distributed_backend: flagcx
  no_shared_fs: ${experiment.runner.no_shared_fs}
  ...
  ...
  checkpoint:
    save_interval: ${experiment.save_steps}
    load: ${experiment.load}
    ckpt_format: ${experiment.ckpt_format}

  hetero:
    enable_hetero: True
    hetero_use_cpu_communication: False
    use_partial_reduce_for_shared_embedding: True
     mesh format [tp1,cp1,ep1,dp1,pp1,(tp2,cp2...)]
    hetero_pipeline_layer_split: [28,28]
    hetero_process_meshes: [1,1,1,8,1,1,1,1,8,1]
    hetero_device_types: ["A800","C550"]

    standalone_embedding_stage: False
    hetero_current_device_type: "A800"

    ...
    ...

data:
  data_path: /path/pile_wikipedia_demo
  split: 1
  no_mmap_bin_files: true
  tokenizer:
    legacy_tokenizer: true
    tokenizer_type: QwenTokenizerFS
    tokenizer_path: xxx
    vocab_size: 151851
    padded_vocab_size: 151936
    make_vocab_size_divisible_by: 64
```

#### File Path: ./muxi_hostfile

```
ip slots=8 type=A800
ip slots=8 type=C550
```

### 2.3 Start Training

```bash
python run.py --config-path ./examples/qwen3conf  --config-name train_hetero_10b action=run
```

### 2.4 Stop Training

```bash
python run.py --config-path ./examples/qwen3conf  --config-name train_hetero_10b action=stop
```


