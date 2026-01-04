# README

This directory contains scripts for converting checkpoints between DCP (Distributed Checkpoint) and SFPT (Single File Per Tensor) formats.

## Scripts

- `dcp_to_sfpt.py` - Converts a DCP checkpoint to SFPT format.
- `sfpt_to_dcp.py` - Converts an SFPT checkpoint to DCP format.

## Usage

**Convert DCP to SFPT:**
1. Get the DCP checkpoint non-homogeneous layers from the training run.
    * Add the environment variable to experiment-level configuration file:
        ```yaml
        envs:
          FS_NON_HOMOGENEOUS_LAYERS: True
        ```

    * Add the following to the task-level configuration file:
        ```yaml
          use_dist_ckpt: True
          ckpt_format: torch_dist
          ckpt_fully_parallel_save: True
          ckpt_fully_parallel_load: True
        ```

2. Set the `PYTHONPATH` environment variable:

    ```bash
    # FlagScale_ROOT is the root directory of the FlagScale repository
    export PYTHONPATH=$FlagScale_ROOT/megatron:$FlagScale_ROOT
    ```

3. Run the conversion script:
    ```bash
    torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 \
      --master_addr localhost --master_port 1234 \
      dcp_to_sfpt.py --input_dir /path/to/dcp_checkpoint --output_dir /path/to/output_sfpt_checkpoint
    ```

**Convert SFPT to DCP:**

1. Set the `PYTHONPATH` environment variable:
    ```bash
    # FlagScale_ROOT is the root directory of the FlagScale repository
    export PYTHONPATH=$FlagScale_ROOT/megatron:$FlagScale_ROOT
    ```

2. Run the conversion script:
    ```bash
    FS_SFPT_CKPT_SAVE=1 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 \
      --master_addr localhost --master_port 1234 \
      sfpt_to_dcp.py --input_dir /path/to/sfpt_checkpoint --output_dir /path/to/output_dcp_checkpoint
    ```

3. Use the DCP checkpoint for further fine-tuning.
    * Add the environment variables to experiment-level configuration file:
        ```yaml
        envs:
          FS_NON_HOMOGENEOUS_LAYERS: True
          FS_SFPT_CKPT_LOAD: True
        ```

    * Add the following to the task-level configuration file:
        ```yaml
          use_dist_ckpt: True
          ckpt_format: torch_dist
          ckpt_fully_parallel_save: True
          ckpt_fully_parallel_load: True
          finetune: True
          load: /path/to/output_dcp_checkpoint
        ```
