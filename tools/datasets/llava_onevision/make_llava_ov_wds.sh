#!/bin/bash
# set environment variables of the gpu cluster
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export NCCL_IB_HCA=mlx5_0,mlx5_1

# set LLaVA-Next code directory
LLaVA_NeXT_HOME="Path_Of_LLaVA-NeXT"
# set the vision model directory
VISION_MODEL_PATH="Path_Of_VISION_MODEL"
PROMPT_VERSION="qwen_2"

# # stage1
# image_aspect_ratio=square

# other stages
image_aspect_ratio=anyres_max_9

set -u
  DATA_PATH=$1
  EXPNAME_PATH=$2
  HOSTFILE=$3
set +u

echo "BASE_RUN_NAME: ${EXPNAME_PATH}"

# set the llm checkpoint directory
CKPT_PATH="Path_Of_LLM"

mkdir -p $CKPT_PATH
mkdir -p $EXPNAME_PATH
LOGFILE=$EXPNAME_PATH/exp.log
NNodes=`wc -l ${HOSTFILE} | cut -d " " -f1`
MASTER_ADDR=`head -n 1 ${HOSTFILE} | cut -d " " -f1`
echo "Master node: ${MASTER_ADDR}"
echo ${NNodes}
echo ${MASTER_ADDR}

# twice indicates how many tasks are to be launched to process the data.
twice=2
for ((j = 0; j < twice; j++))
do
  i=1
  rank=0
  for ip in `cat ${HOSTFILE} | cut -d " " -f1`
  do
      echo "Starting node ${i}/${NNodes}: ${ip}"
      ssh $ip \
      "cd ${PWD} && \
      sysctl fs.inotify.max_user_watches=524288 && \
      export WANDB_MODE=offline && \
      export ACCELERATE_CPU_AFFINITY=1 && \
      export PYTHONPATH=$LLaVA_NeXT_HOME:$PYTHONPATH && \
      export TORCH_NCCL_ENABLE_MONITORING=0 && \
      export FLAGSCALE_LAUNCH_TIMES=${twice} && \
      export FLAGSCALE_LAUNCH_INDEX=${j} && \
      /root/miniconda3/envs/flagscale/bin/torchrun --nproc_per_node=8 --nnodes=${NNodes} --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=$((13888 + j)) llava_ov_wds.py \
          --model_name_or_path ${CKPT_PATH} \
          --version ${PROMPT_VERSION} \
          --data_path $DATA_PATH \
          --image_folder <image_folder> \
          --video_folder <video_folder> \
          --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
          --mm_vision_tower_lr=2e-6 \
          --vision_tower ${VISION_MODEL_PATH} \
          --mm_projector_type mlp2x_gelu \
          --mm_vision_select_layer -2 \
          --mm_use_im_start_end False \
          --mm_use_im_patch_token False \
          --mm_spatial_pool_mode "bilinear" \
          --group_by_modality_length True \
          --image_aspect_ratio ${image_aspect_ratio} \
          --image_grid_pinpoints '(1x1),...,(6x6)' \
          --mm_patch_merge_type spatial_unpad \
          --bf16 True \
          --run_name $EXPNAME_PATH \
          --output_dir "${EXPNAME_PATH}" \
          --num_train_epochs 1 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 1000 \
          --save_total_limit 20 \
          --learning_rate 1e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 1 \
          --model_max_length 32768 \
          --gradient_checkpointing True \
          --dataloader_num_workers 4 \
          --lazy_preprocess True \
          --torch_compile True \
          --torch_compile_backend "inductor" \
          --dataloader_drop_last True \
          --seed 42 \
          --do_train False \
          --frames_upbound 32 1>$LOGFILE.$ip.launch$j 2>&1" &
      i=`expr $i + 1`
      rank=`expr $rank + 1`
  done
done
