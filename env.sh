#!/bin/bash
source /opt/dtk/env.sh

DTK_CUDA_PATH=/opt/dtk-25.04.2/cuda/cuda-12/targets/x86_64-linux/lib
TORCH_LIB_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
export LD_LIBRARY_PATH=$DTK_CUDA_PATH:$TORCH_LIB_PATH:$LD_LIBRARY_PATH
export GEMS_VENDOR="hygon"
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=ALL
export GLOO_SOCKET_IFNAME=bond0
export FLAGCX_SOCKET_IFNAME=bond0
export FLAGCX_USENET=mlx5_101,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108
export FLAGCX_USEDEV=1
#export FLAGCX_IB_HCA=mlx5_101,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108
#export FLAGCX_ALGO=Ring
export FLAGCX_MAX_NCHANNELS=16
export FLAGCX_MIN_NCHANNELS=16
export FLAGCX_NET_GDR_LEVEL=7
export FLAGCX_NET_GDR_READ=1
export FLAGCX_SDMA_COPY_ENABLE=0
