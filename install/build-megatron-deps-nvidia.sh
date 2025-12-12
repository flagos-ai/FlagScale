#!/bin/bash

set -e

# Extract CUDA version and format as cu128, cu124, etc.
CUDA_VERSION=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | tr -d ',')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d '.' -f 1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d '.' -f 2)
CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}" 
# Install torch based on CUDA version
if [ "${CUDA_MAJOR}.${CUDA_MINOR}" = "12.4" ]; then
    # CUDA 12.4: install torch 2.5.1+cu124
    echo "Installing torch 2.5.1 for CUDA 12.4"
    pip install --no-cache-dir --verbose torch==2.5.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
elif [ "${CUDA_MAJOR}.${CUDA_MINOR}" = "12.8" ]; then
    # CUDA 12.8: install torch 2.7.1+cu128
    echo "Installing torch 2.7.1 for CUDA 12.8"
    pip install --no-cache-dir --verbose torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128
else
    # Other CUDA versions: install torch without version specification
    echo "Installing torch (latest) for CUDA ${CUDA_VERSION}"
    pip install --no-cache-dir --verbose torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}
fi

pip install --no-cache-dir --no-build-isolation --verbose git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1
pip install --no-cache-dir --no-build-isolation --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.2

# flash-attention install for megatron-lm
cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
flash_attn_version="2.8.0.post2"
pip install --no-cache-dir --verbose https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl

# pip install --verbose flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# rm flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl

# transformer engine install for megatron-lm
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout e9a5fa4e  # Date:   Thu Sep 4 22:39:53 2025 +0200
uv pip install --no-build-isolation --verbose . 
cd ..
rm -r ./TransformerEngine    

# apex install for megatron-lm
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./
cd ..
rm -r ./apex

