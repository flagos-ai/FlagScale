"""
Setup script for float_split_stride_pin CUDA extension.

This extension provides GPU-accelerated float split/merge operations
for MemRift compression.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="float_split_stride_pin",
    ext_modules=[
        CUDAExtension(
            "float_split_stride_pin._ext",
            ["../memrift/csrc/float_split_stride_pin.cu"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                # 多卡/多平台可改用 $TORCH_CUDA_ARCH_LIST
            ],
        )
    ],
    packages=["float_split_stride_pin"],
    cmdclass={"build_ext": BuildExtension},
)
