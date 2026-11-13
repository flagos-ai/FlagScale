"""Build the optional MemRift split/merge CUDA extension."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_repo_root = Path(__file__).resolve().parents[4]
os.chdir(_repo_root)


setup(
    name="flagscale-memrift-kernels",
    ext_modules=[
        CUDAExtension(
            "flagscale.compress.memrift._float_split_stride_pin",
            [str(Path(__file__).resolve().with_name("float_split_stride_pin.cu"))],
            extra_cuda_cflags=["-O3", "-lineinfo"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
