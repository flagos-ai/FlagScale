import os
import platform
import subprocess
import sys

from packaging.version import parse

FLAGSCALE_VERSION = "0.8.0"

BACKEND_DEVICE_VERSION = {
    "vllm": {
        "gpu": [
            "0.9.2.dev0+gb6553be1b.d20250708.cu124-cp310-cp310-linux_x86_64",
            "0.9.2.dev0+gb6553be1b.d20250708.cu124-cp312-cp312-linux_x86_64",
        ],
        "metax": ["0.7.2+maca2.29.2.7torch2.4-cp310-cp310-linux_x86_64"],
    }
}


def get_whl_version(backend, device):
    if "metax" in device.lower():
        device = "metax"

    if device == "gpu":
        return _get_whl_version_gpu(backend)
    elif device == "metax":
        return _get_whl_version_metax(backend)
    else:
        raise NotImplementedError(f"Unsupported device type: {device}")


def _get_whl_version_gpu(backend):
    def _get_cuda_version():
        """Get the CUDA version from nvcc.
        Adapted from https://github.com/vllm-project/vllm/blob/main/setup.py
        """
        from torch.utils.cpp_extension import CUDA_HOME

        assert CUDA_HOME is not None, "CUDA_HOME is not set"
        nvcc_output = subprocess.check_output(
            [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
        )
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        nvcc_cuda_version = "cu" + str(nvcc_cuda_version).replace(".", "")
        return nvcc_cuda_version

    cuda_version = _get_cuda_version()
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_str = platform.system().lower()  # 'linux' / 'darwin' / 'windows'
    arch = platform.machine().lower()  # 'x86_64' / 'aarch64' / etc.
    current_platform_tag = f"{platform_str}_{arch}"

    if backend == "vllm":
        suffix = f"{cuda_version}-{python_version}-{python_version}-{current_platform_tag}"
        versions = BACKEND_DEVICE_VERSION[backend]["gpu"]
        compatible_versions = []
        for version in versions:
            if version.endswith(suffix):
                compatible_versions.append(version)
        return versions, compatible_versions
    elif backend == "robotics":
        pass
    else:
        raise NotImplementedError(f"Unsupported backend: {backend}")


def _get_whl_version_metax(backend):
    def get_maca_version():
        default_version = "2.29.2.7"
        maca_path = os.getenv('MACA_PATH')
        if not maca_path:
            return default_version
        if not os.path.exists(maca_path):
            return default_version
        else:
            version_file = os.path.join(maca_path, 'Version.txt')
            if not os.path.exists(version_file) or not os.path.isfile(version_file):
                return default_version
            with open(version_file, 'r') as f:
                version = f.readline().strip().split(":")[-1]
                return version

    maca_version = get_maca_version()
    import torch

    torch_version = torch.__version__
    torch_version = ".".join(torch_version.split(".")[:2])
    metax_version = "maca" + maca_version + "torch" + torch_version
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_str = platform.system().lower()  # 'linux' / 'darwin' / 'windows'
    arch = platform.machine().lower()  # 'x86_64' / 'aarch64' / etc.
    current_platform_tag = f"{platform_str}_{arch}"

    if backend == "vllm":
        suffix = f"{metax_version}-{python_version}-{python_version}-{current_platform_tag}"
        versions = BACKEND_DEVICE_VERSION[backend]["metax"]
        compatible_versions = []
        for version in versions:
            if version.endswith(suffix):
                compatible_versions.append(version)
        return versions, compatible_versions
    else:
        raise NotImplementedError(f"Unsupported backend: {backend}")
