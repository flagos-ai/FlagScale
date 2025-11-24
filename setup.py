import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

try:
    import git  # from GitPython
except:
    try:
        print("[INFO] GitPython not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
        import git
    except:
        print(
            "[ERROR] Failed to install flagscale. Please use 'pip install . --no-build-isolation' to reinstall when the pip version > 23.1."
        )
        sys.exit(1)

SUPPORTED_DEVICES = ["cpu", "gpu", "ascend", "cambricon", "bi", "metax", "kunlunxin", "musa"]
VLLM_UNPATCH_DEVICES = ["ascend", "cambricon", "bi", "metax", "kunlunxin"]


def _check_backend(backend):
    if backend not in ["llama.cpp", "Megatron-LM", "sglang", "vllm", "Megatron-Energon"]:
        raise ValueError(f"Invalid backend {backend}.")


def check_backends(backends):
    for backend in backends:
        _check_backend(backend)


def check_vllm_unpatch_device(device):
    is_supported = False
    for supported_device in VLLM_UNPATCH_DEVICES:
        if supported_device in device.lower():
            is_supported = True
            return is_supported
    return is_supported


def check_device(device):
    is_supported = False
    for supported_device in SUPPORTED_DEVICES:
        if supported_device in device.lower():
            is_supported = True
            return
    if not is_supported:
        raise ValueError(f"Unsupported device {device}. Supported devices are {SUPPORTED_DEVICES}.")


# Call for the extensions
def _build_vllm(device):
    assert device != "cpu"
    vllm_path = os.path.join(os.path.dirname(__file__), "third_party", "vllm")
    if device != "gpu":
        vllm_path = os.path.join(
            os.path.dirname(__file__), "build", device, "FlagScale", "third_party", "vllm"
        )
    # Set env
    env = os.environ.copy()
    if "metax" in device.lower():
        if "MACA_PATH" not in env:
            env["MACA_PATH"] = "/opt/maca"
        if "CUDA_PATH" not in env:
            env["CUDA_PATH"] = "/usr/local/cuda"
        env["CUCC_PATH"] = f'{env["MACA_PATH"]}/tools/cu-bridge'
        env["PATH"] = (
            f'{env["CUDA_PATH"]}/bin:'
            f'{env["MACA_PATH"]}/mxgpu_llvm/bin:'
            f'{env["MACA_PATH"]}/bin:'
            f'{env["CUCC_PATH"]}/tools:'
            f'{env["CUCC_PATH"]}/bin:'
            f'{env.get("PATH", "")}'
        )
        env["LD_LIBRARY_PATH"] = (
            f'{env["MACA_PATH"]}/lib:'
            f'{env["MACA_PATH"]}/ompi/lib:'
            f'{env["MACA_PATH"]}/mxgpu_llvm/lib:'
            f'{env.get("LD_LIBRARY_PATH", "")}'
        )
        env["VLLM_INSTALL_PUNICA_KERNELS"] = "1"
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '.', '--no-build-isolation', '--verbose'],
        cwd=vllm_path,
        env=env,
    )


def _build_sglang(device):
    assert device != "cpu"
    sglang_path = os.path.join(os.path.dirname(__file__), "third_party", "sglang")
    if device != "gpu":
        sglang_path = os.path.join(
            os.path.dirname(__file__), "build", device, "FlagScale", "third_party", "sglang"
        )
    subprocess.check_call(
        [
            sys.executable,
            '-m',
            'pip',
            'install',
            '-e',
            'python[all]',
            '--no-build-isolation',
            '--verbose',
        ],
        cwd=sglang_path,
    )


def _build_llama_cpp(device):
    llama_cpp_path = os.path.join(os.path.dirname(__file__), "third_party", "llama.cpp")
    print(f"[build_ext] Build llama.cpp for {device}")
    if device == "gpu":
        subprocess.check_call(["cmake", "-B", "build", "-DGGML_CUDA=ON"], cwd=llama_cpp_path)
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", "-j64"], cwd=llama_cpp_path
        )
    elif device == "musa":
        subprocess.check_call(["cmake", "-B", "build", "-DGGML_MUSA=ON"], cwd=llama_cpp_path)
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", "-j8"], cwd=llama_cpp_path
        )
    elif device == "cpu":
        subprocess.check_call(["cmake", "-B", "build"], cwd=llama_cpp_path)
        subprocess.check_call(
            ["cmake", "--build", "build", "--config", "Release", "-j8"], cwd=llama_cpp_path
        )
    else:
        raise ValueError(f"Unsupported device {device} for llama.cpp backend.")


def _build_megatron_energon(device):
    try:
        import editables
        import hatch_vcs
        import hatchling
    except:
        try:
            print("[INFO] hatchling not found. Installing...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "hatchling", "--no-build-isolation"]
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "hatch-vcs", "--no-build-isolation"]
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "editables", "--no-build-isolation"]
            )
            import editables
            import hatch_vcs
            import hatchling
        except:
            print("[ERROR] Failed to install hatchling, hatch-vcs and editables.")
            sys.exit(1)
    energon_path = os.path.join(os.path.dirname(__file__), "third_party", "Megatron-Energon")
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-build-isolation', '--verbose'],
        cwd=energon_path,
    )


class FlagScaleBuild(_build):
    """
    Build the FlagScale backends.
    """

    user_options = _build.user_options + [
        ('backend=', None, 'Build backends'),
        ('device=', None, 'Device type for build'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.backend = None
        self.device = None

    def finalize_options(self):
        super().finalize_options()
        if self.backend is None:
            self.backend = os.environ.get("FLAGSCALE_BACKEND")
        if self.device is None:
            self.device = os.environ.get("FLAGSCALE_DEVICE", "gpu")
        if self.backend is not None:
            # Set the environment variables for backends and device to use in the install command
            # os.environ["FLAGSCALE_BACKEND"] = self.backend
            # os.environ["FLAGSCALE_DEVICE"] = self.device
            check_device(self.device)

            from tools.patch.patch import normalize_backend

            backends = self.backend.split(",")
            normalized = []
            for backend in backends:
                item = normalize_backend(backend.strip())
                if isinstance(item, list):
                    normalized.extend(item)
                else:
                    normalized.append(item)
            self.backend = normalized
            print(f"[build] Received backend = {self.backend}")
            print(f"[build] Received device = {self.device}")
        else:
            print(f"[build] No backend specified, just build FlagScale python codes.")

    def run(self):
        if self.backend is not None:
            build_py_cmd = self.get_finalized_command('build_py')
            build_py_cmd.backend = self.backend
            build_py_cmd.device = self.device
            build_py_cmd.ensure_finalized()

            build_ext_cmd = self.get_finalized_command('build_ext')
            build_ext_cmd.backend = self.backend
            build_ext_cmd.device = self.device
            build_ext_cmd.ensure_finalized()

            self.run_command('build_py')
            self.run_command('build_ext')
        super().run()


class FlagScaleBuildPy(_build_py):
    """
    Unpatch the FlagScale backends.
    """

    user_options = _build_py.user_options + [
        ('backend=', None, 'Build backends'),
        ('device=', None, 'Device type for build'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.backend = None
        self.device = None

    def unpatch_backend(self):
        from tools.patch.unpatch import unpatch

        main_path = os.path.dirname(__file__)
        for backend in self.backend:
            if backend == "FlagScale":
                continue
            backend_commit = None
            if backend == "Megatron-LM":
                backend_commit = os.getenv(f"FLAGSCALE_MEGATRON_COMMIT", None)
            elif backend == "Megatron-Energon":
                backend_commit = os.getenv(f"FLAGSCALE_ENERGON_COMMIT", None)
            elif backend == "sglang":
                backend_commit = os.getenv(f"FLAGSCALE_SGLANG_COMMIT", None)
            elif backend == "vllm":
                backend_commit = os.getenv(f"FLAGSCALE_VLLM_COMMIT", None)
            elif backend == "llama.cpp":
                backend_commit = os.getenv(f"FLAGSCALE_LLAMA_CPP_COMMIT", None)
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            print(f"[build_py] Device {self.device} initializing the {backend} backend.")
            force = os.getenv("FLAGSCALE_FORCE_INIT", False)
            unpatch(
                main_path,
                src,
                dst,
                backend,
                force=force,
                backend_commit=backend_commit,
                fs_extension=True,
            )
            # ===== Copy for packaging =====
            if backend == "Megatron-LM":
                rel_src = os.path.join("third_party", backend, "megatron")
                abs_src = os.path.join(main_path, rel_src)
                abs_dst = os.path.join(self.build_lib, "flag_scale", rel_src)
                print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                if os.path.exists(abs_dst):
                    shutil.rmtree(abs_dst)
                shutil.copytree(abs_src, abs_dst)

        # ===== Copy for packaging for Megatron-Energon =====
        if "Megatron-Energon" in self.backend:
            assert "Megatron-LM" in self.backend, "Megatron-Energon requires Megatron-LM"
            abs_src = os.path.join(
                main_path, "third_party", "Megatron-Energon", "src", "megatron", "energon"
            )
            abs_dst = os.path.join(
                self.build_lib, "flag_scale", "third_party", "Megatron-LM", "megatron", "energon"
            )
            print(f"[build_py] Copying {abs_src} -> {abs_dst}")
            if os.path.exists(abs_dst):
                shutil.rmtree(abs_dst)
            shutil.copytree(abs_src, abs_dst)

            # Source code for Megatron-Energon is copied to the megatron directory
            abs_dst = os.path.join(main_path, "third_party", "Megatron-LM", "megatron", "energon")
            print(f"[build_py] Copying {abs_src} -> {abs_dst}")
            if os.path.exists(abs_dst):
                shutil.rmtree(abs_dst)
            shutil.copytree(abs_src, abs_dst)

    def run(self):
        super().run()
        if self.backend:
            print(f"[build_py] Running with backend = {self.backend}")
            assert self.device is not None
            from tools.patch.unpatch import apply_hardware_patch

            # At present, only vLLM supports domestic chips, and the remaining backends have not been supported yet.
            # FlagScale just modified the vLLM and Megatron-LM
            main_path = os.path.dirname(__file__)
            if "vllm" in self.backend or "Megatron-LM" in self.backend:
                if check_vllm_unpatch_device(self.device):
                    print(f"[build_py] Device {self.device} unpatching the vllm backend.")
                    # Unpatch the backed in specified device
                    from git import Repo

                    main_repo = Repo(main_path)
                    commit = os.getenv("FLAGSCALE_UNPATCH_COMMIT", None)
                    if commit is None:
                        commit = main_repo.head.commit.hexsha
                    # Checkout to the commit and apply the patch to build FlagScale
                    key_path = os.getenv("FLAGSCALE_KEY_PATH", None)
                    apply_hardware_patch(
                        self.device, self.backend, commit, main_path, True, key_path=key_path
                    )
                    build_lib_flagscale = os.path.join(self.build_lib, "flag_scale")
                    src_flagscale = os.path.join(main_path, "build", self.device, "FlagScale")

                    for f in os.listdir(build_lib_flagscale):
                        if f.endswith(".py"):
                            file_path = os.path.join(build_lib_flagscale, f)
                            print(f"[build_py] Removing file {file_path}")
                            os.remove(file_path)

                    for f in os.listdir(src_flagscale):
                        if f.endswith(".py"):
                            src_file = os.path.join(src_flagscale, f)
                            dst_file = os.path.join(build_lib_flagscale, f)
                            print(f"[build_py] Copying file {src_file} -> {dst_file}")
                            shutil.copy2(src_file, dst_file)

                    dirs_to_copy = ["flagscale", "examples", "tools", "tests"]
                    for d in dirs_to_copy:
                        src_dir = os.path.join(src_flagscale, d)
                        dst_dir = os.path.join(build_lib_flagscale, d)
                        if os.path.exists(dst_dir):
                            print(f"[build_py] Removing directory {dst_dir}")
                            shutil.rmtree(dst_dir)
                        if os.path.exists(src_dir):
                            print(f"[build_py] Copying directory {src_dir} -> {dst_dir}")
                            shutil.copytree(src_dir, dst_dir)

                    # ===== Copy for packaging =====
                    if "Megatron-LM" in self.backend:
                        rel_src = os.path.join(
                            main_path,
                            "build",
                            self.device,
                            "FlagScale",
                            "third_party",
                            "Megatron-LM",
                            "megatron",
                        )
                        abs_src = os.path.join(main_path, rel_src)
                        abs_dst = os.path.join(
                            self.build_lib, "flag_scale", "third_party", "Megatron-LM", "megatron"
                        )
                        print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                        if os.path.exists(abs_dst):
                            shutil.rmtree(abs_dst)
                        shutil.copytree(abs_src, abs_dst)

                    if "Megatron-Energon" in self.backend:
                        assert (
                            "Megatron-LM" in self.backend
                        ), "Megatron-Energon requires Megatron-LM"
                        abs_src = os.path.join(
                            main_path,
                            "build",
                            self.device,
                            "FlagScale",
                            "third_party",
                            "Megatron-Energon",
                            "src",
                            "megatron",
                            "energon",
                        )
                        abs_dst = os.path.join(
                            self.build_lib,
                            "flag_scale",
                            "third_party",
                            "Megatron-LM",
                            "megatron",
                            "energon",
                        )
                        print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                        if os.path.exists(abs_dst):
                            shutil.rmtree(abs_dst)
                        shutil.copytree(abs_src, abs_dst)

                        abs_dst = os.path.join(
                            main_path,
                            "build",
                            self.device,
                            "FlagScale",
                            "third_party",
                            "Megatron-LM",
                            "megatron",
                            "energon",
                        )
                        print(f"[build_py] Copying {abs_src} -> {abs_dst}")
                        if os.path.exists(abs_dst):
                            shutil.rmtree(abs_dst)
                        shutil.copytree(abs_src, abs_dst)
                else:
                    self.unpatch_backend()
            else:
                self.unpatch_backend()


class FlagScaleBuildExt(_build_ext):
    """
    Build or pip install the FlagScale backends.
    """

    user_options = _build_py.user_options + [
        ('backend=', None, 'Build backends'),
        ('device=', None, 'Device type for build'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.backend = None
        self.device = None

    def finalize_options(self):
        super().finalize_options()
        if self.backend:
            print(f"[build_ext] Backend received: {self.backend}")

    def run(self):
        if self.backend:
            print(f"[build_ext] Building extensions for backend = {self.backend}")
            for backend in self.backend:
                if backend == "FlagScale":
                    continue
                elif backend == "vllm":
                    _build_vllm(self.device)
                elif backend == "sglang":
                    _build_sglang(self.device)
                elif backend == "llama.cpp":
                    _build_llama_cpp(self.device)
                elif backend == "Megatron-LM":
                    print(
                        f"[build_ext] Megatron-LM does not need to be built, just copy the source code."
                    )
                elif backend == "Megatron-Energon":
                    _build_megatron_energon(self.device)
                    print(
                        f"[build_ext] Megatron-Energon will be copied to megatron after installed."
                    )
                else:
                    raise ValueError(f"Unknown backend: {backend}")
        super().run()

def _read_requirements_file(requirements_path):
    """读取 requirements 文件并返回依赖列表"""
    requirements_file = os.path.join(os.path.dirname(__file__), requirements_path)
    if not os.path.exists(requirements_file):
        return []
    
    requirements = []
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            # 跳过 -r 开头的行（递归引用）
            if line.startswith('-r') or line.startswith('--'):
                continue
            requirements.append(line)
    return requirements


def _get_install_requires():
    """获取 install_requires 列表"""
    install_requires = []
    
    # 读取 requirements-base.txt
    install_requires.extend(_read_requirements_file('requirements/requirements-base.txt'))
    
    # 读取 requirements-common.txt
    install_requires.extend(_read_requirements_file('requirements/requirements-common.txt'))
    
    # 添加必需的构建依赖（这些不在 requirements 文件中）
    core_deps = [
        "setuptools>=77.0.0",
        "packaging>=24.2",
        "importlib_metadata>=8.5.0",
        "torch==2.7.0", 
        "torchaudio==2.7.0",
        "torchvision==0.22.0",
    ]
    
    # 合并并去重（保留第一次出现的版本）
    all_deps = install_requires + core_deps
    seen = set()
    result = []
    for dep in all_deps:
        # 提取包名（去掉版本号）用于去重
        # 处理各种版本号格式：==, >=, <=, >, <, !=
        pkg_name = dep.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].split("!=")[0].strip()
        pkg_name_lower = pkg_name.lower()
        if pkg_name_lower not in seen:
            seen.add(pkg_name_lower)
            result.append(dep)
    
    return result


def _get_extras_require():
    """构建 extras_require 字典"""
    extras_require = {}
    
    # 获取当前版本号，用于指定 flagscale-megatron-lm 的版本
    from version import FLAGSCALE_VERSION
    
    # robotics-gpu extra
    robotics_gpu_deps = []
    # 添加 common requirements
    robotics_gpu_deps.extend(_read_requirements_file('requirements/requirements-common.txt'))
    # 添加 serving requirements
    robotics_gpu_deps.extend(_read_requirements_file('requirements/serving/requirements.txt'))
    # 添加 robotics serving requirements
    robotics_gpu_deps.extend(_read_requirements_file('requirements/serving/robotics/requirements.txt'))
    # 添加 onnx requirements
    robotics_gpu_deps.extend(_read_requirements_file('requirements/train/robotics/requirements.txt'))
    
    # 添加 flagscale-megatron-lm 包依赖（unpatch 后的 Megatron-LM）
    # 版本号与 flag_scale 主包版本保持一致
    robotics_gpu_deps.append(f"flagscale-megatron-lm=={FLAGSCALE_VERSION}")

    extras_require['robotics-gpu'] = robotics_gpu_deps
    
    return extras_require


from version import FLAGSCALE_VERSION

setup(
    name="flag_scale",
    version=FLAGSCALE_VERSION,
    description="FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). ",
    url="https://github.com/FlagOpen/FlagScale",
    packages=[
        "flag_scale",
        "flag_scale.flagscale",
        "flag_scale.examples",
        "flag_scale.tools",
        "flag_scale.tests",
    ],
    package_dir={
        "flag_scale": "",
        "flag_scale.flagscale": "flagscale",
        "flag_scale.examples": "examples",
        "flag_scale.tools": "tools",
        "flag_scale.tests": "tests",
    },
    package_data={
        "flag_scale.flagscale": ["**/*"],
        "flag_scale.examples": ["**/*"],
        "flag_scale.tools": ["**/*"],
        "flag_scale.tests": ["**/*"],
    },
    install_requires=_get_install_requires(),
    extras_require=_get_extras_require(),
    entry_points={"console_scripts": ["flagscale=flag_scale.flagscale.cli:flagscale"]},
    cmdclass={
        "build": FlagScaleBuild,
        "build_py": FlagScaleBuildPy,
        "build_ext": FlagScaleBuildExt,
    },
)
