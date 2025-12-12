import json
import os
import shutil
import subprocess
import sys

from datetime import datetime

from tools.patch.unpatch import apply_hardware_patch, unpatch

SUPPORTED_DEVICES = ["cpu", "gpu", "ascend", "cambricon", "bi", "metax", "kunlunxin", "musa"]
VLLM_UNPATCH_DEVICES = ["ascend", "cambricon", "bi", "metax", "kunlunxin"]


def _get_cuda_tag():
    """get CUDA tag, e.g. cu128"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # extract version from "Cuda compilation tools, release 12.8, V12.8.93"
            import re

            match = re.search(r'release (\d+)\.(\d+)', result.stdout)
            if match:
                major, minor = match.groups()
                return f"cu{major}{minor}"
    except FileNotFoundError:
        pass
    return None


def run_subprocess_with_error_capture(
    cmd, cwd=None, env=None, description="Command", log_file=None
):
    """
    Run a subprocess with real-time output to both console and log file.

    Args:
        cmd: Command to run (list)
        cwd: Working directory
        env: Environment variables
        description: Description of the command for error messages
        log_file: Path to log file (default: build.log in current directory)

    Raises:
        subprocess.CalledProcessError: If command fails, with detailed error info
    """
    # Default log file
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "build.log")

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    try:
        # Open log file for appending
        with open(log_file, 'a', encoding='utf-8') as log_f:
            # Write header
            log_f.write(f"\n{'=' * 80}\n")
            log_f.write(f"[{description}] Command: {' '.join(cmd)}\n")
            log_f.write(f"[{description}] CWD: {cwd or os.getcwd()}\n")
            log_f.write(f"[{description}] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"{'=' * 80}\n")
            log_f.flush()

            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # Collect output for error reporting
            output_lines = []

            # Read and write output in real-time
            for line in process.stdout:
                # Write to console
                print(line, end='')
                sys.stdout.flush()

                # Write to log file
                log_f.write(line)
                log_f.flush()

                # Collect for error reporting
                output_lines.append(line)

            # Wait for process to complete
            returncode = process.wait()

            # Write footer
            log_f.write(f"\n[{description}] Exit code: {returncode}\n")
            log_f.write(f"{'=' * 80}\n\n")
            log_f.flush()

            # Check if the command failed
            if returncode != 0:
                output = ''.join(output_lines)
                error_msg = f"\n[builder] {description} failed with exit code: {returncode}"
                cmd_msg = f"[builder] Command: {' '.join(cmd)}"

                print(error_msg)
                print(cmd_msg)

                # Write error summary to log
                log_f.write(f"{error_msg}\n")
                log_f.write(f"{cmd_msg}\n")

                # Show last 50 lines of output for context
                if output_lines:
                    context_msg = f"[builder] Last output lines:"
                    print(context_msg)
                    print("=" * 80)
                    for line in output_lines[-50:]:
                        print(line, end='')
                    print("=" * 80)

                    log_f.write(f"{context_msg}\n")
                    log_f.write("=" * 80 + "\n")
                    for line in output_lines[-50:]:
                        log_f.write(line)
                    log_f.write("=" * 80 + "\n")

                print(f"[builder] Full log saved to: {log_file}")

                raise subprocess.CalledProcessError(returncode, cmd, output=output)

            print(f"[builder] Log saved to: {log_file}")
            return returncode

    except subprocess.CalledProcessError:
        raise
    except Exception as e:
        error_msg = f"[builder] Unexpected error running {description}: {e}"
        print(error_msg)

        # Write error to log file
        try:
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n{error_msg}\n")
                import traceback

                log_f.write(traceback.format_exc())
        except Exception as e:
            print(f"[builder] Unexpected error writing to log file: {e}")

        import traceback

        traceback.print_exc()
        raise


def check_backend(backend):
    if backend not in ["llama.cpp", "Megatron-LM", "sglang", "vllm", "Megatron-Energon"]:
        raise ValueError(f"Invalid backend {backend}.")


def check_backends(backends):
    for backend in backends:
        check_backend(backend)


def check_vllm_unpatch_device(device):
    is_supported = False
    for supported_device in VLLM_UNPATCH_DEVICES:
        if supported_device in device.lower():
            is_supported = True
            return is_supported
    return is_supported


def check_device(device):
    for supported_device in SUPPORTED_DEVICES:
        if supported_device in device.lower():
            return
    raise ValueError(f"Unsupported device {device}. Supported devices are {SUPPORTED_DEVICES}.")


def _parse_torch_versions_from_cuda_txt(root_dir):
    """
    Parse torch/torchaudio/torchvision versions from third_party/vllm/requirements/cuda.txt

    Args:
        root_dir: FlagScale root directory

    Returns:
        dict: Dictionary containing 'torch', 'torchaudio', 'torchvision' version numbers
    """
    import re

    cuda_txt_path = os.path.join(root_dir, "third_party", "vllm", "requirements", "cuda.txt")

    if not os.path.exists(cuda_txt_path):
        raise FileNotFoundError(f"cuda.txt not found at {cuda_txt_path}")

    versions = {}
    with open(cuda_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '#' in line:
                line = line[: line.index('#')]
            line = line.strip()
            if not line:
                continue

            for package in ['torch', 'torchaudio', 'torchvision']:
                pattern = rf'^{package}==([\d.]+)'
                match = re.search(pattern, line)
                if match:
                    versions[package] = match.group(1)

    required_packages = ['torch', 'torchaudio', 'torchvision']
    missing = [pkg for pkg in required_packages if pkg not in versions]
    if missing:
        raise ValueError(f"Missing torch package versions in cuda.txt: {missing}")

    return versions


def build_vllm(device, root_dir):
    assert device != "cpu"
    vllm_path = os.path.join(root_dir, "third_party", "vllm")
    if device != "gpu":
        vllm_path = os.path.join(root_dir, "build", device, "FlagScale", "third_party", "vllm")
    # Set env
    env = os.environ.copy()
    print(f"[builder] Environment: {env}")
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
    try:
        if device == "gpu":
            # prevent incompatible torch version when building vllm
            torch_versions = _parse_torch_versions_from_cuda_txt(root_dir)
            run_subprocess_with_error_capture(
                [
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    f'torch=={torch_versions["torch"]}',
                    f'torchvision=={torch_versions["torchvision"]}',
                    f'torchaudio=={torch_versions["torchaudio"]}',
                    '--extra-index-url',
                    f'https://download.pytorch.org/whl/{_get_cuda_tag()}',
                ],
                description="vllm build",
            )

        env["MAX_JOBS"] = str(os.environ.get("MAX_JOBS", 32))
        run_subprocess_with_error_capture(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '--no-cache-dir',
                '.',
                '--verbose',
                '--no-build-isolation',
            ],
            cwd=vllm_path,
            env=env,
            description="vllm build",
        )
    except subprocess.CalledProcessError:
        print(f"[builder] Failed to build vllm. Check the error output above.")
        raise


def build_sglang(device, root_dir):
    assert device != "cpu"
    sglang_path = os.path.join(root_dir, "third_party", "sglang")
    if device != "gpu":
        sglang_path = os.path.join(root_dir, "build", device, "FlagScale", "third_party", "sglang")
    try:
        run_subprocess_with_error_capture(
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
            description="sglang build",
        )
    except subprocess.CalledProcessError:
        print(f"[builder] Failed to build sglang. Check the error output above.")
        raise


def build_llama_cpp(device, root_dir):
    llama_cpp_path = os.path.join(root_dir, "third_party", "llama.cpp")
    print(f"[build_ext] Build llama.cpp for {device}")
    try:
        if device == "gpu":
            run_subprocess_with_error_capture(
                ["cmake", "-B", "build", "-DGGML_CUDA=ON"],
                cwd=llama_cpp_path,
                description="llama.cpp cmake configure (GPU)",
            )
            run_subprocess_with_error_capture(
                ["cmake", "--build", "build", "--config", "Release", "-j64"],
                cwd=llama_cpp_path,
                description="llama.cpp build (GPU)",
            )
        elif device == "musa":
            run_subprocess_with_error_capture(
                ["cmake", "-B", "build", "-DGGML_MUSA=ON"],
                cwd=llama_cpp_path,
                description="llama.cpp cmake configure (MUSA)",
            )
            run_subprocess_with_error_capture(
                ["cmake", "--build", "build", "--config", "Release", "-j8"],
                cwd=llama_cpp_path,
                description="llama.cpp build (MUSA)",
            )
        elif device == "cpu":
            run_subprocess_with_error_capture(
                ["cmake", "-B", "build"],
                cwd=llama_cpp_path,
                description="llama.cpp cmake configure (CPU)",
            )
            run_subprocess_with_error_capture(
                ["cmake", "--build", "build", "--config", "Release", "-j8"],
                cwd=llama_cpp_path,
                description="llama.cpp build (CPU)",
            )
        else:
            raise ValueError(f"Unsupported device {device} for llama.cpp backend.")
    except subprocess.CalledProcessError:
        print(f"[builder] Failed to build llama.cpp. Check the error output above.")
        raise


def build_megatron_energon(device, root_dir):
    try:
        import editables
        import hatch_vcs
        import hatchling
    except Exception:
        try:
            print("[INFO] hatchling not found. Installing...")
            run_subprocess_with_error_capture(
                [sys.executable, "-m", "pip", "install", "hatchling", "--no-build-isolation"],
                description="hatchling installation",
            )
            run_subprocess_with_error_capture(
                [sys.executable, "-m", "pip", "install", "hatch-vcs", "--no-build-isolation"],
                description="hatch-vcs installation",
            )
            run_subprocess_with_error_capture(
                [sys.executable, "-m", "pip", "install", "editables", "--no-build-isolation"],
                description="editables installation",
            )
            import editables
            import hatch_vcs
            import hatchling
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install hatchling, hatch-vcs and editables.")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to install hatchling, hatch-vcs and editables: {e}")
            sys.exit(1)

    energon_path = os.path.join(root_dir, "third_party", "Megatron-Energon")
    try:
        run_subprocess_with_error_capture(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '-e',
                '.[av_decode]',
                '--no-build-isolation',
                '--verbose',
            ],
            cwd=energon_path,
            description="Megatron-Energon build",
        )

        # Copy energon to Megatron-LM directory
        print(f"[builder] Copying energon to Megatron-LM directory...")
        energon_src = os.path.join(
            root_dir, "third_party", "Megatron-Energon", "src", "megatron", "energon"
        )
        energon_dst = os.path.join(root_dir, "third_party", "Megatron-LM", "megatron", "energon")

        if not os.path.exists(energon_src):
            raise FileNotFoundError(f"Energon source directory not found: {energon_src}")

        # Check if Megatron-LM exists
        megatron_lm_path = os.path.join(root_dir, "third_party", "Megatron-LM")
        if not os.path.exists(megatron_lm_path):
            print(f"[builder] Warning: Megatron-LM not found at {megatron_lm_path}")
            print(f"[builder] Megatron-Energon requires Megatron-LM to be initialized first")
            raise ValueError("Megatron-LM must be initialized before building Megatron-Energon")

        # Remove existing energon directory if present
        if os.path.exists(energon_dst):
            print(f"[builder] Removing existing energon directory: {energon_dst}")
            shutil.rmtree(energon_dst)

        # Copy energon directory
        print(f"[builder] Copying {energon_src} -> {energon_dst}")
        shutil.copytree(energon_src, energon_dst)
        print(f"[builder] Successfully copied energon to Megatron-LM")

    except subprocess.CalledProcessError:
        print(f"[builder] Failed to build Megatron-Energon. Check the error output above.")
        raise


def build_megatron_lm(device, root_dir):
    """
    Build Megatron-LM dependencies.
    This installs flash-attention, transformer-engine, and apex.

    Args:
        device: Device type (should be "gpu" for Megatron-LM)
        root_dir: Root directory of FlagScale
    """
    if device == "gpu":
        # Path to the build script
        build_script = os.path.join(root_dir, "install", "build-megatron-deps-nvidia.sh")
        if not os.path.exists(build_script):
            raise FileNotFoundError(f"Build script not found: {build_script}")
        print(f"[builder] Executing build script: {build_script}")
        try:
            # Make sure the script is executable
            os.chmod(build_script, 0o755)

            # Execute the build script with error capture
            run_subprocess_with_error_capture(
                ["bash", build_script],
                cwd=os.path.dirname(build_script),
                description="Megatron-LM dependencies build script",
            )

            print(f"[builder] Successfully built Megatron-LM dependencies")
        except subprocess.CalledProcessError:
            # Error details already printed by run_subprocess_with_error_capture
            raise
        except Exception as e:
            print(f"[builder] Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            raise


def unpatch_backend(backend, device, root_dir):
    if backend == "FlagScale":
        return

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

    dst = os.path.join(root_dir, "third_party", backend)
    src = os.path.join(root_dir, "flagscale", "backends", backend)

    # Marker file path to track unpatch status
    marker_file = os.path.join(dst, ".flagscale_unpatched")

    # Check if unpatch has already been completed
    if os.path.exists(marker_file):
        try:
            with open(marker_file, 'r', encoding='utf-8') as f:
                marker_data = json.load(f)

            # Verify that the marker file information matches current configuration
            if (
                marker_data.get("backend") == backend
                and marker_data.get("device") == device
                and marker_data.get("backend_commit") == backend_commit
            ):
                print(
                    f"[build_py] Backend {backend} for device {device} has already been unpatched. Skipping unpatch."
                )
                print(f"[build_py] Marker file: {marker_file}")
                return
            else:
                print(f"[build_py] Marker file exists but configuration changed. Re-unpatching...")
                os.remove(marker_file)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[build_py] Invalid marker file format: {e}. Re-unpatching...")
            os.remove(marker_file)

    print(f"[build_py] Device {device} initializing the {backend} backend.")
    # backend_commit needs to be in dictionary format {backend_name: commit}
    backend_commit_dict = {backend: backend_commit} if backend_commit else {}
    unpatch(
        root_dir,
        src,
        dst,
        backend,
        force=False,
        backend_commit=backend_commit_dict,
        fs_extension=True,
    )

    # Create marker file after successful unpatch
    try:
        marker_data = {"backend": backend, "device": device, "backend_commit": backend_commit}
        # Ensure directory exists
        os.makedirs(dst, exist_ok=True)
        with open(marker_file, 'w', encoding='utf-8') as f:
            json.dump(marker_data, f, indent=2)
        print(f"[build_py] Created unpatch marker file: {marker_file}")
    except Exception as e:
        print(f"[build_py] Warning: Failed to create marker file: {e}")


def unpatch_hardware_backend(backend, device, build_lib, root_dir):
    print(f"[build_py] Device {device} unpatching the vllm backend.")
    # Unpatch the backend in specified device
    from git import Repo

    main_repo = Repo(root_dir)
    commit = os.getenv("FLAGSCALE_UNPATCH_COMMIT", None)
    if commit is None:
        commit = main_repo.head.commit.hexsha
    # Checkout to the commit and apply the patch to build FlagScale
    key_path = os.getenv("FLAGSCALE_KEY_PATH", None)
    apply_hardware_patch(device, backend, commit, root_dir, True, key_path=key_path)


def build_backend(backend, device, root_dir):
    """
    Build a backend from source (without unpatch).
    This function assumes the backend has already been unpatched.

    Args:
        backend: Backend name (e.g., "vllm", "Megatron-LM")
        device: Device type (e.g., "gpu", "cpu", "metax")
        root_dir: Root directory of FlagScale
    """
    print(f"[builder] Building {backend} for {device}...")

    # Determine the correct path for building
    # For hardware-specific devices, the code is in build/{device}/FlagScale/third_party/
    if check_vllm_unpatch_device(device) and (backend == "vllm" or backend == "Megatron-LM"):
        build_root = os.path.join(root_dir, "build", device, "FlagScale")
        print(f"[builder] Using hardware-specific build path: {build_root}")
    else:
        build_root = root_dir

    if backend == "vllm":
        build_vllm(device, build_root)
        print(f"[builder] Successfully built and installed vllm")
    elif backend == "sglang":
        build_sglang(device, build_root)
        print(f"[builder] Successfully built and installed sglang")
    elif backend == "llama.cpp":
        build_llama_cpp(device, build_root)
        print(f"[builder] Successfully built llama.cpp")
    elif backend == "Megatron-LM":
        # Megatron-LM itself is pure Python, but we need to build its dependencies
        build_megatron_lm(device, build_root)
        print(f"[builder] Megatron-LM dependencies installed, source code is ready to use")
    elif backend == "Megatron-Energon":
        build_megatron_energon(device, build_root)
        print(f"[builder] Successfully built and installed Megatron-Energon")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print(f"[builder] Completed build for {backend}")


def build_and_install_backend(backend, device, root_dir):
    """
    Unpatch and build a backend from source.
    This function first unpatches the backend to initialize submodules and apply patches,
    then builds and installs the backend.

    Args:
        backend: Backend name (e.g., "vllm", "Megatron-LM")
        device: Device type (e.g., "gpu", "cpu", "metax")
        root_dir: Root directory of FlagScale
    """
    print(f"[builder] Starting build and install for {backend} on {device}")

    # Step 1: Unpatch the backend first
    print(f"[builder] Step 1: Unpatching {backend}...")

    # At present, only vLLM supports domestic chips, and the remaining backends have not been supported yet.
    # FlagScale just modified the vLLM and Megatron-LM
    if backend == "vllm" or backend == "Megatron-LM":
        if check_vllm_unpatch_device(device):
            # Use hardware-specific unpatch for domestic chips
            print(f"[builder] Using hardware-specific unpatch for {device}")
            try:
                # For hardware unpatch, we need to pass backend as a list
                backend_list = [backend] if isinstance(backend, str) else backend
                unpatch_hardware_backend(backend_list, device, None, root_dir)
                print(f"[builder] Successfully unpatched {backend} with hardware patch")
            except Exception as e:
                print(f"[builder] Warning: Failed to unpatch {backend} with hardware patch: {e}")
                raise
        else:
            # Use standard unpatch for GPU
            try:
                unpatch_backend(backend, device, root_dir)
                print(f"[builder] Successfully unpatched {backend}")
            except Exception as e:
                print(f"[builder] Warning: Failed to unpatch {backend}: {e}")
                # Continue anyway as some backends might not need unpatching
    else:
        # Other backends use standard unpatch
        try:
            unpatch_backend(backend, device, root_dir)
            print(f"[builder] Successfully unpatched {backend}")
        except Exception as e:
            print(f"[builder] Warning: Failed to unpatch {backend}: {e}")
            # Continue anyway as some backends might not need unpatching

    # Step 2: Build and install the backend
    print(f"[builder] Step 2: Building {backend}...")
    build_backend(backend, device, root_dir)
