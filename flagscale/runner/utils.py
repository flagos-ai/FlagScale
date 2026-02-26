import asyncio
import collections
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
from tqdm.asyncio import tqdm

from flagscale.logger import logger

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


class JobStatus(Enum):
    RUNNING = "Running"
    TRANSITIONAL = "Transitional (Stopping or Starting)"
    COMPLETED_OR_IDLE = "Completed or Not Started"


def log_and_raise_error(message):
    logger.error(message)
    raise ValueError(message)


def validate_serve_config(config: DictConfig):
    """Validate that config has serve field with correct structure.

    Requirements:
    - config must have 'serve' field
    - config.serve must be ListConfig or list type
    - Each element in the list must be a dict and contain 'serve_id' field

    Args:
        config: DictConfig object to validate

    Raises:
        ValueError: If config doesn't have 'serve' field or elements lack 'serve_id'
        TypeError: If config.serve is not ListConfig/list or elements are not dicts
    """
    if not hasattr(config, "serve"):
        logger.error(f"Config content:\n{OmegaConf.to_yaml(config)}")
        raise ValueError("config must have 'serve' field")

    serve_config = config.serve

    # Check if serve_config is ListConfig or list type
    if not isinstance(serve_config, (ListConfig, list)):
        logger.error(
            f"Config validation failed: config.serve must be ListConfig or list type, "
            f"got {type(serve_config).__name__}"
        )
        logger.error(f"Config.serve content: {serve_config}")
        raise TypeError(
            f"config.serve must be ListConfig or list type, got {type(serve_config).__name__}"
        )

    # Check each element is a dict and contains serve_id field
    for idx, item in enumerate(serve_config):
        if not isinstance(item, (dict, DictConfig)):
            logger.error(f"Element at index {idx} content: {item}")
            raise TypeError(
                f"Element at index {idx} in serve list must be a dict or DictConfig, "
                f"got {type(item).__name__}"
            )

        # Convert DictConfig to dict for checking keys
        if isinstance(item, DictConfig):
            item_dict = OmegaConf.to_container(item, resolve=False)
        else:
            item_dict = item

        if "serve_id" not in item_dict:
            logger.error(f"Element at index {idx} content: {item_dict}")
            raise ValueError(f"Element at index {idx} in serve list must contain 'serve_id' field")


def is_ray_master_running(
    master_ip: str, port: int = 6379, timeout: float = 5.0
) -> tuple[bool, str | None]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            logger.info(f"Attempt to connect {master_ip}:{port}, timeout is {timeout}")
            result = s.connect_ex((master_ip, port))

            if result != 0:
                return False, f"waiting for master node {master_ip}:{port}"
            else:
                return True, "master node is ready"

    except TimeoutError:
        return False, f"connect {master_ip}:{port} timeout"
    except Exception as e:
        return False, f"check the error: {e!s}"


def wait_for_ray_master(
    master_ip: str, port: int = 6379, max_attempts: int = 180, interval: int = 10
) -> bool:
    logger.info(f"Master info is {master_ip}:{port}")
    for attempt in range(max_attempts):
        status, msg = is_ray_master_running(master_ip, port)
        logger.info(f"Check Ray master status (attempt {attempt + 1}/{max_attempts}): {msg}")

        if status:
            return True

        if attempt < max_attempts - 1:
            time.sleep(interval)

    return False


def parse_hostfile(hostfile_path):
    if hostfile_path is None or not os.path.isfile(hostfile_path):
        logger.warning(
            f"Hostfile {hostfile_path} not found. The training will proceed using only local resources."
        )
        return None

    # e.g., worker0 slots=8 type=A100
    pattern = re.compile(r"^(\S+)\s+slots=(\d+)(?:\s+type=(\S+))?")

    resources = collections.OrderedDict()

    with open(hostfile_path, "r") as fd:
        hostfile_lines = fd.readlines()

    for line in hostfile_lines:
        line = line.strip()
        match = pattern.search(line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            machine_type = match.group(3) if match.group(3) else None
            if host in resources:
                log_and_raise_error(f"Hostfile contains multiple entries for host: {host}.")
            resources[host] = {"slots": num_slots, "type": machine_type}
        else:
            log_and_raise_error(f"Invalid entry in hostfile: {line}.")

    assert all(info["type"] is None for _, info in resources.items()) or all(
        info["type"] is not None for _, info in resources.items()
    ), "All hosts must have the a machine type or no machine type specified."

    if len(resources) == 0:
        log_and_raise_error(
            "Hostfile is empty or not formatted correctly. Please check the hostfile."
        )

    return resources


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_host_name_or_ip():
    host_name = socket.gethostname()
    if host_name:
        return host_name
    try:
        # doesn't even have to be reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        IP = sock.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        if (
            "sock" in locals()
        ):  # Ensure 'sock' was successfully created before attempting to close it
            sock.close()
    return IP


def get_addr():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith == "127.0.0.1":
                return ip
    except:
        pass

    try:
        ip = socket.gethostbyname(socket.getfqdn())
        if not ip.startswith == "127.0.0.1":
            return ip
    except:
        pass

    return socket.gethostname()


def run_local_command(cmd, dryrun=False, query=False):
    logger.info(f"Run the local command: {cmd}")
    if dryrun:
        return
    if query:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result
    else:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"Command {cmd} failed with return code {result.returncode}.")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            sys.exit(result.returncode)


def run_ssh_command(host, cmd, port=None, dryrun=False, query=False, background=True):
    if background:
        if port:
            ssh_cmd = f"ssh -f -n -p {port} {host} '{cmd}'"
        else:
            ssh_cmd = f"ssh -f -n {host} '{cmd}'"
    else:
        if port:
            ssh_cmd = f"ssh -n -p {port} {host} '{cmd}'"
        else:
            ssh_cmd = f"ssh -n {host} '{cmd}'"
    if not query:
        logger.info(f"Running the ssh command: {ssh_cmd}")
    if dryrun:
        return
    if background:
        result = subprocess.run(
            ssh_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"SSH command {ssh_cmd} failed with return code {result.returncode}.")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            sys.exit(result.returncode)
    else:
        result = subprocess.run(
            ssh_cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0 and not query:
            print(f"SSH command {ssh_cmd} failed with return code {result.returncode}.")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            sys.exit(result.returncode)
    if query:
        return result


def get_remote_file_mtime(host, filepath, port=22):
    """
    Get modification time of a file on a remote host (in seconds since epoch).

    Args:
        host (str): The address of the remote host (e.g., 'user@hostname').
        filepath (str): The path to the file on the remote host.
        port (int): SSH port number (default: 22).

    Returns:
        int: The modification time in seconds since epoch if successful; returns -1 if an error occurs.
    """
    try:
        cmd = ["ssh", "-p", str(port), host, f"stat -c%Y {filepath}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        return -1


def get_remote_file_size(host, filepath, port=22):
    """
    Get size of a file on a remote host (in bytes).

    Args:
        host (str): The address of the remote host (e.g., 'user@hostname').
        filepath (str): The path to the file on the remote host.
        port (int): SSH port number (default: 22).

    Returns:
        int: The size in bytes if successful; returns -1 if an error occurs.
    """
    try:
        cmd = ["ssh", "-p", str(port), host, f"stat -c%s {filepath}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception:
        return -1


def run_scp_command(host, src, dst, port=None, dryrun=False):
    if port:
        scp_cmd = f"scp -P {port} -r {src} {host}:{dst} "
    else:
        scp_cmd = f"scp -r {src} {host}:{dst} "
    logger.info(f"Run the scp command: {scp_cmd}")
    if dryrun:
        return
    result = subprocess.run(
        scp_cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"SCP command {scp_cmd} failed with return code {result.returncode}.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        sys.exit(result.returncode)


def flatten_dict_to_args_verl(config_dict, pre_str=""):
    args = []
    if "config-path" in config_dict:
        args.append(f"--config-path={config_dict['config-path']}")
        config_dict.pop("config-path")

    if "config-name" in config_dict:
        args.append(f"--config-name={config_dict['config-name']}")
        config_dict.pop("config-name")

    for key, value in config_dict.items():
        if isinstance(value, dict):
            if key == "append_kargs":
                target_str = "+"
            else:
                target_str = f"{key}."
            args.extend(flatten_dict_to_args_verl(value, pre_str + target_str))
        elif isinstance(value, list):
            json_str = json.dumps(value)
            args.append(f"{pre_str + key}={json_str}")
        elif isinstance(value, bool):
            args.append(f"{pre_str + key}={value}")
        else:
            args.append(f"{pre_str + key}=" + f"{value}")

    return args


def flatten_dict_to_args(config_dict, ignore_keys=[], do_dash_replace=True):
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        if do_dash_replace:
            key = key.replace("_", "-")
        if isinstance(value, dict):
            args.extend(flatten_dict_to_args(value, ignore_keys, do_dash_replace=do_dash_replace))
        elif isinstance(value, list):
            args.append(f"--{key}")
            for v in value:
                args.append(f"{v}")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(f"{value}")
    return args


def get_nnodes(nnodes_from_hostfile=None, nnodes_from_args=None):
    assert nnodes_from_hostfile is not None or nnodes_from_args is not None
    if nnodes_from_hostfile is not None and nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return min(nnodes_from_hostfile, int(nnodes_from_args))
    elif nnodes_from_hostfile is not None:
        return nnodes_from_hostfile
    elif nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return int(nnodes_from_args)


def get_nproc_per_node(nproc_from_hostfile=None, nproc_from_args=None, num_visible_devices=None):
    if nproc_from_hostfile is not None and nproc_from_args is not None:
        nproc = min(nproc_from_hostfile, int(nproc_from_args))
        if num_visible_devices:
            return min(nproc, num_visible_devices)
        else:
            return nproc
    elif nproc_from_hostfile is not None:
        if num_visible_devices:
            return min(nproc_from_hostfile, num_visible_devices)
        else:
            return nproc_from_hostfile
    elif nproc_from_args is not None:
        if num_visible_devices:
            return min(int(nproc_from_args), num_visible_devices)
        else:
            return nproc_from_args
    else:
        if num_visible_devices:
            return num_visible_devices
        else:
            return 1


def update_nodes_envs(env_config, ip_addr, resource_info):
    device_type = resource_info.get("type", None)
    assert ip_addr is not None, "ip address is required in resource_info"

    cur_node_config = {}
    if isinstance(env_config, DictConfig):
        cur_node_config = OmegaConf.to_container(env_config, resolve=True)
    else:
        cur_node_config = env_config.copy()

    device_types_envs = cur_node_config.pop("device_type_specific", None)
    nodes_envs = cur_node_config.pop("node_specific", None)
    if device_types_envs is None and nodes_envs is None:
        logger.warning(
            "type in hostfile is not specified. All the nodes use the same arguments inlucding evnironment variables."
        )
        return cur_node_config

    # update then envs according to the device type
    if device_types_envs is not None and device_type is not None:
        cur_node_config.update(device_types_envs.get(device_type, {}))
    # update the envs according to the ip address
    if nodes_envs is not None:
        cur_node_config.update(nodes_envs.get(ip_addr, {}))

    return cur_node_config


def add_decive_extra_config(config, device_type):
    if device_type is None:
        return OmegaConf.to_container(config, resolve=True)
    cur_node_config = {}
    temp_dict = {}
    if isinstance(config, DictConfig):
        temp_dict = OmegaConf.to_container(config, resolve=True)
    else:
        temp_dict = config
    for key, value in temp_dict.items():
        if isinstance(value, dict):
            if key == device_type:
                cur_node_config.update(value)
            else:
                cur_node_config[key] = value
    return cur_node_config


def update_cmd_with_node_specific_config(cmd, node_specific_config=None):
    """
    Update the command string with additional configuration options for speicial device.
    Parameters:
        cmd (str): The original command string to be updated.
        node_specific_config (dict): A dictionary containing configuration
                                     options to be applied to the command.
    Returns:
        str: The updated command string after applying the configuration.
    """
    if node_specific_config is None or len(node_specific_config) == 0:
        return cmd
    cmd_parts = shlex.split(cmd)

    for key, value in node_specific_config.items():
        key = key.replace("_", "-")
        option = f"--{key}"
        if option in cmd_parts:
            idx = cmd_parts.index(option)
            if value.lower() == "true":
                # If the option has a value, remove it.
                if idx + 1 < len(cmd_parts) and not cmd_parts[idx + 1].startswith("--"):
                    cmd_parts.pop(idx + 1)
                continue
            elif value.lower() == "false":
                # If the option has a value, remove it along with the option.
                if idx + 1 < len(cmd_parts) and not cmd_parts[idx + 1].startswith("--"):
                    del cmd_parts[idx : idx + 2]
                else:
                    cmd_parts.pop(idx)
        else:
            if value.lower() == "true":
                cmd_parts.append(option)
            # NOTE: disable adding new options
            # elif value.lower() != "false":
            #     cmd_parts.extend([option, value])
    return shlex.join(cmd_parts)


def is_ip_addr(master):
    """Check if master is ip address."""

    if not isinstance(master, str):
        return False
    pattern = r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
    result = re.match(pattern, master)
    return bool(result)


def is_master_node(lws_leader_address):
    host_name = lws_leader_address.split(".")[0]
    local_hostname = socket.gethostname().split(".")[0]

    return host_name == local_hostname


def get_ip_addr():
    """Get ip address."""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(socket.getfqdn(hostname))
    except:
        ip = "127.0.0.1"
    return ip


def is_master(config, resources=None):
    """Check if current node is master."""
    nnodes = config.experiment.runner.get("nnodes", 1)

    hostfile = None
    if config.experiment.runner.get("hostfile", None):
        hostfile = config.experiment.runner["hostfile"]
    if os.environ.get("AIRS_SWITCH", None):
        if os.environ.get("AIRS_HOSTFILE_PATH", None):
            hostfile = os.environ["AIRS_HOSTFILE_PATH"]

    if not resources:
        resources = parse_hostfile(hostfile)
    if not resources and nnodes > 1:
        raise ValueError("In the multi-node mode, please set the hostfile")

    if resources:
        master = next(iter(resources.keys()))
        if is_ip_addr(master):
            return get_ip_addr() in [master, "127.0.0.1"]
        else:
            output = subprocess.run(
                "hostname", check=True, shell=True, text=True, capture_output=True
            )
            hostname = output.stdout.strip()
            return hostname == master
    # Local host Scene
    return True


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = None
    best_of: int = 1
    logprobs: int | None = None
    extra_body: dict | None = None
    multi_modal_content: dict | None = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""


def dummy_random_input(
    tokenizer, prefix_len=0, input_len=1024, output_len=1024, num_prompts=1000, range_ratio=1.0
):
    prefix_token_ids = np.random.randint(0, tokenizer.vocab_size, size=prefix_len).tolist()

    input_lens = np.random.randint(int(input_len * range_ratio), input_len + 1, size=num_prompts)
    output_lens = np.random.randint(int(output_len * range_ratio), output_len + 1, size=num_prompts)
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(
            prefix_token_ids
            + [(offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])]
        )

        input_requests.append((prompt, int(prefix_len + input_lens[i]), int(output_lens[i]), None))

    return input_requests


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": (
                request_func_input.model_name
                if request_func_input.model_name
                else request_func_input.model
            ),
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
            # max_completion_tokens is invalid for llama.cpp
            "n_predict": request_func_input.output_len,
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""

                            # llamap.cpp's last response has "choices", bot delta is null
                            # sglang's response has key "usage" but value is null
                            if usage := data.get("usage", {}):
                                if completion_tokens := usage.get("completion_tokens"):
                                    output.output_tokens = completion_tokens

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def get_request(input_requests):
    input_requests = iter(input_requests)
    # Calculate scale parameter theta to maintain the desired request_rate.
    for request in input_requests:
        yield request


async def benchmark(
    api_url,
    model,
    served_model_name,
    tokenizer,
    input_requests,
    selected_percentile_metrics,
    selected_percentiles,
):
    async def limited_request_func(request_func_input, pbar):
        return await request_func(request_func_input=request_func_input, pbar=pbar)

    request_func = async_request_openai_chat_completions
    req_model_id = model
    req_model_name = served_model_name if served_model_name is not None else model

    pbar = tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests):
        prompt, prompt_len, output_len, mm_content = request

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            multi_modal_content=mm_content,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs = await asyncio.gather(*tasks)
    pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    ### import here to avoid dependency issue
    from flagscale.serve.metric import calculate_metrics

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print(
        "{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput)
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):", getattr(metrics, f"mean_{metric_attribute_name}_ms")
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


class ResourceManager:
    def __init__(self, nodes):
        """
        Initialize the ResourceManager with a list of nodes.
        Each element in the list should be a two-item list:
          - The first item is the node address (a string).
          - The second item is a dictionary containing at least the key "slots".
            If "type" is not provided, it defaults to "gpu" with a warning.
        The first node is treated as the master node, and the rest are worker nodes.
        """
        self.nodes = self._initialize_nodes(nodes)

    def _initialize_nodes(self, nodes):
        """
        Convert the input nodes list into the internal nodes representation.
        Each node is converted into a dictionary with keys:
          "address", "slots", "type", and "used" (initialized to 0).
        If the "type" is not provided in a node, default it to "gpu" and issue a warning.
        """
        initialized_nodes = []
        for node in nodes:
            if len(node) != 2:
                raise ValueError("Each node must include an address and node data")
            address, info = node
            if "slots" not in info:
                raise ValueError("Node data must contain 'slots'")
            if "type" not in info:
                logger.warning(
                    f"Node {address} does not provide a resource type. Defaulting to 'gpu'."
                )
            resource_type = info.get("type", "gpu")
            initialized_nodes.append(
                {
                    "address": address,
                    "slots": info["slots"],
                    "type": resource_type,
                    "used": 0,  # Initialize used slot count to 0
                }
            )
        return initialized_nodes

    def get_whole_card_num(self, resource_type="gpu"):
        """
        Return the total number of slots across all nodes with the specified resource type.
        The return type is int.
        """
        total = 0
        for node in self.nodes:
            if node["type"] == resource_type:
                total += node["slots"]
        return total

    def get_available_card_num(self, resource_type="gpu"):
        """
        Return the total number of available slots (slots minus used) across all nodes with the specified resource type.
        The return type is int.
        """
        total = 0
        for node in self.nodes:
            if node["type"] == resource_type:
                total += node["slots"] - node["used"]
        return total

    def get_available_card_ids(self, resource_type="gpu", address="auto", num=1):
        """
        Allocate 'num' resource cards from a node and return a list of card indices.

        For the default case (address="auto"), traverse nodes in order: master node first, then worker nodes.
        - If a node's available slots (slots - used) are >= num, allocate num consecutive indices (based on the current used value)
          and update the node's used count, returning the allocated indices (0-indexed) as a list.
        - If the available slots are insufficient at a particular node and address is "auto", continue searching through other nodes.
        - If an explicit address is provided, check only that node; if it doesn't exist or lacks sufficient available slots, raise an error.
        - If none of the nodes can satisfy the request, raise an error indicating insufficient resources.
        """
        # Check the specified node if address is not "auto"
        if address != "auto":
            node_found = None
            for node in self.nodes:
                if node["address"] == address and node["type"] == resource_type:
                    node_found = node
                    break
            if node_found is None:
                raise ValueError(f"Node {address} does not exist or resource type mismatch")
            free = node_found["slots"] - node_found["used"]
            if free < num:
                raise ValueError("Insufficient resources")
            allocated_ids = list(range(node_found["used"], node_found["used"] + num))
            node_found["used"] += num
            return allocated_ids, address

        # For address == "auto", traverse all nodes (master node first, then worker nodes)
        for node in self.nodes:
            if node["type"] == resource_type:
                free = node["slots"] - node["used"]
                if free >= num:
                    allocated_ids = list(range(node["used"], node["used"] + num))
                    node["used"] += num
                    return allocated_ids, node["address"]

        # If no node satisfies the allocation request, raise an error.
        resource_status = self.get_status()
        raise ValueError(
            f"Require number {num} of resource_type {resource_type} But there is insufficient resources: \n{resource_status}"
        )

    def get_status(self):
        """
        Return the status of all nodes as a dictionary.
        Each key in the returned dictionary is the node's address, and its value is a dictionary with:
          - type: the resource type.
          - slots: the total number of slots.
          - used: the number of allocated slots.
          - available: the number of available slots (slots - used).
        """
        status = {}
        for node in self.nodes:
            status[node["address"]] = {
                "type": node["type"],
                "slots": node["slots"],
                "used": node["used"],
                "available": node["slots"] - node["used"],
            }
        return status
