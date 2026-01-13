import logging as logger
import os
import socket
import subprocess
import sys

from omegaconf import OmegaConf

from flagscale import serve
from flagscale.runner.utils import get_free_port
from flagscale.serve.args_mapping.mapping import ARGS_CONVERTER
from flagscale.utils import flatten_dict_to_args


def check_port_occupied(ip_addr, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((ip_addr, port))
            return False
    except OSError:
        return True


def vllm_serve(args):
    common_args = args.get("engine_args", {})
    vllm_args = args.get("engine_args_specific", {}).get("vllm", {})
    vllm_args.update(common_args)
    command = ["vllm", "serve"]
    if vllm_args.get("model", None):
        command.append(vllm_args["model"])
        other_args = flatten_dict_to_args(vllm_args, ["model"])
        command.extend(other_args)
    else:
        raise ValueError("Either model must be specified in model config.")

    # Start the subprocess
    logger.info(f"[Serve]: Starting vllm serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current vLLM PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def llama_cpp_serve(args):
    common_args = args.get("engine_args", {})
    llama_cpp_args = args.get("engine_args_specific", {}).get("llama_cpp", {})

    command = ["llama-server"]
    if common_args.get("model", None):
        converted_args = ARGS_CONVERTER.convert("llama_cpp", common_args)
        command.extend(["--model", converted_args["model"]])
        common_args_flatten = flatten_dict_to_args(converted_args, ["model"])
        command.extend(common_args_flatten)
        llama_cpp_args_flatten = flatten_dict_to_args(llama_cpp_args, ["model"])
        command.extend(llama_cpp_args_flatten)
    else:
        raise ValueError("Either model must be specified in model config.")

    # Start the subprocess
    logger.info(f"[Serve]: Starting llama-cpp serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current Llama PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def sglang_serve(args):
    common_args = args.get("engine_args", {})
    sglang_args = args.get("engine_args_specific", {}).get("sglang", {})
    if sglang_args.get("dist-init-addr", None):
        logger.warning(
            f"sglang dist-init-addr:{sglang_args['dist-init-addr']} exists, will be overwrite by master_addr, master_port"
        )
        was_struct = OmegaConf.is_struct(sglang_args)
        OmegaConf.set_struct(sglang_args, False)
        sglang_args.pop("dist-init-addr")
        if was_struct:
            OmegaConf.set_struct(sglang_args, True)

    command = ["python", "-m", "sglang.launch_server"]
    if common_args.get("model", None):
        converted_args = ARGS_CONVERTER.convert("sglang", common_args)
        command.extend(["--model-path", converted_args["model_path"]])
        common_args_flatten = flatten_dict_to_args(converted_args, ["model"])
        command.extend(common_args_flatten)
        sglang_args_flatten = flatten_dict_to_args(sglang_args, ["model"])
        command.extend(sglang_args_flatten)
    else:
        raise ValueError("Either model must be specified in sglang model.")

    # set default args align with sglang.launch_server
    command.extend(["--node-rank", str(0)])
    nnodes = serve.task_config.experiment.runner.get("nnodes", 1)
    command.extend(["--nnodes", str(nnodes)])
    addr = str(serve.task_config.experiment.runner.get("master_addr", "127.0.0.1"))
    port = int(serve.task_config.experiment.runner.get("master_port", get_free_port()))
    # check ip address is available
    addr_str = addr + ":" + str(port)
    if check_port_occupied(addr, port):
        raise ValueError(f"addr: {addr_str} is invalid")
    command.extend(["--dist-init-addr", addr_str])

    # Start the subprocess
    logger.info(f"[Serve]: Starting sglang serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current Llama PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def main():
    serve.load_args()
    serve_config = serve.task_config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve.task_config}")

    model_config = None
    for item in serve_config:
        if item.get("serve_id", None) is not None:
            model_config = item
            break

    if model_config is None:
        raise ValueError(f"No valid configuration found in task config: {serve.task_config}")

    backend_value = serve.task_config.get("experiment", {}).get("task", {}).get("backend")
    if backend_value is None:
        engine = model_config.get("engine", None)
    else:
        engine = backend_value

    if engine == "vllm":
        return_code = vllm_serve(model_config)
    elif engine == "llama_cpp":
        return_code = llama_cpp_serve(model_config)
    elif engine == "sglang":
        if model_config.get("serve_id", None) is None:
            logger.warning("serve_id in yaml config should be specified in sglang model")
        return_code = sglang_serve(model_config)
    else:
        raise ValueError(
            f"Unsupported inference engine: {engine}, current config {serve.task_config}"
        )

    logger.info(f"[Serve]: {engine} serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
