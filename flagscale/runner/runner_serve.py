import asyncio
import collections
import contextlib
import copy
import json
import math
import os
import shlex
import signal
import socket
import subprocess

import psutil
from omegaconf import DictConfig, OmegaConf

from flagscale.runner.runner_base_legacy import JobStatus, RunnerBase
from flagscale.runner.utils import (
    ResourceManager,
    benchmark,
    dummy_random_input,
    flatten_dict_to_args,
    get_addr,
    get_free_port,
    get_ip_addr,
    get_nproc_per_node,
    is_ip_addr,
    is_master_node,
    logger,
    parse_hostfile,
    run_local_command,
    wait_for_ray_master,
)


def _get_multiple_free_ports(num=1, exclude_ports=[]):
    allocated_ports = []
    for i in range(num):
        port = get_free_port()
        while port in allocated_ports or port in exclude_ports:
            port = get_free_port()
        allocated_ports.append(port)
    return allocated_ports


def _get_args_vllm(config: DictConfig):
    # see the following link for more details
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_dict = OmegaConf.to_container(config, resolve=True)

    # step2: restructuring the config
    # config_dict = config_dict["serve"]
    config_dict["logging"].pop("log_dir")
    config_dict["logging"].pop("scripts_dir")
    config_dict["logging"].pop("pids_dir")
    if not config_dict.get("logging"):
        config_dict.pop("logging")

    # step3: dict -> yaml
    logging_config = config.logging
    new_config = OmegaConf.create(config_dict)
    new_conf_file = os.path.join(logging_config.scripts_dir, "serve.yaml")

    # step4: write the new yaml file to `outputs_dir/serve_logs/scripts/serve.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


def _reset_serve_port(config):
    model_port = None
    deploy_port = config.experiment.get("runner", {}).get("deploy", {}).get("port", None)
    cli_args_port = config.experiment.get("runner", {}).get("cli_args", {}).get("port", None)

    OmegaConf.set_struct(config, False)

    if cli_args_port:
        deploy_port = cli_args_port
        config.experiment.runner.deploy.port = cli_args_port

    for item in config.serve:
        if item.get("serve_id", None) is not None:
            if deploy_port:
                model_port = deploy_port
                item.engine_args["port"] = deploy_port
            else:
                model_port = item.engine_args.get("port", 8000)
            break
    OmegaConf.set_struct(config, True)
    if not model_port:
        logger.warning(f"No 'model_port' configuration found in task config: {config}")
    return model_port


def _get_inference_engine(config):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    if serve_config and len(serve_config) > 1:
        logger.warning(f"Multiple 'serve' configurations found in task config: {serve_config}")

    engine = serve_config[0].get("engine", None)
    return engine


def _get_engine_args(config, backend="vllm"):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    engine_args = {}

    for item in serve_config:
        if item.get("serve_id", None) is not None:
            engine_args = item.get("engine_args", {})
            break
    if not engine_args:
        raise ValueError(f"No 'engine_args' configuration found in task config: {serve_config}")

    return engine_args


def _get_profile_args(config, backend="vllm"):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")

    profile_args = {}
    for item in serve_config:
        if item.get("serve_id", None) is not None:
            profile_args = item.get("profile", {})
            break
    return profile_args


def _update_auto_engine_args(config, backend="vllm", new_engine_args={}):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    engine_args = {}
    for item in serve_config:
        if item.get("serve_id", None) is not None:
            engine_args = item.get("engine_args", {})

            if new_engine_args.get("tensor_parallel_size", None):
                tensor_parallel_size = int(new_engine_args.get("tensor_parallel_size"))
            else:
                nproc_per_node = int(config.experiment.runner.get("nproc_per_node", 1))
                tensor_parallel_size = 2 ** int(math.floor(math.log2(nproc_per_node)))
            if new_engine_args.get("pipeline_parallel_size", None):
                pipeline_parallel_size = int(new_engine_args.get("pipeline_parallel_size"))
            else:
                node_nums = int(config.experiment.runner.get("nnodes", 1))
                if node_nums <= 0:
                    raise ValueError(
                        f"Number of nodes (nnodes) must be a positive integer, but got {node_nums}."
                    )
                else:
                    pipeline_parallel_size = 2 ** int(math.floor(math.log2(node_nums)))
            new_engine_args["tensor_parallel_size"] = tensor_parallel_size
            new_engine_args["pipeline_parallel_size"] = pipeline_parallel_size
            engine_args.update(new_engine_args)
            item.engine_args = engine_args

            break
    if not engine_args:
        raise ValueError(f"No 'engine_args' configuration found in task config: {serve_config}")

    return engine_args


def _update_config_serve(config: DictConfig):
    deploy_config = config.experiment.get("runner", {}).get("deploy", {})

    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if deploy_config.get("prefill_decode_disaggregation", False) and config.action != "stop":
        deploy_config["pd_proxy_port"] = get_free_port()

    if config.get("logging", None) is None:
        config.logging = DictConfig({})

    cli_model_path = config.experiment.get("runner", {}).get("cli_args", {}).get("model_path", None)
    cli_engine_args_str = (
        config.experiment.get("runner", {}).get("cli_args", {}).get("engine_args", None)
    )
    cli_engine_args = json.loads(cli_engine_args_str) if cli_engine_args_str else {}

    if cli_model_path or cli_engine_args:
        for item in config.serve:
            if item.get("serve_id", None) is not None:
                if cli_model_path:
                    item.engine_args["model"] = cli_model_path
                if cli_engine_args:
                    item.engine_args.update(cli_engine_args)

    if config.experiment.runner.get("type", "ssh") == "cloud":
        # set auto tp and pp size
        _update_auto_engine_args(config, new_engine_args=cli_engine_args)

    log_dir = os.path.join(exp_dir, "serve_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.logging.log_dir = log_dir
    config.logging.scripts_dir = scripts_dir
    config.logging.pids_dir = pids_dir

    os.makedirs(config.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


def match_address(address):
    """Check if current node is matched."""

    if is_ip_addr(address):
        return get_ip_addr() == address
    else:
        hostname = socket.gethostname()
        return hostname == address


def parse_cloud_hostfile(hostfile_path):
    if hostfile_path is None or not os.path.isfile(hostfile_path):
        logger.warning(
            f"Hostfile {hostfile_path} not found. The task will proceed using only local resources."
        )
        return None

    resources = collections.OrderedDict()

    with open(hostfile_path, "r") as fd:
        hostfile_lines = fd.readlines()

    for line in hostfile_lines:
        line = line.strip()
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        else:
            host = line
            num_slots = int(os.getenv("AIRS_ACCELERATOR_NUM", "1"))
            machine_type = "gpu"
            resources[host] = {"slots": num_slots, "type": machine_type}

    assert all(info["type"] is None for _, info in resources.items()) or all(
        info["type"] is not None for _, info in resources.items()
    ), "All hosts must have the a machine type or no machine type specified."

    if len(resources) == 0:
        raise ValueError("Hostfile is empty or not formatted correctly. Please check the hostfile.")

    return resources


def _generate_run_script_serve(config, host, node_rank, cmd, background=True, with_test=False):
    nodes = config.get("nodes", None)
    logging_config = config.logging

    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    if no_shared_fs:
        host_output_file = os.path.join(logging_config.log_dir, "host.output")
    else:
        host_output_file = os.path.join(logging_config.log_dir, f"host_{node_rank}_{host}.output")
    host_run_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
    )
    host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmds_config = config.experiment.get("cmds", None)
    ssh_port = config.experiment.runner.get("ssh_port", 22)
    docker_name = config.experiment.runner.get("docker", None)
    if cmds_config:
        before_start_cmd = cmds_config.get("before_start", "")
    else:
        before_start_cmd = ""
    cmd += f" --log-dir={logging_config.log_dir}"
    try:
        import vllm

        vllm_path = os.path.dirname(vllm.__path__[0])
    except Exception:
        vllm_path = f"{root_dir}/vllm"
    deploy_config = config.experiment.get("runner", {}).get("deploy", {})
    envs = config.experiment.get("envs", {})
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write("\n")
        f.write(f"{before_start_cmd}\n")
        f.write("\n")

        f.write('if [ -z "$PYTHONPATH" ]; then\n')
        f.write(f"    export PYTHONPATH={vllm_path}:{root_dir}\n")
        f.write("else\n")
        f.write(f'    export PYTHONPATH="$PYTHONPATH:{vllm_path}:{root_dir}"\n')
        f.write("fi\n")
        f.write("\n")
        envs_str = " && ".join(
            f"export {key}={value}" for key, value in envs.items() if key != "nodes_envs"
        )
        f.write(f"{envs_str}\n")
        use_vllm_v1 = (str(os.getenv("VLLM_USE_V1", "true")).lower() in ("1", "true")) and (
            str(envs.get("VLLM_USE_V1", "true")).lower() in ("1", "true")
        )

        if nodes:
            if deploy_config.get("prefill_decode_disaggregation", False):
                resource_manager = ResourceManager(nodes)
                master_ip = nodes[0][0]
                target_port = nodes[0][1].get("port")
                p_num = deploy_config.get("prefill_num", 1)
                d_num = deploy_config.get("decode_num", 1)
                ports_num = (p_num + d_num) * 2
                kv_related_ports = _get_multiple_free_ports(ports_num)
                pd_proxy_port = deploy_config.get("pd_proxy_port", None)
                if not pd_proxy_port:
                    raise ValueError("PD disaggregation requires a proxy port to be set.")

                engine_args = _get_engine_args(config)
                command_items = ["vllm", "serve"]
                command_items.append(engine_args["model"])
                other_args = flatten_dict_to_args(engine_args, ["model", "port"])
                command_items.extend(other_args)
                vllm_command = " ".join(command_items)
                if before_start_cmd:
                    vllm_command = f"{before_start_cmd} && " + vllm_command
                if envs_str:
                    vllm_command = f"{envs_str} && " + vllm_command
                p_address = deploy_config.get("prefill_address", "auto")
                d_address = deploy_config.get("decode_address", "auto")
                tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
                pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
                each_instance_card_num = tensor_parallel_size * pipeline_parallel_size
                default_log_dir = deploy_config.get(
                    "prefill_decode_log_dir", logging_config.log_dir
                )

                f.write("# clean nodes \n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        if not node.get("type", None):
                            raise ValueError(
                                f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                            )
                        if not node.get("slots", None):
                            raise ValueError(
                                f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                            )
                        node_cmd = f"mkdir -p {default_log_dir} && pkill -f vllm"

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'

                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")

                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write("pkill -f 'run_disagg_xpyd_router'\n")
                f.write(f"mkdir -p {default_log_dir}\n")
                f.write("\n")

                f.write("echo '=========== launch prefill instance ==========='\n")

                for i in range(p_num):
                    kv_port = kv_related_ports.pop()
                    http_port = kv_related_ports.pop()
                    if use_vllm_v1:
                        p_kv_config = {
                            "kv_connector": "P2pNcclConnector",
                            "kv_role": "kv_producer",
                            "kv_port": str(kv_port),
                            "kv_buffer_size": "1e1",
                            "kv_connector_extra_config": {
                                "proxy_ip": master_ip,
                                "proxy_port": str(pd_proxy_port),
                                "http_port": str(http_port),
                            },
                        }

                    else:
                        p_kv_config = {
                            "kv_connector": "P2pConnector",
                            "kv_role": "kv_producer",
                            "kv_port": str(kv_port),
                            "kv_connector_extra_config": {
                                "proxy_ip": master_ip,
                                "proxy_port": str(pd_proxy_port),
                                "http_port": str(http_port),
                            },
                        }
                    logger.info(
                        f"============= prefill instance {i}, p_kv_config: {p_kv_config} ============="
                    )
                    card_ids, update_p_address = resource_manager.get_available_card_ids(
                        address=p_address, num=each_instance_card_num
                    )
                    card_ids_str = ",".join(map(str, card_ids))
                    ids_env = f"export CUDA_VISIBLE_DEVICES={card_ids_str}"

                    p_kv_config_json = json.dumps(p_kv_config)
                    p_instance_log_path = os.path.join(default_log_dir, f"prefill_{i}.log")

                    if update_p_address != master_ip and len(nodes) > 1:
                        p_kv_config_format_json = p_kv_config_json.replace('"', '\\"')
                        node_cmd = f"{ids_env} && {vllm_command} --port {http_port} --kv-transfer-config '\\''{p_kv_config_format_json}'\\''"
                        if docker_name:
                            ssh_cmd = f"ssh -f -n -p {ssh_port} {update_p_address} \"docker exec {docker_name} /bin/bash -c '{node_cmd} > {p_instance_log_path} 2>&1 &'\""
                        else:
                            ssh_cmd = f'ssh -f -n -p {ssh_port} {update_p_address} "{node_cmd} > {p_instance_log_path} 2>&1 &"'
                        f.write(f"{ssh_cmd}\n\n")
                    else:
                        p_cmd = f"{ids_env} && {vllm_command} --port {http_port} --kv-transfer-config '\\''{p_kv_config_json}'\\''"
                        f.write(f"p_{i}_cmd='{p_cmd}'\n")
                        f.write("\n")
                        f.write(
                            f'nohup bash -c "$p_{i}_cmd; sync" >> {p_instance_log_path} 2>&1 &\n\n'
                        )

                f.write("echo '=========== launch decode instance ==========='\n")
                decode_gpu_memory_utilization = deploy_config.get(
                    "decode_gpu_memory_utilization", 0.7
                )

                for j in range(d_num):
                    kv_port = kv_related_ports.pop()
                    http_port = kv_related_ports.pop()
                    if use_vllm_v1:
                        d_kv_config = {
                            "kv_connector": "P2pNcclConnector",
                            "kv_role": "kv_consumer",
                            "kv_port": str(kv_port),
                            "kv_buffer_size": "8e9",
                            "kv_connector_extra_config": {
                                "proxy_ip": master_ip,
                                "proxy_port": str(pd_proxy_port),
                                "http_port": str(http_port),
                            },
                        }
                    else:
                        d_kv_config = {
                            "kv_connector": "P2pConnector",
                            "kv_role": "kv_consumer",
                            "kv_port": str(kv_port),
                            "kv_connector_extra_config": {
                                "proxy_ip": master_ip,
                                "proxy_port": str(pd_proxy_port),
                                "http_port": str(http_port),
                            },
                        }
                    logger.info(
                        f"============= decode instance {j}, d_kv_config: {d_kv_config} ============="
                    )
                    card_ids, update_d_address = resource_manager.get_available_card_ids(
                        address=d_address, num=each_instance_card_num
                    )
                    card_ids_str = ",".join(map(str, card_ids))
                    ids_env = f"export CUDA_VISIBLE_DEVICES={card_ids_str}"

                    d_kv_config_json = json.dumps(d_kv_config)
                    d_instance_log_path = os.path.join(default_log_dir, f"decode_{j}.log")

                    if update_d_address != master_ip and len(nodes) > 1:
                        d_kv_config_format_json = d_kv_config_json.replace('"', '\\"')
                        node_cmd = f"{ids_env} && {vllm_command} --port {http_port} --gpu-memory-utilization {decode_gpu_memory_utilization} --kv-transfer-config '\\''{d_kv_config_format_json}'\\''"
                        if docker_name:
                            ssh_cmd = f"ssh -f -n -p {ssh_port} {update_d_address} \"docker exec {docker_name} /bin/bash -c '{node_cmd} > {d_instance_log_path} 2>&1 &'\""
                        else:
                            ssh_cmd = f'ssh -f -n -p {ssh_port} {update_d_address} "{node_cmd} > {d_instance_log_path} 2>&1 &"'
                        f.write(f"{ssh_cmd}\n\n")
                    else:
                        d_cmd = f"{ids_env} && {vllm_command} --port {http_port} --gpu-memory-utilization {decode_gpu_memory_utilization} --kv-transfer-config '\\''{d_kv_config_json}'\\''"
                        f.write(f"d_{j}_cmd='{d_cmd}'\n")
                        f.write("\n")
                        f.write(
                            f'nohup bash -c "$d_{j}_cmd; sync" >> {d_instance_log_path} 2>&1 &\n\n'
                        )

            else:
                engine = _get_inference_engine(config)

                f.write("ray_path=$(realpath $(which ray))\n")
                master_ip = nodes[0][0]
                target_port = nodes[0][1].get("port")

                f.write("# clean nodes \n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        if not node.get("type", None):
                            raise ValueError(
                                f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                            )
                        if not node.get("slots", None):
                            raise ValueError(
                                f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                            )
                        node_cmd = "${ray_path} stop"

                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        if envs_str:
                            node_cmd = f"{envs_str} && " + node_cmd

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'

                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")
                if before_start_cmd:
                    f.write(f"{before_start_cmd} && ${{ray_path}} stop\n")
                else:
                    f.write("${ray_path} stop\n")
                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write("\n")

                master_port = target_port if target_port else get_free_port()

                address = f"{master_ip}:{master_port}"
                nodes_envs = config.experiment.get("envs", {}).get("nodes_envs", {})
                node_args = config.experiment.get("node_args", {})
                for index, (ip, node) in enumerate(nodes):
                    per_node_cmd = None
                    if nodes_envs and nodes_envs.get(ip, None) is not None:
                        per_node_cmd = " && ".join(
                            f"export {key}={value}" for key, value in nodes_envs[ip].items()
                        )
                    if not node.get("type", None):
                        raise ValueError(
                            f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                        )
                    if not node.get("slots", None):
                        raise ValueError(
                            f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                        )

                    if engine == "sglang":
                        from flagscale.serve.args_mapping.mapping import ARGS_CONVERTER

                        if index == 0:
                            if per_node_cmd:
                                f.write(f"{per_node_cmd}\n")
                        if index != 0:
                            logger.info(f"generate run script args, config: {config}")
                            args = None
                            for item in config.get("serve", []):
                                if item.get("serve_id", None) is not None:
                                    args = item
                                    break
                            if args is None:
                                raise ValueError(
                                    f"No sglang model configuration found in task config: {config}"
                                )
                            common_args = copy.deepcopy(args.get("engine_args", {}))
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

                            command = ["nohup", "python", "-m", "sglang.launch_server"]
                            if common_args.get("model", None):
                                # if node specific args
                                if (
                                    node_args.get(ip, None) is not None
                                    and node_args[ip].get("engine_args", None) is not None
                                ):
                                    for key, value in node_args[ip]["engine_args"].items():
                                        common_args[key] = value
                                        logger.info(
                                            f"node_args[{ip}] overwrite engine_args {key} = {value}"
                                        )

                                converted_args = ARGS_CONVERTER.convert("sglang", common_args)
                                common_args_flatten = flatten_dict_to_args(
                                    converted_args, ["model"]
                                )
                                command.extend(common_args_flatten)
                                sglang_args_flatten = flatten_dict_to_args(sglang_args, ["model"])
                                command.extend(sglang_args_flatten)
                            else:
                                raise ValueError(
                                    "Either model should be specified in sglang model."
                                )

                            command.extend(["--node-rank", str(index)])
                            nnodes = config.experiment.runner.get("nnodes", None)
                            addr = config.experiment.runner.get("master_addr", None)
                            port = config.experiment.runner.get("master_port", None)
                            if nnodes is None or addr is None or port is None:
                                raise ValueError(
                                    "nnodes, master_addr, master_port must be specified in runner when engine is sglang with multi-nodes mode."
                                )
                            command.extend(["--nnodes", str(nnodes)])
                            command.extend(["--dist-init-addr", str(addr) + ":" + str(port)])
                            command.append("> /dev/null 2>&1 &")

                            if docker_name:
                                node_cmd = " ".join(command)
                            else:
                                # Directly connecting to a remote Docker environment requires processing the command
                                command.insert(0, "(")
                                command.append(") && disown")
                                node_cmd = " ".join(command)
                            if per_node_cmd:
                                node_cmd = f"{per_node_cmd} && " + node_cmd
                            if before_start_cmd:
                                node_cmd = f"{before_start_cmd} && " + node_cmd
                            if envs_str:
                                node_cmd = f"{envs_str} && " + node_cmd
                            ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                            if docker_name:
                                ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                            f.write(f"{ssh_cmd}\n")
                        continue

                    # if engine == vllm
                    if index == 0:
                        # master node
                        f.write("# start cluster\n")
                        f.write("# master node\n")
                        if node.type == "gpu":
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --num-gpus={node.slots}"
                        elif node.type == "cpu":
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --num-cpus={node.slots}"
                        else:
                            resource = json.dumps({node.type: node.slots}).replace('"', '"')
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --resources='{resource}'"
                        if per_node_cmd:
                            node_cmd = f"{per_node_cmd} && " + node_cmd
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        f.write(f"{node_cmd}\n")
                    else:
                        # worker nodes
                        if index == 1:
                            f.write("\n")
                            f.write("# worker nodes\n")
                        if node.type == "gpu":
                            node_cmd = (
                                f"${{ray_path}} start --address={address} --num-gpus={node.slots}"
                            )

                        elif node.type == "cpu":
                            node_cmd = (
                                f"${{ray_path}} start --address={address} --num-cpus={node.slots}"
                            )
                        else:
                            resource = json.dumps({node.type: node.slots}).replace('"', '\\"')
                            node_cmd = (
                                f"${{ray_path}} start --address={address} --resources='{resource}'"
                            )
                        if per_node_cmd:
                            node_cmd = f"{per_node_cmd} && " + node_cmd
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        if envs_str:
                            node_cmd = f"{envs_str} && " + node_cmd
                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")
        else:
            # Note: config key device_type is specified for single node serving in neither gpu or cpu.
            device_type = None
            nproc_per_node = None
            if config.experiment.get("runner", None) and config.experiment.runner.get(
                "device_type", None
            ):
                device_type = config.experiment.runner.get("device_type", None)
                nproc_per_node = config.experiment.runner.get("nproc_per_node", None)
                if nproc_per_node is None:
                    raise ValueError(
                        f"nproc_per_node must be specified when device_type {device_type} is specified."
                    )
            node_cmd = None

            if deploy_config.get("use_fs_serve", True) and config.serve[0].get("engine", None):
                f.write("ray_path=$(realpath $(which ray))\n")
                if not device_type:
                    node_cmd = "${ray_path} start --head"
                elif device_type == "gpu":
                    node_cmd = f"${{ray_path}} start --head --num-gpus={nproc_per_node}"
                elif device_type == "cpu":
                    node_cmd = f"${{ray_path}} start --head --num-cpus={nproc_per_node}"
                else:
                    resource = json.dumps({device_type: nproc_per_node}).replace('"', '\\"')
                    node_cmd = f"${{ray_path}} start --head --resources='{resource}'"
            if before_start_cmd:
                node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
            if node_cmd:
                f.write(f"{node_cmd}\n")

        f.write(f"mkdir -p {logging_config.log_dir}\n")
        f.write(f"mkdir -p {logging_config.pids_dir}\n")
        f.write("\n")
        f.write(f"cd {root_dir}\n")
        f.write("\n")
        f.write(f'cmd="{cmd}"\n')
        f.write("\n")
        # TODO: need a option to control whether to append or overwrite the output file
        # Now, it always appends to the output file
        f.write("echo '=========== launch task ==========='\n")
        if background:
            f.write(
                f'nohup bash -c "$cmd; sync" >> {host_output_file} 2>&1 & echo $! > {host_pid_file}\n'
            )
        else:
            f.write(f'bash -c "$cmd; sync" >> {host_output_file} 2>&1\n')
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_run_script_file, 0o755)

    return host_run_script_file


def _generate_cloud_run_script_serve(
    config, host, node_rank, cmd, background=True, with_test=False
):
    logging_config = config.logging
    node_id = get_addr()
    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    if no_shared_fs:
        host_output_file = os.path.join(logging_config.log_dir, "host.output")
    else:
        host_output_file = os.path.join(logging_config.log_dir, f"host_{node_rank}_{host}.output")
    host_run_script_file = os.path.join(
        logging_config.scripts_dir, node_id, f"host_{node_rank}_{host}_run.sh"
    )
    host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

    os.makedirs(logging_config.scripts_dir, exist_ok=True)
    if node_id:
        os.makedirs(os.path.join(logging_config.scripts_dir, node_id), exist_ok=True)

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        before_start_cmd = cmds_config.get("before_start", "")
    else:
        before_start_cmd = ""
    cmd += f" --log-dir={logging_config.log_dir}"
    try:
        import vllm

        vllm_path = os.path.dirname(vllm.__path__[0])
    except Exception:
        vllm_path = f"{root_dir}/vllm"
    deploy_config = config.experiment.get("runner", {}).get("deploy", {})
    envs = config.experiment.get("envs", {})
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write("\n")
        f.write(f"{before_start_cmd}\n")
        f.write("\n")

        f.write('if [ -z "$PYTHONPATH" ]; then\n')
        f.write(f"    export PYTHONPATH={vllm_path}:{root_dir}\n")
        f.write("else\n")
        f.write(f'    export PYTHONPATH="$PYTHONPATH:{vllm_path}:{root_dir}"\n')
        f.write("fi\n")
        f.write("\n")
        envs_str = " && ".join(f"export {key}={value}" for key, value in envs.items())
        f.write(f"{envs_str}\n")

        if node_id:
            f.write("ray_path=$(realpath $(which ray))\n")
            master_name_or_addr = config.experiment.runner.get("master_addr")
            master_port = int(config.experiment.runner.get("master_port"))

            current_node_is_master = False
            master_addr = master_name_or_addr
            if is_ip_addr(master_name_or_addr):
                current_node_is_master = match_address(master_addr)
            else:
                current_node_is_master = is_master_node(master_name_or_addr)

            address = f"{master_addr}:{master_port}"

            node = {
                "type": config.experiment.runner.get("device_type", "gpu"),
                "slots": int(os.getenv("AIRS_ACCELERATOR_NUM", "1")),
            }
            node = OmegaConf.create(node)

            if not node.get("type", None):
                raise ValueError(
                    f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                )
            if not node.get("slots", None):
                raise ValueError(
                    f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                )

            if current_node_is_master:
                # master node
                f.write("# start cluster\n")
                f.write("# master node\n")
                if node.type == "gpu":
                    node_cmd = (
                        f"${{ray_path}} start --head --port={master_port} --num-gpus={node.slots}"
                    )
                elif node.type == "cpu":
                    node_cmd = (
                        f"${{ray_path}} start --head --port={master_port} --num-cpus={node.slots}"
                    )
                else:
                    resource = json.dumps({node.type: node.slots}).replace('"', '"')
                    node_cmd = (
                        f"${{ray_path}} start --head --port={master_port} --resources='{resource}'"
                    )
                if before_start_cmd:
                    node_cmd = f"{before_start_cmd} && " + node_cmd
                f.write(f"{node_cmd}\n")
            else:
                # worker nodes
                f.write("\n")
                f.write("# worker nodes\n")
                if wait_for_ray_master(master_addr, master_port):
                    if node.type == "gpu":
                        node_cmd = (
                            f"${{ray_path}} start --address={address} --num-gpus={node.slots}"
                        )

                    elif node.type == "cpu":
                        node_cmd = (
                            f"${{ray_path}} start --address={address} --num-cpus={node.slots}"
                        )
                    else:
                        resource = json.dumps({node.type: node.slots}).replace('"', '"')
                        node_cmd = (
                            f"${{ray_path}} start --address={address} --resources='{resource}'"
                        )
                    if before_start_cmd:
                        node_cmd = f"{before_start_cmd} && " + node_cmd
                    f.write(f"{node_cmd}\n")
                else:
                    raise ValueError("The current node can not connect to master node")

        else:
            # Note: config key device_type is specified for single node serving in neither gpu or cpu.
            device_type = None
            nproc_per_node = None
            if config.experiment.get("runner", None) and config.experiment.runner.get(
                "device_type", None
            ):
                device_type = config.experiment.runner.get("device_type", None)
                nproc_per_node = config.experiment.runner.get("nproc_per_node", None)
                if nproc_per_node is None:
                    raise ValueError(
                        f"nproc_per_node must be specified when device_type {device_type} is specified."
                    )
            node_cmd = None

            if deploy_config.get("use_fs_serve", True) and config.serve[0].get("engine", None):
                f.write("ray_path=$(realpath $(which ray))\n")
                if not device_type:
                    node_cmd = "${ray_path} start --head"
                elif device_type == "gpu":
                    node_cmd = f"${{ray_path}} start --head --num-gpus={nproc_per_node}"
                elif device_type == "cpu":
                    node_cmd = f"${{ray_path}} start --head --num-cpus={nproc_per_node}"
                else:
                    resource = json.dumps({device_type: nproc_per_node}).replace('"', '"')
                    node_cmd = f"${{ray_path}} start --head --resources='{resource}'"
            if before_start_cmd:
                node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
            if node_cmd:
                f.write(f"{node_cmd}\n")

        if not node_id or current_node_is_master:
            f.write(f"mkdir -p {logging_config.log_dir}\n")
            f.write(f"mkdir -p {logging_config.pids_dir}\n")
            f.write("\n")
            f.write(f"cd {root_dir}\n")
            f.write("\n")
            f.write(f'cmd="{cmd}"\n')
            f.write("\n")
            # TODO: need a option to control whether to append or overwrite the output file
            # Now, it always appends to the output file
            f.write("echo '=========== launch task ==========='\n")
            if background:
                f.write(
                    f'nohup bash -c "$cmd; sync" >> {host_output_file} 2>&1 & echo $! > {host_pid_file}\n'
                )
            else:
                f.write(f'bash -c "$cmd; sync" >> {host_output_file} 2>&1\n')
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_run_script_file, 0o755)

    return host_run_script_file


def _generate_stop_script(config, host, node_rank):
    logging_config = config.logging

    host_stop_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
    )

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        after_stop = cmds_config.get("after_stop", "")
    else:
        after_stop = ""

    nodes = config.get("nodes", None)

    cmds_config = config.experiment.get("cmds", None)
    ssh_port = config.experiment.runner.get("ssh_port", 22)
    docker_name = config.experiment.runner.get("docker", None)
    if cmds_config:
        before_start_cmd = cmds_config.get("before_start", "")
    else:
        before_start_cmd = ""

    deploy_config = config.experiment.get("runner", {}).get("deploy", {})
    envs = config.experiment.get("envs", {})
    with open(host_stop_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write("\n")
        f.write(f"{before_start_cmd}\n")
        f.write("\n")
        envs_str = " && ".join(f"export {key}={value}" for key, value in envs.items())
        f.write(f"{envs_str}\n")

        if nodes:
            if deploy_config.get("prefill_decode_disaggregation", False):
                f.write("# clean nodes \n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        node_cmd = "pkill -f vllm && pkill -f python"
                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")

                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write("pkill -f 'run_disagg_xpyd_router'\n")
                f.write("\n")

            else:
                f.write("ray_path=$(realpath $(which ray))\n")
                f.write("# clean nodes \n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        node_cmd = "${ray_path} stop && pkill -f python"
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        if envs_str:
                            node_cmd = f"{envs_str} && " + node_cmd

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'

                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")
                if before_start_cmd:
                    f.write(f"{before_start_cmd} && ${{ray_path}} stop\n")
                else:
                    f.write("${ray_path} stop\n")
                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write("pkill -f multiprocessing\n")
                f.write("\n")
        else:
            node_cmd = None
            if deploy_config.get("use_fs_serve", True) and config.serve[0].get("engine", None):
                f.write("ray_path=$(realpath $(which ray))\n")
                node_cmd = "${ray_path} stop"
            if before_start_cmd:
                node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
            if node_cmd:
                f.write(f"{node_cmd}\n")
            f.write("pkill -f 'run_inference_engine'\n")
            f.write("pkill -f 'run_fs_serve_vllm'\n")
            f.write("pkill -f 'vllm serve'\n")
            f.write("pkill -f multiprocessing\n")
            f.write("\n")
        f.write(f"{after_stop}\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_stop_script_file, 0o755)

    return host_stop_script_file


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


class SSHServeRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self.deploy_config = self.config.experiment.get("runner", {}).get("deploy", {})
        if not self.config.experiment.task.get("entrypoint", None):
            self.inference_engine = _get_inference_engine(self.config)
            self.port = _reset_serve_port(config)
        else:
            self.inference_engine = None
            self.port = None
        self.use_fs_serve = self.deploy_config.get("use_fs_serve", True)

        self._prepare()
        self.host = None

    def _prepare(self):
        _update_config_serve(self.config)
        self.user_args = _get_args_vllm(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        entrypoint = self.config.experiment.task.get("entrypoint", None)
        if self.inference_engine:
            if (
                self.config.experiment.get("runner", {})
                .get("deploy", {})
                .get("prefill_decode_disaggregation", False)
            ):
                self.user_script = "flagscale/serve/run_disagg_xpyd_router.py"
            elif not self.use_fs_serve:
                self.user_script = "flagscale/serve/run_inference_engine.py"
            else:
                self.user_script = "flagscale/serve/run_fs_serve_vllm.py"
        elif isinstance(entrypoint, str) and entrypoint.endswith(".py"):
            self.user_script = entrypoint
        elif entrypoint is None:
            self.user_script = "flagscale/serve/run_serve.py"
        else:
            raise ValueError(
                f"Invalid config entrypoint: {entrypoint}, must be a python file path or null."
            )
        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        if hostfile_path:
            if os.path.isabs(hostfile_path):
                hostfile_path = hostfile_path
            else:
                hostfile_path = os.path.join(os.getcwd(), hostfile_path)
            if not os.path.exists(hostfile_path):
                raise ValueError(f"The hostfile {hostfile_path} does not exist")
        self.resources = None
        if hostfile_path:
            self.resources = parse_hostfile(hostfile_path)
            for key, value in self.resources.items():
                if not value.get("type", None):
                    logger.warning(
                        f"The hostfile key type is not set for host {key}, using gpu by default"
                    )
                    self.resources[key]["type"] = "gpu"
            if self.resources:
                OmegaConf.set_struct(self.config, False)
                self.config["nodes"] = list(self.resources.items())
                OmegaConf.set_struct(self.config, True)

        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        with_test=False,
        dryrun=False,
    ):
        export_cmd = []
        for k, v in self.user_envs.items():
            if k != "nodes_envs":
                export_cmd += [f"{k}={v}"]

        cmd = shlex.join([*export_cmd, "python", self.user_script, *self.user_args])

        host_run_script_file = _generate_run_script_serve(
            self.config, host, node_rank, cmd, background=True, with_test=with_test
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, with_test=False, dryrun=False):
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

        # If hostfile is not provided, run the job on localhost
        nproc_from_args = runner_config.get("nproc_per_node", None)
        nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
        available_addr = runner_config.get("master_addr", "localhost")
        available_port = runner_config.get("master_port", get_free_port())
        self._run_each(
            "localhost",
            available_addr,
            available_port,
            1,
            0,
            nproc_per_node,
            with_test=with_test,
            dryrun=dryrun,
        )
        self.host = available_addr

    def _stop_each(self, host, node_rank):
        logging_config = self.config.logging
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")
        with open(host_pid_file, "r") as f:
            pid = f.readlines()[0]
            pid = int(pid.strip())
        kill_process_tree(pid)

        host_stop_script_file = _generate_stop_script(self.config, host, node_rank)
        logging_config = self.config.logging
        cmd = f"bash {host_stop_script_file}"
        logger.info(f"Run the local command: {cmd}")
        subprocess.run(
            cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

    def stop(self):
        self._stop_each("localhost", 0)
        return

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.logging

        host_query_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")
        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to query the job.
            f.write(
                "    pid=$(ps aux | grep -E 'run_fs_serve_vllm|run_inference_engine' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file

    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        result = ""
        try:
            result = run_local_command(f"bash {host_query_script_file}", query=True)
        except Exception as e:
            logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_status(self):
        "Query Job status."
        results = []
        result = self._query_each("localhost", 0)
        results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status

    def _serve_alive(self):
        engine_args = _get_engine_args(self.config)
        model_name = engine_args.get("served_model_name", None) or engine_args.get("model", None)

        if not model_name:
            raise ValueError("No model specified in config file.")

        from openai import OpenAI

        # Modify OpenAI's API key and API base to use vLLM's API server.
        api_key = "EMPTY"
        api_url = f"http://{self.host}:{self.port}/v1"
        logger.info(f"Testing API {api_url}")

        try:
            client = OpenAI(api_key=api_key, base_url=api_url)
            messages = [{"role": "user", "content": "who are you?"}]
            client.chat.completions.create(model=model_name, messages=messages)
        except Exception:
            # logger.info(f"API {api_url} is not ready, please wait a moment")
            return False

        return True

    def _profile_serve(self):
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer_mode = "auto"
        engine_args = _get_engine_args(self.config)

        trust_remote_code = engine_args.get("trust_remote_code", False)

        served_model_name = engine_args.get("served_model_name", None)
        model_name = engine_args.get("model", None)

        if not model_name:
            raise ValueError("No model specified in config file.")

        tokenizer = get_tokenizer(
            model_name, tokenizer_mode=tokenizer_mode, trust_remote_code=trust_remote_code
        )

        profile_args = _get_profile_args(self.config)
        prefix_len = profile_args.get("prefix_len", 0)
        input_len = profile_args.get("input_len", 1024)
        output_len = profile_args.get("output_len", 1024)
        num_prompts = profile_args.get("num_prompts", 200)
        range_ratio = profile_args.get("range_ratio", 0.5)
        dummy_input_requests = dummy_random_input(
            tokenizer=tokenizer,
            prefix_len=prefix_len,
            input_len=input_len,
            output_len=output_len,
            num_prompts=num_prompts,
            range_ratio=range_ratio,
        )
        api_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        logger.info(f"Profiling API {api_url}")

        ### allow metric = [\"ttft\", \"tpot\", \"itl\", \"e2el\"]
        ### allow percentiles = [\"25,50,75\"]
        result = asyncio.run(
            benchmark(
                api_url,
                model=model_name,
                served_model_name=served_model_name,
                tokenizer=tokenizer,
                input_requests=dummy_input_requests,
                selected_percentile_metrics="ttft,tpot,itl,e2el".split(","),
                selected_percentiles=[float(99)],
            )
        )
        return result


class CloudServeRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self.deploy_config = self.config.experiment.get("runner", {}).get("deploy", {})
        if not self.config.experiment.task.get("entrypoint", None):
            self.inference_engine = _get_inference_engine(self.config)
            self.port = _reset_serve_port(config)
        else:
            self.inference_engine = None
            self.port = None
        self.use_fs_serve = self.deploy_config.get("use_fs_serve", True)

        self._prepare()
        self.host = None

    def _prepare(self):
        self.resources = None
        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        if hostfile_path:
            if os.path.isabs(hostfile_path):
                hostfile_path = hostfile_path
            else:
                hostfile_path = os.path.join(os.getcwd(), hostfile_path)
            if not os.path.exists(hostfile_path):
                raise ValueError(f"The hostfile {hostfile_path} does not exist")

        if hostfile_path:
            self.resources = parse_cloud_hostfile(hostfile_path)
            for key, value in self.resources.items():
                if not value.get("type", None):
                    logger.warning(
                        f"The hostfile key type is not set for host {key}, using gpu by default"
                    )
                    self.resources[key]["type"] = "gpu"
            if self.resources:
                OmegaConf.set_struct(self.config, False)
                self.config["nodes"] = list(self.resources.items())
                OmegaConf.set_struct(self.config, True)

        _update_config_serve(self.config)
        self.user_args = _get_args_vllm(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        entrypoint = self.config.experiment.task.get("entrypoint", None)
        if self.inference_engine:
            if (
                self.config.experiment.get("runner", {})
                .get("deploy", {})
                .get("prefill_decode_disaggregation", False)
            ):
                self.user_script = "flagscale/serve/run_disagg_xpyd_router.py"
            elif not self.use_fs_serve:
                self.user_script = "flagscale/serve/run_inference_engine.py"
            else:
                self.user_script = "flagscale/serve/run_fs_serve_vllm.py"
        elif isinstance(entrypoint, str) and entrypoint.endswith(".py"):
            self.user_script = entrypoint
        elif self.use_fs_serve and self.deploy_config.get("enable_composition", False):
            self.user_script = "flagscale/serve/run_serve.py"
        else:
            raise ValueError(
                f"Invalid config entrypoint: {entrypoint}, must be a python file path or null."
            )
        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        with_test=False,
        dryrun=False,
    ):
        export_cmd = []
        for k, v in self.user_envs.items():
            export_cmd += [f"{k}={v}"]

        cmd = shlex.join([*export_cmd, "python", self.user_script, *self.user_args])

        host_run_script_file = _generate_cloud_run_script_serve(
            self.config, host, node_rank, cmd, background=True, with_test=with_test
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, with_test=False, dryrun=False):
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

        # If hostfile is not provided, run the job on localhost
        nproc_from_args = runner_config.get("nproc_per_node", None)
        nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
        available_addr = runner_config.get("master_addr", "localhost")
        available_port = runner_config.get("master_port", get_free_port())
        self._run_each(
            "localhost",
            available_addr,
            available_port,
            1,
            0,
            nproc_per_node,
            with_test=with_test,
            dryrun=dryrun,
        )
        self.host = available_addr
