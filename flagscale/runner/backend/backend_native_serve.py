import json
import os

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend.backend_base import BackendBase
from flagscale.runner.utils import get_free_port, logger, parse_hostfile


def _get_args_ray(config: DictConfig):
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
                if item.get("engine_args", None) is not None:
                    item.engine_args["port"] = deploy_port
            else:
                model_port = item.engine_args.get("port", 8000)
            break
    OmegaConf.set_struct(config, True)
    if not model_port:
        logger.warning(f"No 'model_port' configuration found in task config: {config}")
    return model_port


def _update_config_serve(config: DictConfig):
    _reset_serve_port(config)

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

    log_dir = os.path.join(exp_dir, "serve_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.logging.log_dir = log_dir
    config.logging.scripts_dir = scripts_dir
    config.logging.pids_dir = pids_dir

    os.makedirs(config.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


class NativeServeBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self.deploy_config = self.config.experiment.get("runner", {}).get("deploy", {})
        self.use_fs_serve = self.deploy_config.get("use_fs_serve", True)
        self._prepare()

    def _prepare(self):
        _update_config_serve(self.config)
        self.user_args = _get_args_ray(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        entrypoint = self.config.experiment.task.get("entrypoint", None)
        enable_composition = self.config.experiment.runner.deploy.get("enable_composition", False)

        if entrypoint:
            self.user_script = entrypoint
        elif self.use_fs_serve and not enable_composition:
            self.user_script = "flagscale/serve/run_fs_serve_vllm.py"
        else:
            self.user_script = "flagscale/serve/run_serve.py"

        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        self.resources = None
        if hostfile_path:
            if not os.path.isabs(hostfile_path):
                hostfile_path = os.path.join(os.getcwd(), hostfile_path)
            if os.path.exists(hostfile_path):
                self.resources = parse_hostfile(hostfile_path)
                for key, value in self.resources.items():
                    if not value.get("type", None):
                        logger.warning(
                            f"The hostfile key type is not set for host {key}, using gpu by default"
                        )
                        self.resources[key]["type"] = "gpu"

                OmegaConf.set_struct(self.config, False)
                self.config["nodes"] = list(self.resources.items())
                OmegaConf.set_struct(self.config, True)
            else:
                raise ValueError(f"The hostfile {hostfile_path} does not exist")

        logger.info("\n************** Ray Configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def generate_run_script(self, config, host, node_rank, cmd, background=True, with_test=False):
        nodes = config.get("nodes", None)
        logging_config = config.logging

        no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
        if no_shared_fs:
            host_output_file = os.path.join(logging_config.log_dir, "host.output")
        else:
            host_output_file = os.path.join(
                logging_config.log_dir, f"host_{node_rank}_{host}.output"
            )

        host_run_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
        )
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

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

            if nodes:
                f.write("ray_path=$(realpath $(which ray))\n")
                master_ip = nodes[0][0]
                target_port = nodes[0][1].get("port")

                f.write("# clean nodes \n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        if not node.get("type", None):
                            raise ValueError(f"Node type must be specified for node {node}.")
                        if not node.get("slots", None):
                            raise ValueError(f"Number of slots must be specified for node {node}.")

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

                for index, (ip, node) in enumerate(nodes):
                    per_node_cmd = None
                    if nodes_envs and nodes_envs.get(ip, None) is not None:
                        per_node_cmd = " && ".join(
                            f"export {key}={value}" for key, value in nodes_envs[ip].items()
                        )

                    if not node.get("type", None):
                        raise ValueError(f"Node type must be specified for node {node}.")
                    if not node.get("slots", None):
                        raise ValueError(f"Number of slots must be specified for node {node}.")

                    if index == 0:
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
                if self.use_fs_serve and config.serve[0].get("engine", None):
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
            f.write("echo '=========== launch task (RayBackend) ==========='\n")

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

    def generate_stop_script(self, config, host, node_rank):
        logging_config = config.logging
        host_stop_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
        )
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        cmds_config = config.experiment.get("cmds", None)
        ssh_port = config.experiment.runner.get("ssh_port", 22)
        docker_name = config.experiment.runner.get("docker", None)
        nodes = config.get("nodes", None)

        after_stop = cmds_config.get("after_stop", "") if cmds_config else ""
        before_start_cmd = cmds_config.get("before_start", "") if cmds_config else ""
        envs = config.experiment.get("envs", {})
        envs_str = " && ".join(f"export {key}={value}" for key, value in envs.items())

        with open(host_stop_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("set -x\n")
            f.write(f"{before_start_cmd}\n")
            f.write(f"{envs_str}\n\n")

            f.write("ray_path=$(realpath $(which ray))\n")

            if nodes:
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
            else:
                node_cmd = None
                if self.use_fs_serve and config.serve[0].get("engine", None):
                    node_cmd = "${ray_path} stop"
                if before_start_cmd:
                    node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
                if node_cmd:
                    f.write(f"{node_cmd}\n")

            f.write("pkill -f 'run_inference_engine'\n")
            f.write("pkill -f 'run_fs_serve_vllm'\n")
            f.write("pkill -f 'vllm serve'\n")
            f.write("pkill -f multiprocessing\n")
            f.write("pkill -f VLLM\n")

            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    pkill -P $pid\n")
            f.write("    kill $pid\n")
            f.write("fi\n")

            if after_stop:
                f.write(f"{after_stop}\n")

            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.chmod(host_stop_script_file, 0o755)
        return host_stop_script_file
