import collections
import json
import os
import socket

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend.backend_base import BackendBase
from flagscale.runner.utils import (
    ResourceManager,
    flatten_dict_to_args,
    get_addr,
    get_free_port,
    get_ip_addr,
    is_ip_addr,
    is_master_node,
    logger,
    parse_hostfile,
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
    # step1: yaml -> dict
    assert config.experiment.task.backend in ["vllm"], "This function only supports vllm backend."
    config_dict = OmegaConf.to_container(config, resolve=True)

    # step2: restructuring the config
    config_dict = config_dict["inference"]
    config_dict["logging"].pop("log_dir")
    config_dict["logging"].pop("scripts_dir")
    config_dict["logging"].pop("pids_dir")
    if not config_dict.get("logging"):
        config_dict.pop("logging")

    # step3: dict -> yaml
    logging_config = config.inference.logging
    new_config = OmegaConf.create(config_dict)
    new_conf_file = os.path.join(logging_config.scripts_dir, "inference.yaml")

    # step4: write the new yaml file to `outputs_dir/inference_logs/scripts/inference.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


def _get_serve_engine(config):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    if serve_config and len(serve_config) > 1:
        logger.warning(f"Multiple 'serve' configurations found in task config: {serve_config}")

    engine = serve_config[0].get("engine", None)
    return engine


def _get_serve_engine_args(config, backend="vllm"):
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


def _get_args_sglang(config: DictConfig):
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


def _get_args_cloud(config: DictConfig):
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


def _update_config_inference(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if config.get("logging", None) is None:
        config.inference.logging = DictConfig({})

    log_dir = os.path.join(exp_dir, "inference_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.inference.logging.log_dir = log_dir
    config.inference.logging.scripts_dir = scripts_dir
    config.inference.logging.pids_dir = pids_dir

    os.makedirs(config.inference.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


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


def match_address(address):
    """Check if current node is matched."""

    if is_ip_addr(address):
        return get_ip_addr() == address
    else:
        hostname = socket.gethostname()
        return hostname == address


class VllmBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "inference" or self.task_type == "serve", (
            f"Unsupported task type: {self.task_type}"
        )
        self.deploy_config = self.config.experiment.get("runner", {}).get("deploy", {})
        self.use_fs_serve = self.deploy_config.get("use_fs_serve", True)
        self.launcher_type = self.config.experiment.runner.get("type", "ssh")

        self._prepare()

    def _prepare(self):
        if self.task_type == "inference":
            _update_config_inference(self.config)
            self.user_args = _get_args_vllm(self.config)
            self.user_envs = self.config.experiment.get("envs", {})
            self.user_script = self.config.experiment.task.entrypoint
            self.resources = parse_hostfile(self.config.experiment.runner.get("hostfile", None))
        elif self.task_type == "serve" and self.launcher_type == "cloud":
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
            self.user_args = _get_args_cloud(self.config)
            self.user_envs = self.config.experiment.get("envs", {})
            entrypoint = self.config.experiment.task.get("entrypoint", None)

            inference_engine = "vllm"
            if not entrypoint and _get_serve_engine(self.config):
                assert inference_engine == _get_serve_engine(self.config), (
                    f"_get_serve_engine(self.config) is {_get_serve_engine(self.config)} should be vllm"
                )

            if inference_engine:
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
        elif self.task_type == "serve" and self.launcher_type == "ssh":
            _update_config_serve(self.config)
            self.user_envs = self.config.experiment.get("envs", {})
            self.user_args = _get_args_sglang(self.config)

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

            if (
                self.config.experiment.get("runner", {})
                .get("deploy", {})
                .get("prefill_decode_disaggregation", False)
            ):
                self.user_script = "flagscale/serve/run_disagg_xpyd_router.py"
            elif not self.config.experiment.runner.deploy.use_fs_serve:
                self.user_script = "flagscale/serve/run_inference_engine.py"
            else:
                self.user_script = "flagscale/serve/run_fs_serve_vllm.py"

        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def generate_run_script(self, config, host, node_rank, cmd, background=True, with_test=False):
        if self.task_type == "inference":
            logging_config = config.inference.logging

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
            if cmds_config:
                before_start = cmds_config.get("before_start", "")
            else:
                before_start = ""
            with open(host_run_script_file, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write(f"{before_start}\n")
                f.write(f"mkdir -p {logging_config.log_dir}\n")
                f.write(f"mkdir -p {logging_config.pids_dir}\n")
                f.write("\n")
                f.write(f"cd {root_dir}\n")
                f.write("\n")
                f.write(f"export PYTHONPATH={root_dir}:${{PYTHONPATH}}\n")
                f.write("\n")
                f.write(f'cmd="{cmd}"\n')
                f.write("\n")
                if with_test:
                    f.write(f'bash -c "$cmd; sync"  >> {host_output_file} \n')
                else:
                    # TODO: need a option to control whether to append or overwrite the output file
                    # Now, it always appends to the output file
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
        elif self.task_type == "serve" and self.launcher_type == "cloud":
            nodes = config.get("nodes", None)
            logging_config = config.logging
            node_id = get_addr()

            no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                host_output_file = os.path.join(logging_config.log_dir, "host.output")
            else:
                host_output_file = os.path.join(
                    logging_config.log_dir, f"host_{node_rank}_{host}.output"
                )

            if node_id:
                host_run_script_file = os.path.join(
                    logging_config.scripts_dir, node_id, f"host_{node_rank}_{host}_run.sh"
                )
                os.makedirs(os.path.join(logging_config.scripts_dir, node_id), exist_ok=True)
            else:
                host_run_script_file = os.path.join(
                    logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
                )

            host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

            os.makedirs(logging_config.scripts_dir, exist_ok=True)

            root_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
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

                    ip = get_ip_addr()
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
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --num-gpus={node.slots}"
                        elif node.type == "cpu":
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --num-cpus={node.slots}"
                        else:
                            resource = json.dumps({node.type: node.slots}).replace('"', '"')
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --resources='{resource}'"
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        f.write(f"{node_cmd}\n")
                    else:
                        # worker nodes
                        f.write("\n")
                        f.write("# worker nodes\n")
                        if wait_for_ray_master(master_addr, master_port):
                            if node.type == "gpu":
                                node_cmd = f"${{ray_path}} start --address={address} --num-gpus={node.slots}"

                            elif node.type == "cpu":
                                node_cmd = f"${{ray_path}} start --address={address} --num-cpus={node.slots}"
                            else:
                                resource = json.dumps({node.type: node.slots}).replace('"', '"')
                                node_cmd = f"${{ray_path}} start --address={address} --resources='{resource}'"
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

                    if deploy_config.get("use_fs_serve", True) and config.serve[0].get(
                        "engine", None
                    ):
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
                        node_cmd = (
                            f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
                        )
                    if node_cmd:
                        f.write(f"{node_cmd}\n")

                # Only write launch command if we are master or in single node mode (implied by original logic)
                if not node_id or current_node_is_master:
                    f.write(f"mkdir -p {logging_config.log_dir}\n")
                    f.write(f"mkdir -p {logging_config.pids_dir}\n")
                    f.write("\n")
                    f.write(f"cd {root_dir}\n")
                    f.write("\n")
                    f.write(f'cmd="{cmd}"\n')
                    f.write("\n")

                    f.write("echo '=========== launch task (Cloud - VllmBackend) ==========='\n")
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
        elif self.task_type == "serve" and self.launcher_type == "ssh":
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

                        engine_args = _get_serve_engine_args(config)
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
                                    node_cmd = f"${{ray_path}} start --address={address} --num-gpus={node.slots}"

                                elif node.type == "cpu":
                                    node_cmd = f"${{ray_path}} start --address={address} --num-cpus={node.slots}"
                                else:
                                    resource = json.dumps({node.type: node.slots}).replace(
                                        '"', '\\"'
                                    )
                                    node_cmd = f"${{ray_path}} start --address={address} --resources='{resource}'"
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

                    if deploy_config.get("use_fs_serve", True) and config.serve[0].get(
                        "engine", None
                    ):
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
                        node_cmd = (
                            f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
                        )
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

    def generate_stop_script(self, config, host, node_rank):
        if self.task_type == "inference":
            logging_config = config.inference.logging

            host_stop_script_file = os.path.join(
                logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
            )

            host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

            os.makedirs(logging_config.scripts_dir, exist_ok=True)

            cmds_config = config.experiment.get("cmds", None)
            if cmds_config:
                after_stop = cmds_config.get("after_stop", "")
            else:
                after_stop = ""
            with open(host_stop_script_file, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("if [ -f " + host_pid_file + " ]; then\n")
                f.write("    pid=$(cat " + host_pid_file + ")\n")
                f.write("    pkill -P $pid\n")
                f.write("else\n")
                # TODO: This is a temporary fix. We need to find a better way to stop the job.
                f.write("    pkill -f 'python'\n")
                f.write("fi\n")
                f.write(f"{after_stop}\n")
                f.flush()
                os.fsync(f.fileno())
            os.chmod(host_stop_script_file, 0o755)
        elif self.task_type == "serve" and self.launcher_type == "cloud":
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
                    if deploy_config.get("use_fs_serve", True) and config.serve[0].get(
                        "engine", None
                    ):
                        f.write("ray_path=$(realpath $(which ray))\n")
                        node_cmd = "${ray_path} stop"
                    if before_start_cmd:
                        node_cmd = (
                            f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
                        )
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
        elif self.task_type == "serve" and self.launcher_type == "ssh":
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
                    if deploy_config.get("use_fs_serve", True) and config.serve[0].get(
                        "engine", None
                    ):
                        f.write("ray_path=$(realpath $(which ray))\n")
                        node_cmd = "${ray_path} stop"
                    if before_start_cmd:
                        node_cmd = (
                            f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
                        )
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
