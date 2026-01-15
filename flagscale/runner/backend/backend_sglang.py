import copy
import json
import os

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend.backend_base import BackendBase
from flagscale.runner.utils import flatten_dict_to_args, get_free_port, logger, parse_hostfile
from flagscale.serve.args_mapping.mapping import ARGS_CONVERTER


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


class SglangBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
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

        logger.info("\n************** Sglang Configuration **************")
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
            import sglang

            sglang_path = os.path.dirname(sglang.__path__[0])
        except Exception:
            sglang_path = f"{root_dir}/sglang"

        envs = config.experiment.get("envs", {})

        with open(host_run_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("set -x\n")
            f.write("\n")
            f.write(f"{before_start_cmd}\n")
            f.write("\n")

            f.write('if [ -z "$PYTHONPATH" ]; then\n')
            f.write(f"    export PYTHONPATH={sglang_path}:{root_dir}\n")
            f.write("else\n")
            f.write(f'    export PYTHONPATH="$PYTHONPATH:{sglang_path}:{root_dir}"\n')
            f.write("fi\n")
            f.write("\n")

            envs_str = " && ".join(
                f"export {key}={value}" for key, value in envs.items() if key != "nodes_envs"
            )
            f.write(f"{envs_str}\n")

            if nodes:
                f.write("# clean nodes \n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        if not node.get("type", None):
                            raise ValueError(f"Node type must be specified for node {node}.")
                        if not node.get("slots", None):
                            raise ValueError(f"Number of slots must be specified for node {node}.")

                        node_cmd = "pkill -f 'sglang.launch_server' && pkill -f python"
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        if envs_str:
                            node_cmd = f"{envs_str} && " + node_cmd

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")

                if before_start_cmd:
                    f.write(f"{before_start_cmd} && pkill -f 'sglang.launch_server'\n")
                else:
                    f.write("pkill -f 'sglang.launch_server'\n")

                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write("\n")

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
                            raise ValueError("No sglang model configuration found in task config.")

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

                            if ARGS_CONVERTER:
                                converted_args = ARGS_CONVERTER.convert("sglang", common_args)
                            else:
                                converted_args = common_args

                            common_args_flatten = flatten_dict_to_args(converted_args, ["model"])
                            command.extend(common_args_flatten)

                            sglang_args_flatten = flatten_dict_to_args(sglang_args, ["model"])
                            command.extend(sglang_args_flatten)
                        else:
                            raise ValueError("Either model should be specified in sglang model.")

                        command.extend(["--node-rank", str(index)])

                        runner_config = config.experiment.runner
                        nnodes_conf = runner_config.get("nnodes", None)
                        addr_conf = runner_config.get("master_addr", None)
                        port_conf = runner_config.get("master_port", None)

                        if nnodes_conf is None or addr_conf is None or port_conf is None:
                            raise ValueError(
                                "nnodes, master_addr, master_port must be specified in runner when engine is sglang with multi-nodes mode."
                            )

                        command.extend(["--nnodes", str(nnodes_conf)])
                        command.extend(["--dist-init-addr", str(addr_conf) + ":" + str(port_conf)])
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

                if before_start_cmd:
                    node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
                if node_cmd:
                    f.write(f"{node_cmd}\n")

            logger.info(f"in generate_run_script_serve_sglang, write cmd: {cmd}")
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
        """
        Adapted for Sglang process cleanup.
        """
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

        ssh_port = config.experiment.runner.get("ssh_port", 22)
        docker_name = config.experiment.runner.get("docker", None)
        if cmds_config:
            before_start_cmd = cmds_config.get("before_start", "")
        else:
            before_start_cmd = ""

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
                f.write("# clean nodes\n")
                if len(nodes) > 1:
                    for ip, node in nodes[1:]:
                        node_cmd = "pkill -f 'sglang.launch_server' && pkill -f python"
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        if envs_str:
                            node_cmd = f"{envs_str} && " + node_cmd

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")

            f.write("pkill -f 'sglang.launch_server'\n")

            if after_stop:
                f.write(f"{after_stop}\n")

            f.flush()
            os.fsync(f.fileno())

        os.chmod(host_stop_script_file, 0o755)
        return host_stop_script_file
