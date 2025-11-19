import multiprocessing
import os
import shlex
import time

from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.elastic.monitor_service import MonitorService
from flagscale.runner.runner_base import JobStatus, RunnerBase
from flagscale.runner.utils import (
    add_decive_extra_config,
    flatten_dict_to_args,
    get_free_port,
    get_host_name_or_ip,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
    run_scp_command,
    run_ssh_command,
    update_cmd_with_node_specific_config,
    update_nodes_envs,
)

_MAX_CPU_COUNT = multiprocessing.cpu_count()


def _get_args_megatron(config: DictConfig):
    assert (
        config.experiment.task.backend == "megatron"
    ), "This function only supports megatron backend."

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict["train"]

    new_config_dict = {}
    new_config_dict.update(config_dict["system"])
    new_config_dict.update(config_dict["model"])
    new_config_dict.update(config_dict["data"])

    ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir"]
    # Flatten the dictionary to a list of arguments
    args = flatten_dict_to_args(new_config_dict, ignore_keys)

    return args


def _get_args_robotics(config: DictConfig):
    assert (
        config.experiment.task.backend == "robotics"
    ), "This function only supports robotics backend."

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict["train"]

    new_config_dict = {}
    new_config_dict.update(config_dict["model"])
    ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir"]
    # Flatten the dictionary to a list of arguments
    args = flatten_dict_to_args(new_config_dict, ignore_keys)
    args = [config_dict["data"]["config_name"]] + args
    return args


def _get_args_pi0(config: DictConfig):
    assert config.experiment.task.backend == "pi0", "This function only supports lerobot backend."

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict["train"]

    new_config_dict = {}
    new_config_dict.update(config_dict["system"])
    new_config_dict.update(config_dict["model"])
    new_config_dict.update(config_dict["data"])

    ignore_keys = [
        "log_dir",
        "details_dir",
        "scripts_dir",
        "pids_dir",
        "save",
        "output_dir",
        "load",
        "tensorboard_dir",
        "wandb_save_dir",
    ]
    # Flatten the dictionary to a list of arguments
    args = flatten_dict_to_args(new_config_dict, ignore_keys=ignore_keys)
    return args


def _update_config_train(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if config.experiment.runner.get("no_shared_fs", False):
        config.train.system.no_shared_fs = True

    config = config.train.system

    if config.get("checkpoint", None) is None:
        config.checkpoint = DictConfig({})

    if config.get("logging", None) is None:
        config.logging = DictConfig({})

    ckpt_save_dir = (
        os.path.abspath(config.checkpoint.save)
        if config.checkpoint.get("save", None)
        else os.path.join(exp_dir, "checkpoints")
    )
    ckpt_load_dir = (
        os.path.abspath(config.checkpoint.load)
        if config.checkpoint.get("load", None)
        else os.path.join(exp_dir, "checkpoints")
    )
    wandb_dir = (
        os.path.abspath(config.logging.wandb_save_dir)
        if config.logging.get("wandb_save_dir", None)
        else os.path.join(exp_dir, "wandb")
    )
    tensorboard_dir = (
        os.path.abspath(config.logging.tensorboard_dir)
        if config.logging.get("tensorboard_dir", None)
        else os.path.join(exp_dir, "tensorboard")
    )
    log_dir = (
        os.path.abspath(config.logging.log_dir)
        if config.logging.get("log_dir", None)
        else os.path.join(exp_dir, "logs")
    )
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")
    details_dir = os.path.join(log_dir, "details")

    config.checkpoint.save = ckpt_save_dir
    config.checkpoint.load = ckpt_load_dir
    config.logging.log_dir = log_dir
    config.logging.scripts_dir = scripts_dir
    config.logging.pids_dir = pids_dir
    config.logging.details_dir = details_dir
    config.logging.tensorboard_dir = tensorboard_dir
    config.logging.wandb_save_dir = wandb_dir

    OmegaConf.set_struct(config, False)


def _get_runner_cmd_train(
    host, master_addr, master_port, nnodes, node_rank, nproc_per_node, config: DictConfig
):
    runner_config = config.experiment.runner
    logging_config = config.train.system.logging

    if runner_config.get("per_node_task", False):
        nnodes = 1
        node_rank = 0
        master_addr = "localhost"

    rdzv_id = runner_config.get("rdzv_id", "default")
    log_dir = runner_config.get("log_dir", logging_config.details_dir)
    log_dir = os.path.abspath(log_dir)
    no_shared_fs = runner_config.get("no_shared_fs", False)
    if no_shared_fs:
        log_dir = os.path.join(log_dir, f"host")
    else:
        log_dir = os.path.join(log_dir, f"host_{node_rank}_{host}")
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    rdzv_backend = runner_config.get("rdzv_backend", "c10d")
    rdzv_endpoint = runner_config.get("rdzv_endpoint", f"{master_addr}:{master_port}")
    redirect = runner_config.get("redirects", "3")
    tee = runner_config.get("tee", "3")
    backend = runner_config.get("backend", "torchrun")

    runner_args = OmegaConf.to_container(runner_config, resolve=True)
    if "type" in runner_args:
        del runner_args["type"]
    if "backend" in runner_args:
        del runner_args["backend"]
    if "per_node_task" in runner_args:
        del runner_args["per_node_task"]
    if "hostfile" in runner_args:
        del runner_args["hostfile"]
    if "ssh_port" in runner_args:
        del runner_args["ssh_port"]
    if "master_addr" in runner_args:
        del runner_args["master_addr"]
    if "master_port" in runner_args:
        del runner_args["master_port"]
    if "enable_monitoring" in runner_args:
        del runner_args["enable_monitoring"]
    if "enable_gpu_health_check" in runner_args:
        del runner_args["enable_gpu_health_check"]

    runner_args["rdzv_id"] = rdzv_id
    # runner_args["master_addr"] = master_addr
    # runner_args["master_port"] = master_port
    runner_args["nnodes"] = nnodes
    runner_args["node_rank"] = node_rank
    runner_args["nproc_per_node"] = nproc_per_node
    runner_args["rdzv_backend"] = rdzv_backend
    runner_args["rdzv_endpoint"] = rdzv_endpoint

    runner_args["log_dir"] = log_dir if backend == "torchrun" else os.path.join(log_dir, rdzv_id)
    runner_args["redirects"] = redirect
    runner_args["tee"] = tee

    runner_cmd = [backend]
    for key, value in runner_args.items():
        if isinstance(value, bool):
            if value:
                runner_cmd.append(f"--{key}")
        else:
            runner_cmd.append(f"--{key}")
            runner_cmd.append(f"{value}")
    return runner_cmd


def _generate_run_script_train(
    config,
    host,
    node_rank,
    cmd,
    background=True,
    with_test=False,
    root_dir=None,
    enable_monitoring=False,
):
    system_config = config.train.system
    logging_config = config.train.system.logging

    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    if no_shared_fs:
        host_output_file = os.path.join(logging_config.log_dir, f"host.output")
    else:
        host_output_file = os.path.join(logging_config.log_dir, f"host_{node_rank}_{host}.output")
    host_run_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
    )
    host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

    os.makedirs(logging_config.scripts_dir, exist_ok=True)
    if root_dir is not None:
        root_dir = os.path.abspath(root_dir)
    else:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists(root_dir), f"ROOT_DIR {root_dir} does not exist."
    megatron_dir = os.path.join(root_dir, "third_party", "Megatron-LM")
    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        before_start = cmds_config.get("before_start", "")
    else:
        before_start = ""
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"{before_start}\n")
        f.write(f"mkdir -p {system_config.checkpoint.load}\n")
        f.write(f"mkdir -p {system_config.checkpoint.save}\n")
        f.write(f"mkdir -p {system_config.logging.log_dir}\n")
        f.write(f"mkdir -p {system_config.logging.pids_dir}\n")
        f.write(f"mkdir -p {system_config.logging.details_dir}\n")
        f.write(f"mkdir -p {system_config.logging.tensorboard_dir}\n")
        f.write(f"mkdir -p {system_config.logging.wandb_save_dir}\n")
        f.write(f"\n")
        f.write(f"cd {root_dir}\n")
        f.write(f"\n")
        f.write(f"export PYTHONPATH={megatron_dir}:{root_dir}:${{PYTHONPATH}}\n")
        f.write(f"\n")
        f.write(f'cmd="{cmd}"\n')
        f.write(f"\n")
        if enable_monitoring:
            monitor_launcher_path = os.path.join(
                root_dir, "flagscale", "runner", "elastic", "monitor_launcher.py"
            )
            ssh_port = config.experiment.runner.get("ssh_port", 22)
            f.write(f'# Start monitoring service in background\n')
            f.write(f'python {monitor_launcher_path} \\\n')
            f.write(f'  --log-dir "{logging_config.log_dir}" \\\n')
            f.write(f'  --pid-file "{host_pid_file}" \\\n')
            f.write(f'  {"--no-shared-fs" if no_shared_fs else ""} \\\n')
            f.write(f'  --ssh-port {ssh_port} \\\n')
            f.write(f'  --interval 5 \\\n')
            f.write(f'  --enable-log-collection \\\n')
            f.write(f'  --enable-diagnostic \\\n')
            f.write(f'  > /tmp/monitor_output_{node_rank}_{host}.log 2>&1 &\n')
            f.write(f'echo "Monitor service started in background"\n')
        f.write(f'\n')

        if with_test:
            f.write(f'bash -c "$cmd; sync" \n')
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

    return host_run_script_file


def _generate_stop_script_train(config, host, node_rank):
    if getattr(config, "train", None):
        logging_config = config.train.system.logging
    else:
        logging_config = config.inference.system.logging

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
        f.write("    pkill -f 'torchrun'\n")
        f.write("fi\n")
        f.write(f"{after_stop}\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_stop_script_file, 0o755)

    return host_stop_script_file


def run_node(
    func,
    node_rank,
    host,
    resource_info,
    user_envs,
    runner_config,
    nnodes,
    available_ip,
    available_port,
    with_test,
    dryrun,
    enable_monitoring=True,
):
    cur_envs = update_nodes_envs(user_envs, host, resource_info)
    # Get the number of visible devices from the environment variable, e.g. CUDA_VISIBLE_DEVICES, MLU_VISIBLE_DEVICES
    # visible_devices = cur_envs.get("CUDA_VISIBLE_DEVICES", None)
    visible_devices = next((v for k, v in cur_envs.items() if k.endswith("_VISIBLE_DEVICES")), None)
    if visible_devices is not None and isinstance(visible_devices, str):
        visible_devices = visible_devices.split(",")
        num_visible_devices = len(visible_devices)
    nproc_from_hostfile = resource_info["slots"]
    nproc_from_args = runner_config.get("nproc_per_node", None)
    nproc_per_node = get_nproc_per_node(nproc_from_hostfile, nproc_from_args, num_visible_devices)
    master_addr = runner_config.get("master_addr", available_ip)
    master_port = runner_config.get("master_port", available_port)
    func(
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        device_type=resource_info["type"],
        with_test=with_test,
        dryrun=dryrun,
        cur_envs=cur_envs,
    )


class SSHTrainRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "train", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        _update_config_train(self.config)
        if self.config.experiment.task.backend == "megatron":
            self.user_args = _get_args_megatron(self.config)
        elif self.config.experiment.task.backend == "robotics":
            self.user_args = _get_args_robotics(self.config)
        elif self.config.experiment.task.backend == "pi0":
            self.user_args = _get_args_pi0(self.config)
        self.rdzv_id = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_script = self.config.experiment.task.entrypoint
        self.resources = parse_hostfile(self.config.experiment.runner.get("hostfile", None))
        self.device_type_specific = self.config.get("device_type_specific", None)
        self.node_specific = self.config.get("node_specific", None)
        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def _run_gpu_health_check_on_node(
        self, host, node_rank, master_addr, master_port, nnodes, nproc_per_node
    ):
        """Run GPU health check on a specific node"""
        import subprocess

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu_health_check_path = os.path.join(
            root_dir, "flagscale", "runner", "elastic", "gpu_health_check.py"
        )

        # Get parallel configuration
        tp_size = self.config.train.model.get("tensor_model_parallel_size", 1)
        pp_size = self.config.train.model.get("pipeline_model_parallel_size", 1)

        # Build command
        if nnodes > 1 or nproc_per_node > 1:
            # Use torchrun for distributed health check
            cmd = [
                "torchrun",
                f"--nnodes={nnodes}",
                f"--nproc_per_node={nproc_per_node}",
                f"--node_rank={node_rank}",  # Use the correct node rank for this node
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                gpu_health_check_path,
                "--tensor-model-parallel-size",
                str(tp_size),
                "--pipeline-model-parallel-size",
                str(pp_size),
                "--distributed-backend",
                "nccl",
                "--distributed-timeout-minutes",
                "5",
            ]
        else:
            # Single GPU mode
            cmd = [
                "python",
                gpu_health_check_path,
                "--tensor-model-parallel-size",
                str(tp_size),
                "--pipeline-model-parallel-size",
                str(pp_size),
                "--distributed-backend",
                "nccl",
                "--distributed-timeout-minutes",
                "5",
            ]

        cmd_str = " ".join(cmd)

        if host != "localhost":
            # Run on remote host via SSH
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            logger.info(f"Running GPU health check on {host} (node_rank={node_rank})")

            try:
                result = run_ssh_command(host, cmd_str, ssh_port, query=True)
                # SSH command returns subprocess.CompletedProcess or similar
                return result.returncode == 0 if hasattr(result, 'returncode') else True
            except Exception as e:
                logger.error(f"Failed to run GPU health check on {host}: {e}")
                return False
        else:
            # Run locally
            logger.info(f"Running GPU health check locally (node_rank={node_rank})")
            logger.info(f"Command: {cmd_str}")

            try:
                result = subprocess.run(cmd, check=False, text=True, capture_output=False)
                return result.returncode == 0
            except Exception as e:
                logger.error(f"Failed to run GPU health check locally: {e}")
                return False

    def _run_gpu_health_check(self):
        """Run GPU health check across all nodes"""
        # Check if the health check script exists
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu_health_check_path = os.path.join(
            root_dir, "flagscale", "runner", "elastic", "gpu_health_check.py"
        )

        if not os.path.exists(gpu_health_check_path):
            logger.error(f"GPU health check script not found at {gpu_health_check_path}")
            return False
        runner_config = self.config.experiment.runner

        if self.resources is not None:
            # Multi-node mode: run health check on each node
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)

            # Get master address and port
            available_ip = list(self.resources.keys())[0]
            master_addr = runner_config.get("master_addr", available_ip)
            master_port = runner_config.get("master_port", 29500)

            logger.info(f"Running GPU health check across {nnodes} nodes")

            # Run health check on each node with proper node_rank
            all_results = []
            for node_rank, (host, resource_info) in enumerate(self.resources.items()):
                if node_rank >= nnodes:
                    break

                # Get nproc_per_node for this specific node
                nproc_from_hostfile = resource_info["slots"]
                nproc_from_args = runner_config.get("nproc_per_node", None)

                # Get CUDA_VISIBLE_DEVICES if set
                cur_envs = add_decive_extra_config(self.user_envs, resource_info["type"])
                visible_devices = cur_envs.get("CUDA_VISIBLE_DEVICES", None)
                num_visible_devices = None
                if visible_devices is not None and isinstance(visible_devices, str):
                    visible_devices = visible_devices.split(",")
                    num_visible_devices = len(visible_devices)

                nproc_per_node = get_nproc_per_node(
                    nproc_from_hostfile, nproc_from_args, num_visible_devices
                )

                logger.info(f"Checking node {node_rank} ({host}) with {nproc_per_node} GPUs")

                result = self._run_gpu_health_check_on_node(
                    host, node_rank, master_addr, master_port, nnodes, nproc_per_node
                )
                all_results.append(result)

                if not result:
                    logger.error(f"GPU health check failed on node {node_rank} ({host})")
                    return False

            logger.info("GPU health check passed on all nodes")
            return all(all_results)

        else:
            # Single-node mode
            nnodes = 1
            node_rank = 0
            host = "localhost"

            visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
            num_visible_devices = None
            if visible_devices is not None and isinstance(visible_devices, str):
                visible_devices = visible_devices.split(",")
                num_visible_devices = len(visible_devices)

            nproc_from_args = runner_config.get("nproc_per_node", None)
            nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)

            master_addr = runner_config.get("master_addr", "localhost")
            master_port = runner_config.get("master_port", 29500)

            logger.info(f"Running single-node GPU health check with {nproc_per_node} GPUs")

            return self._run_gpu_health_check_on_node(
                host, node_rank, master_addr, master_port, nnodes, nproc_per_node
            )

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        device_type=None,
        with_test=False,
        dryrun=False,
        cur_envs=None,
        enable_monitoring=True,
    ):
        export_cmd = []

        for k, v in cur_envs.items():
            export_cmd += [f"{k}={v}"]

        runner_cmd = _get_runner_cmd_train(
            host, master_addr, master_port, nnodes, node_rank, nproc_per_node, self.config
        )
        # update hetero-current-device-type according to the device_type in hostfile
        if device_type is not None:
            if "--hetero-current-device-type" in self.user_args:
                idx = self.user_args.index("--hetero-current-device-type")
                self.user_args[idx + 1] = device_type
            else:
                self.user_args += ["--hetero-current-device-type", device_type]

        cmd = shlex.join(export_cmd + runner_cmd + [self.user_script] + self.user_args)
        # update cmd with node_specific_config
        node_specific_config = {}
        if device_type is not None:
            node_specific_config = (
                self.device_type_specific.get(device_type, {}) if self.device_type_specific else {}
            )
        node_specific_config.update(self.node_specific.get(host, {}) if self.node_specific else {})
        cmd = update_cmd_with_node_specific_config(cmd, node_specific_config)

        logging_config = self.config.train.system.logging
        host_run_script_file = _generate_run_script_train(
            self.config,
            host,
            node_rank,
            cmd,
            background=True,
            with_test=with_test,
            root_dir=node_specific_config.get("build_dir", None),
            enable_monitoring=enable_monitoring,
        )

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, dryrun)

            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_run_script_file, logging_config.scripts_dir, ssh_port, dryrun
                )

            # Step 3: run the host_run_script_file on the remote host
            run_ssh_command(host, f"bash {host_run_script_file}", ssh_port, dryrun)
        else:
            run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(
        self,
        with_test=False,
        dryrun=False,
        monitor=False,
        interval=10,
        enable_log_collection=True,
        enable_diagnostic=True,
        enable_monitoring=False,
        enable_gpu_health_check=None,
    ):
        if enable_gpu_health_check is None:
            enable_gpu_health_check = self.config.experiment.runner.get(
                "enable_gpu_health_check", False
            )
        # Run GPU health check first if enabled (before script generation)
        if enable_gpu_health_check:
            logger.info("Starting GPU health check before training setup...")
            if not self._run_gpu_health_check():
                logger.error("GPU health check failed! Aborting training setup.")
                return
            logger.info("GPU health check passed successfully!")
            logger.info("Proceeding with training script generation...")

        num_visible_devices = None
        runner_config = self.config.experiment.runner

        # If hostfile is provided, use the resources from the hostfile
        if self.resources is not None:
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)
            available_ip = list(self.resources.keys())[0]
            available_port = get_free_port()
            num_processes = min(nnodes, _MAX_CPU_COUNT)
            with multiprocessing.Pool(processes=num_processes) as pool:
                tasks = []
                for node_rank, (host, resource_info) in enumerate(self.resources.items()):
                    if node_rank >= nnodes:
                        break
                    args = (
                        self._run_each,
                        node_rank,
                        host,
                        resource_info,
                        self.user_envs,
                        runner_config,
                        nnodes,
                        available_ip,
                        available_port,
                        with_test,
                        dryrun,
                        enable_monitoring,
                    )
                    tasks.append(args)
                pool.starmap(run_node, tasks)
        else:
            # If hostfile is not provided, run the job on localhost
            visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is not None and isinstance(visible_devices, str):
                visible_devices = visible_devices.split(",")
                num_visible_devices = len(visible_devices)
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
                cur_envs=self.user_envs,
                enable_monitoring=enable_monitoring,
            )
        # If need monitor, query status continually
        if monitor:
            # sleep to wait task already started
            time.sleep(interval)
            try:
                while True:
                    status = self._query_status()
                    logger.info(f"Job Status: {status.name}")
                    if status == JobStatus.COMPLETED_OR_IDLE:
                        break
                    time.sleep(interval)
                logger.info("Job Ended.")
            except Exception as e:
                logger.info(e)

        if enable_monitoring:
            logger.info("Starting monitoring service...")
            monitor_service = MonitorService(self.config, self, interval)
            monitor_service.start_monitoring(
                enable_log_collection=enable_log_collection, enable_diagnostic=enable_diagnostic
            )
            logger.info("Monitoring service started in background")
            logger.info("Training job will continue running, monitor logs will be saved")

            # Return the monitor_service instance for external control.
            return monitor_service

        return None

    def _stop_each(self, host, node_rank):
        host_stop_script_file = _generate_stop_script_train(self.config, host, node_rank)
        logging_config = self.config.train.system.logging

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port)
            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(host, host_stop_script_file, logging_config.scripts_dir, ssh_port)
            # Step 3: run the host_run_script_file on the remote host
            run_ssh_command(host, f"bash {host_stop_script_file}", ssh_port)
        else:
            run_local_command(f"bash {host_stop_script_file}")

    def stop(self):
        if self.resources is None:
            self._stop_each("localhost", 0)
            return

        nnodes = get_nnodes(len(self.resources), self.config.experiment.runner.get("nnodes", None))

        num_processes = min(nnodes, _MAX_CPU_COUNT)
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = []
            for node_rank, (host, _) in enumerate(self.resources.items()):
                if node_rank >= nnodes:
                    break
                args = (host, node_rank)
                tasks.append(args)
            pool.starmap(self._stop_each, tasks)

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.train.system.logging

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
                "    pid=$(ps aux | grep 'torchrun' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file

    def _generate_query_sub_process_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.train.system.logging

        host_query_sub_process_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query_sub_process.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_sub_process_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    ps -eo pid,ppid | awk -v ppid=$pid '$2 == ppid {print $1}'\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to query the job.
            f.write(
                "    pid=$(ps aux | grep 'torchrun' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -eo pid,ppid | awk -v ppid=$pid '$2 == ppid {print $1}'\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_sub_process_script_file, 0o755)

        return host_query_sub_process_script_file

    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        logging_config = self.config.train.system.logging
        result = ""
        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, query=True)
            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(host, host_query_script_file, logging_config.scripts_dir, ssh_port)
            # Step 3: run the host_run_script_file on the remote host
            try:
                result = run_ssh_command(
                    host, f"bash {host_query_script_file}", ssh_port, query=True
                )
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        else:
            try:
                result = run_local_command(f"bash {host_query_script_file}", query=True)
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_each_sub_process(self, host, node_rank):
        "Query each node sub process status."
        host_query_script_file = self._generate_query_sub_process_script(host, node_rank)
        logging_config = self.config.train.system.logging
        result = ""
        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, query=True)
            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(host, host_query_script_file, logging_config.scripts_dir, ssh_port)
            # Step 3: run the host_run_script_file on the remote host
            try:
                result = run_ssh_command(
                    host, f"bash {host_query_script_file}", ssh_port, query=True
                )
            except Exception as e:
                logger.error(f"Failed to query sub process status on {host}: {e}")
        else:
            try:
                result = run_local_command(f"bash {host_query_script_file}", query=True)
            except Exception as e:
                logger.error(f"Failed to query sub process status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_status(self):
        "Query Job status."
        results = []
        if self.resources is None:
            result = self._query_each("localhost", 0)
            results.append(result)

        else:
            host_list = list(self.resources.keys())
            for host, _ in self.resources.items():
                node_rank = host_list.index(host)
                result = self._query_each(host, node_rank)
                results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status

    def _query_sub_process_status(self):
        "Query sub process status."
        results = []
        if self.resources is None:
            result = self._query_each_sub_process("localhost", 0)
            results.append(result)

        else:
            host_list = list(self.resources.keys())
            for host, _ in self.resources.items():
                node_rank = host_list.index(host)
                result = self._query_each_sub_process(host, node_rank)
                results.append(result)
        if all(status for status in results):
            status = True
        else:
            status = False
        return status

    def query_once(self):
        """
        Query job status once (non-blocking).
        There are three kinds of status for a Job:
            RUNNING: The job is running.
            COMPLETED_OR_IDLE: The job is completed or idle.
            TRANSITIONAL: The job is starting or stopping.

        Returns:
            JobStatus: Current job status
        """
        return self._query_status()

    def start_monitoring_service(
        self, interval=10, enable_log_collection=True, enable_diagnostic=True
    ):
        """
        Start independent monitoring service (non-blocking).

        Args:
            interval (int): Monitor interval in seconds
            enable_log_collection (bool): Enable log collection
            enable_diagnostic (bool): Enable diagnostic report generation

        Returns:
            MonitorService: Monitor service instance
        """
        monitor_service = MonitorService(self.config, self, interval)
        monitor_service.start_monitoring(
            enable_log_collection=enable_log_collection, enable_diagnostic=enable_diagnostic
        )
        logger.info(f"Independent monitoring service started with interval={interval}s")
        return monitor_service

    def query(self, interval=10, timeout=None):
        """
        Query job status and log with optional timeout (blocking).
        There are three kinds of status for a Job:
            RUNNING: The job is running.
            COMPLETED_OR_IDLE: The job is completed or idle.
            TRANSITIONAL: The job is starting or stopping.

        Args:
            interval (int, optional): The interval of querying job status. Default: 10.
            timeout (float, optional): The timeout of query job status, if None, the query will keep indefinitely. Default: None.

        Returns:
            None

        Warning:
                    This method is blocking and should be used with caution.
                                Consider using query_once() or start_monitoring_service() for non-blocking alternatives.
        """
        logger.warning(
            "Using blocking query method. Consider using query_once() or start_monitoring_service()"
        )

        if timeout is None:
            logger.warning("Entering indefinite blocking query loop. Press Ctrl+C to exit.")
            try:
                while True:
                    job_status = self._query_status()
                    logger.info(f"Job status: {job_status.name}")
                    time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Query interrupted by user")
        else:
            start_time = time.time()
            cur_time = time.time()
            while cur_time - start_time < timeout:
                job_status = self._query_status()
                logger.info(f"Job status: {job_status.name}")
                time.sleep(interval)
                cur_time = time.time()
            logger.info(f"Query timeout reached ({timeout}s)")


class CloudTrainRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "train", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_script = self.config.experiment.task.entrypoint
        _update_config_train(self.config)
        if self.config.experiment.task.backend == "megatron":
            self.user_args = _get_args_megatron(self.config)
        elif self.config.experiment.task.backend == "robotics":
            self.user_args = _get_args_robotics(self.config)
        logger.info("\n************** configuration ***********")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def _run_gpu_health_check(self):
        """Run GPU health check for CloudTrainRunner - handles both single and multi-node cases"""
        import subprocess

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gpu_health_check_path = os.path.join(
            root_dir, "flagscale", "runner", "elastic", "gpu_health_check.py"
        )

        # Check if the health check script exists
        if not os.path.exists(gpu_health_check_path):
            logger.error(f"GPU health check script not found at {gpu_health_check_path}")
            return False

        # Get parallel configuration
        tp_size = self.config.train.model.get("tensor_model_parallel_size", 1)
        pp_size = self.config.train.model.get("pipeline_model_parallel_size", 1)

        # Get runner configuration for cloud environment
        runner_config = self.config.experiment.runner

        # Detect number of visible devices
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices:
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        # Get node configuration
        nnodes_from_args = runner_config.get("nnodes", None)
        nnodes = get_nnodes(None, nnodes_from_args)

        # Get the actual node_rank for this node in multi-node setup
        # In CloudTrainRunner, node_rank must be provided in the config
        node_rank = runner_config.node_rank if hasattr(runner_config, 'node_rank') else 0

        nproc_from_args = runner_config.get("nproc_per_node", None)
        nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)

        # Get master address and port for multi-node coordination
        # In CloudTrainRunner, master_addr and master_port must be provided in the config
        master_addr = (
            runner_config.master_addr if hasattr(runner_config, 'master_addr') else "localhost"
        )
        master_port = runner_config.master_port if hasattr(runner_config, 'master_port') else 29500

        logger.info(
            f"GPU health check configuration - nnodes: {nnodes}, node_rank: {node_rank}, nproc_per_node: {nproc_per_node}"
        )

        # Build command based on configuration
        if nnodes > 1 or nproc_per_node > 1:
            # Multi-node or multi-GPU: use torchrun
            cmd = [
                "torchrun",
                f"--nnodes={nnodes}",
                f"--nproc_per_node={nproc_per_node}",
                f"--node_rank={node_rank}",  # Use the actual node_rank for this node
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                gpu_health_check_path,
                "--tensor-model-parallel-size",
                str(tp_size),
                "--pipeline-model-parallel-size",
                str(pp_size),
                "--distributed-backend",
                "nccl",
                "--distributed-timeout-minutes",
                "10",
            ]
        else:
            # Single node, single GPU: use python directly
            cmd = [
                "python",
                gpu_health_check_path,
                "--tensor-model-parallel-size",
                str(tp_size),
                "--pipeline-model-parallel-size",
                str(pp_size),
                "--distributed-backend",
                "nccl",
                "--distributed-timeout-minutes",
                "10",
            ]

        logger.info(f"Running GPU health check command: {' '.join(cmd)}")

        if nnodes > 1:
            logger.info(
                f"Note: In multi-node cloud setups, each node must run its health check independently."
            )
            logger.info(
                f"This node ({node_rank}) is participating in the distributed health check."
            )

        # Run the health check command with output to console
        try:
            result = subprocess.run(
                cmd,
                check=False,  # Don't raise exception on non-zero exit
                text=True,
                capture_output=False,  # Let output go to console directly
            )

            if result.returncode == 0:
                logger.info(f"GPU health check passed on node {node_rank}")
                return True
            else:
                logger.error(
                    f"GPU health check failed on node {node_rank} with exit code {result.returncode}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to run GPU health check on node {node_rank}: {e}")
            return False

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

        runner_cmd = _get_runner_cmd_train(
            host, master_addr, master_port, nnodes, node_rank, nproc_per_node, self.config
        )

        cmd = shlex.join(export_cmd + runner_cmd + [self.user_script] + self.user_args)

        host_run_script_file = _generate_run_script_train(
            self.config,
            host,
            node_rank,
            cmd,
            background=False,
            with_test=with_test,
            enable_monitoring=enable_monitoring,
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, with_test=False, dryrun=False, enable_gpu_health_check=None):
        if enable_gpu_health_check is None:
            enable_gpu_health_check = self.config.experiment.runner.get(
                "enable_gpu_health_check", False
            )
        if dryrun:
            logger.info("Dryrun mode is not supported in CloudRunner.")
            return

        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices:
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner
        nnodes_from_args = runner_config.get("nnodes", None)
        nnodes = get_nnodes(None, nnodes_from_args)
        node_rank = runner_config.node_rank
        nproc_from_args = runner_config.get("nproc_per_node", None)
        nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
        master_addr = runner_config.master_addr
        master_port = runner_config.master_port
        host = get_host_name_or_ip()
        self._run_each(
            host,
            master_addr,
            master_port,
            nnodes,
            node_rank,
            nproc_per_node,
            with_test=with_test,
            dryrun=dryrun,
        )
