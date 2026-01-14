import asyncio
import multiprocessing
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.elastic.monitor_service import MonitorService
from flagscale.runner.launcher.launcher_base import LauncherBase
from flagscale.runner.utils import (
    JobStatus,
    add_decive_extra_config,
    benchmark,
    dummy_random_input,
    get_free_port,
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
        log_dir = os.path.join(log_dir, "host")
    else:
        log_dir = os.path.join(log_dir, f"host_{node_rank}_{host}")
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    rdzv_backend = runner_config.get("rdzv_backend", "c10d")
    rdzv_endpoint = runner_config.get("rdzv_endpoint", f"{master_addr}:{master_port}")
    # redirect = runner_config.get("redirects", "3")
    tee = runner_config.get("tee", "3")

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

    runner_args["log_dir"] = log_dir
    runner_args["tee"] = tee

    runner_cmd = ["torchrun"]
    for key, value in runner_args.items():
        if isinstance(value, bool):
            if value:
                runner_cmd.append(f"--{key}")
        else:
            runner_cmd.append(f"--{key}")
            runner_cmd.append(f"{value}")
    return runner_cmd


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
    cur_envs,
    enable_monitoring,
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
        enable_monitoring=enable_monitoring,
    )


class SshLauncher(LauncherBase):
    def __init__(self, config, backend):
        self.config = config
        hostfile = self.config.experiment.runner.get("hostfile", None)
        self.resources = parse_hostfile(hostfile) if hostfile else None
        self.task_type = getattr(self.config.experiment.task, "type", None)
        self.backend = backend
        self.user_args = self.backend.user_args
        self.user_envs = self.backend.user_envs
        self.user_script = self.backend.user_script
        self.gpu_health_check_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "runner",
            "elastic",
            "gpu_health_check.py",
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
        enable_monitoring=False,
    ):
        export_cmd = []
        if cur_envs:
            for k, v in cur_envs.items():
                if k != "nodes_envs":
                    export_cmd += [f"{k}={v}"]
        if self.task_type == "train":
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
                    self.backend.device_type_specific.get(device_type, {})
                    if self.backend.device_type_specific
                    else {}
                )
            node_specific_config.update(
                self.backend.node_specific.get(host, {}) if self.backend.node_specific else {}
            )
            cmd = update_cmd_with_node_specific_config(cmd, node_specific_config)
        elif self.task_type == "rl":
            ray_cmd = []
            if self.resources is not None:
                runtime_env = self.config.experiment.runner.get(
                    "runtime_env", "third_party/verl/verl/trainer/runtime_env.yaml"
                )
                ray_dashboard_port = self.config.experiment.runner.get("ray_dashboard_port", 8265)
                ray_cmd = [
                    "ray",
                    "job",
                    "submit",
                    f"--address=http://{host}:{ray_dashboard_port}",
                    f"--runtime-env={runtime_env}",
                    "--no-wait",
                    "--",
                ]
            cmd = shlex.join(
                [*ray_cmd, *export_cmd, "python3", "-m", self.user_script, *self.user_args]
            )
        else:
            cmd = shlex.join([*export_cmd, "python", self.user_script, *self.user_args])

        if self.task_type == "inference":
            logging_config = self.config.inference.logging
        elif self.task_type == "compress":
            logging_config = self.config.compress.system.logging
        elif self.task_type == "serve":
            logging_config = self.config.logging
        elif self.task_type == "train":
            logging_config = self.config.train.system.logging
        elif self.task_type == "rl":
            logging_config = self.config.system.logging
        # todo: unify logging configs of all tasks
        if self.task_type == "train":
            host_run_script_file = self.backend.generate_run_script(
                self.config,
                host,
                node_rank,
                cmd,
                background=True,
                with_test=with_test,
                root_dir=node_specific_config.get("build_dir", None),
                enable_monitoring=enable_monitoring,
            )
        elif self.task_type == "rl":
            host_run_script_file = self.backend.generate_run_script(
                self.config,
                host,
                node_rank,
                cmd,
                background=True,
                with_test=with_test,
                resources=self.resources,
            )
        else:
            host_run_script_file = self.backend.generate_run_script(
                self.config, host, node_rank, cmd, background=True, with_test=with_test
            )

        if self.task_type == "serve":
            run_local_command(f"bash {host_run_script_file}", dryrun)
        else:
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
        enable_monitoring=None,
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
        # Read from config if not explicitly provided
        if enable_monitoring is None:
            enable_monitoring = self.config.experiment.runner.get("enable_monitoring", False)
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner
        # If hostfile is provided, use the resources from the hostfile
        if self.resources is not None and self.task_type != "serve":
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)
            available_ip = next(iter(self.resources.keys()))
            if self.task_type == "rl":
                available_port = 6379
                self._run_each(
                    "localhost",
                    available_ip,
                    available_port,
                    1,
                    0,
                    0,
                    with_test=with_test,
                    dryrun=dryrun,
                    cur_envs=self.user_envs,
                )
                return None
            available_port = get_free_port()
            if self.task_type == "train":
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
                            None,
                            enable_monitoring,
                        )
                        tasks.append(args)
                    pool.starmap(run_node, tasks)
            else:
                for node_rank, (host, resource_info) in enumerate(self.resources.items()):
                    if node_rank >= nnodes:
                        break
                    nproc_from_hostfile = resource_info["slots"]
                    nproc_from_args = runner_config.get("nproc_per_node", None)
                    nproc_per_node = get_nproc_per_node(
                        nproc_from_hostfile, nproc_from_args, num_visible_devices
                    )
                    master_addr = runner_config.get("master_addr", available_ip)
                    master_port = runner_config.get("master_port", available_port)
                    self._run_each(
                        host,
                        master_addr,
                        master_port,
                        nnodes,
                        node_rank,
                        nproc_per_node,
                        with_test=with_test,
                        dryrun=dryrun,
                        cur_envs=self.user_envs,
                    )
        else:
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

        return None

    def _stop_each(self, host, node_rank):
        if self.task_type == "serve":
            host_stop_script_file = self.backend.generate_stop_script(self.config, host, node_rank)
            logging_config = self.config.logging
        elif self.task_type == "inference":
            host_stop_script_file = self.backend.generate_stop_script(host, node_rank)
            logging_config = self.config.inference.logging
        elif self.task_type == "train":
            host_stop_script_file = self.backend.generate_stop_script(host, node_rank)
            logging_config = self.config.train.system.logging

        if self.task_type == "serve":
            logging_config = self.config.logging
            cmd = f"bash {host_stop_script_file}"
            logger.info(f"Run the local command: {cmd}")
            subprocess.run(
                cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
        else:
            if host != "localhost":
                ssh_port = self.config.experiment.runner.get("ssh_port", 22)
                # Step 1: make sure the scripts_dir exists on the remote host
                run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port)
                # Step 2: copy the host_run_script_file to the remote host
                no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
                if no_shared_fs:
                    run_scp_command(
                        host, host_stop_script_file, logging_config.scripts_dir, ssh_port
                    )
                # Step 3: run the host_run_script_file on the remote host
                run_ssh_command(host, f"bash {host_stop_script_file}", ssh_port)
            else:
                run_local_command(f"bash {host_stop_script_file}")

    def stop(self):
        if self.resources is None or self.task_type == "serve":
            self._stop_each("localhost", 0)
            return

        nnodes = get_nnodes(len(self.resources), self.config.experiment.runner.get("nnodes", None))

        if self.task_type == "train":
            num_processes = min(nnodes, _MAX_CPU_COUNT)
            with multiprocessing.Pool(processes=num_processes) as pool:
                tasks = []
                for node_rank, (host, _) in enumerate(self.resources.items()):
                    if node_rank >= nnodes:
                        break
                    args = (host, node_rank)
                    tasks.append(args)
                pool.starmap(self._stop_each, tasks)
        elif self.task_type == "rl":
            num_processes = min(nnodes, _MAX_CPU_COUNT)
            cmds_config = self.config.experiment.get("cmds", None)
            if cmds_config:
                before_start = cmds_config.get("before_start", "")
            with multiprocessing.Pool(processes=num_processes) as pool:
                tasks = []
                for node_rank, (host, _) in enumerate(self.resources.items()):
                    run_ssh_command(host, f"{before_start};ray stop")
        else:
            for node_rank, (host, _) in enumerate(self.resources.items()):
                if node_rank >= nnodes:
                    break
                self._stop_each(host, node_rank)

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        if self.task_type == "train":
            logging_config = self.config.train.system.logging
        elif self.task_type == "serve":
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
        if self.task_type == "train":
            logging_config = self.config.train.system.logging
        elif self.task_type == "serve":
            logging_config = self.config.logging

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
        if self.task_type == "train":
            logging_config = self.config.train.system.logging
        elif self.task_type == "serve":
            logging_config = self.config.logging
        result = ""
        if self.task_type == "serve":
            try:
                result = run_local_command(f"bash {host_query_script_file}", query=True)
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        else:
            if host != "localhost":
                ssh_port = self.config.experiment.runner.get("ssh_port", 22)
                # Step 1: make sure the scripts_dir exists on the remote host
                run_ssh_command(
                    host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, query=True
                )
                # Step 2: copy the host_run_script_file to the remote host
                no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
                if no_shared_fs:
                    run_scp_command(
                        host, host_query_script_file, logging_config.scripts_dir, ssh_port
                    )
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
        if self.task_type == "train":
            logging_config = self.config.train.system.logging
        elif self.task_type == "serve":
            logging_config = self.config.logging
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
        if self.resources is None or self.task_type == "serve":
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

    def start_monitoring_service(self, interval=10):
        """
        Start independent monitoring service (non-blocking).

        Args:
            interval (int): Monitor interval in seconds

        Returns:
            MonitorService: Monitor service instance
        """
        monitor_service = MonitorService(self.config, self, interval)
        monitor_service.start_monitoring()
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

    def _serve_alive(self):
        engine_args = _get_serve_engine_args(self.config)
        model_name = engine_args.get("served_model_name", None) or engine_args.get("model", None)
        self.port = engine_args.get("port", None)
        self.host = engine_args.get("host", None)
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
        engine_args = _get_serve_engine_args(self.config)

        trust_remote_code = engine_args.get("trust_remote_code", False)

        served_model_name = engine_args.get("served_model_name", None)
        model_name = engine_args.get("model", None)
        self.port = engine_args.get("port", None)
        self.host = engine_args.get("host", None)

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

    def _run_gpu_health_check_on_node(
        self, host, node_rank, master_addr, master_port, nnodes, nproc_per_node
    ):
        """Run GPU health check on a specific node"""
        import subprocess

        # Get parallel configuration
        tp_size = self.config.train.system.get("tensor_model_parallel_size", 1)
        pp_size = self.config.train.system.get("pipeline_model_parallel_size", 1)

        # Build command
        if nnodes > 1 or nproc_per_node > 1:
            # Use torchrun for distributed health check
            import shutil

            TORCHRUN = shutil.which("torchrun")
            cmd = [
                TORCHRUN,
                f"--nnodes={nnodes}",
                f"--nproc_per_node={nproc_per_node}",
                f"--node_rank={node_rank}",  # Use the correct node rank for this node
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                self.gpu_health_check_path,
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
                self.gpu_health_check_path,
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
                # Waiting for the health check to complete and get the actual return code
                result = run_ssh_command(host, cmd_str, ssh_port, query=True, background=False)
                success = result.returncode == 0 if hasattr(result, "returncode") else False
                if not success:
                    logger.error(
                        f"GPU health check failed on {host}: returncode={result.returncode}"
                    )
                    if result.stdout:
                        logger.error(f"stdout: {result.stdout}")
                    if result.stderr:
                        logger.error(f"stderr: {result.stderr}")
                return success
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
        if not os.path.exists(self.gpu_health_check_path):
            logger.error(f"GPU health check script not found at {self.gpu_health_check_path}")
            return False
        runner_config = self.config.experiment.runner

        if self.resources is not None:
            # Multi-node mode: run health check on each node IN PARALLEL
            # This is critical because distributed health checks use NCCL barriers
            # which require all nodes to start simultaneously
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)

            # Get master address and port
            available_ip = next(iter(self.resources.keys()))
            master_addr = runner_config.get("master_addr", available_ip)
            master_port = runner_config.get("master_port", 29500)

            logger.info(f"Running GPU health check across {nnodes} nodes in parallel")

            # Prepare node information for parallel execution
            node_configs = []
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
                node_configs.append(
                    {"host": host, "node_rank": node_rank, "nproc_per_node": nproc_per_node}
                )
                logger.info(f"Preparing node {node_rank} ({host}) with {nproc_per_node} GPUs")

            # Run health checks on all nodes in parallel using threads
            results = {}
            threads = []

            def run_health_check_thread(node_config):
                host = node_config["host"]
                node_rank = node_config["node_rank"]
                nproc_per_node = node_config["nproc_per_node"]
                try:
                    result = self._run_gpu_health_check_on_node(
                        host, node_rank, master_addr, master_port, nnodes, nproc_per_node
                    )
                    results[node_rank] = {"host": host, "success": result}
                except Exception as e:
                    logger.error(
                        f"Exception in health check thread for node {node_rank} ({host}): {e}"
                    )
                    results[node_rank] = {"host": host, "success": False}

            # Start all threads simultaneously
            for node_config in node_configs:
                t = threading.Thread(target=run_health_check_thread, args=(node_config,))
                threads.append(t)
                t.start()

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Check results and report
            all_passed = True
            for node_rank in sorted(results.keys()):
                result_info = results[node_rank]
                host = result_info["host"]
                success = result_info["success"]
                if success:
                    logger.info(f"GPU health check PASSED on node {node_rank} ({host})")
                else:
                    logger.error(f"GPU health check FAILED on node {node_rank} ({host})")
                    all_passed = False

            if all_passed:
                logger.info("GPU health check passed on all nodes")
            else:
                logger.error("GPU health check failed on one or more nodes")
            return all_passed

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
