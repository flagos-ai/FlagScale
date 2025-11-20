import multiprocessing
import os
import shlex
import time

from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.runner_base import JobStatus, RunnerBase
from flagscale.runner.utils import (
    flatten_dict_to_args_verl,
    get_free_port,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
    run_scp_command,
    run_ssh_command,
)

_MAX_CPU_COUNT = multiprocessing.cpu_count()


def _get_args_verl(config: DictConfig):
    assert config.experiment.task.backend == "verl", "This function only supports verl backend."

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict["rl"]

    new_config_dict = {}
    new_config_dict.update(config_dict)

    # Flatten the dictionary to a list of arguments
    args = flatten_dict_to_args_verl(new_config_dict, pre_str="")

    return args


def _update_config_rl(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)
    if config.get("system", None) is None:
        config.system = DictConfig({})

    if config.system.get("logging", None) is None:
        config.system.logging = DictConfig({})

    log_dir = (
        os.path.abspath(config.system.logging.log_dir)
        if config.system.logging.get("log_dir", None)
        else os.path.join(exp_dir, "logs")
    )
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.system.logging.log_dir = log_dir
    config.system.logging.scripts_dir = scripts_dir
    config.system.logging.pids_dir = pids_dir

    OmegaConf.set_struct(config, True)


def _generate_run_script_rl(
    config, host, node_rank, cmd, background=True, with_test=False, resources=None
):
    system_config = config.system
    logging_config = config.system.logging

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

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    verl_dir = os.path.join(root_dir, "third_party", "verl")
    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        before_start = cmds_config.get("before_start", "")
    else:
        before_start = ""
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"{before_start}\n")
        if resources is not None:
            available_ip = list(resources.keys())[0]
            ray_port = config.experiment.runner.get("ray_port", 6379)
            ray_dashboard_port = config.experiment.runner.get("ray_dashboard_port", 8265)
            for node_rank, (host, resource_info) in enumerate(resources.items()):
                if node_rank == 0:
                    f.write(
                        f'ray start --head --port={ray_port} --dashboard-host=0.0.0.0 --dashboard-port={ray_dashboard_port} --num-gpus={resource_info["slots"]}\n'
                    )
                else:
                    f.write(
                        f'ssh -f -n {host} "{before_start};ray start --address={available_ip}:{ray_port} --num-gpus={resource_info["slots"]}"\n'
                    )

        f.write(f"mkdir -p {system_config.logging.log_dir}\n")
        f.write(f"mkdir -p {system_config.logging.pids_dir}\n")
        f.write(f"\n")
        f.write(f"cd {root_dir}\n")
        f.write(f"\n")
        f.write(f"export PYTHONPATH={verl_dir}:{root_dir}:${{PYTHONPATH}}\n")
        f.write(f"\n")
        f.write(f'cmd="{cmd}"\n')
        f.write(f"\n")
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

    return host_run_script_file


def _generate_stop_script_rl(config, host, node_rank):
    if getattr(config, "rl", None):
        logging_config = config.system.logging
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


class SSHRLRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "rl", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        _update_config_rl(self.config)
        self.user_args = _get_args_verl(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_script = self.config.experiment.task.entrypoint
        self.resources = parse_hostfile(self.config.experiment.runner.get("hostfile", None))
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
        device_type=None,
        with_test=False,
        dryrun=False,
        cur_envs=None,
    ):
        export_cmd = []

        for k, v in cur_envs.items():
            export_cmd += [f"{k}={v}"]
        ray_cmd = []
        if self.resources is not None:
            runtime_env = self.config.experiment.runner.get(
                "runtime_env", 'third_party/verl/verl/trainer/runtime_env.yaml'
            )
            ray_dashboard_port = self.config.experiment.runner.get("ray_dashboard_port", 8265)
            ray_cmd = [
                'ray',
                'job',
                'submit',
                f'--address=http://{host}:{ray_dashboard_port}',
                f'--runtime-env={runtime_env}',
                '--no-wait',
                '--',
            ]
        cmd = shlex.join(
            ray_cmd + export_cmd + ['python3', '-m'] + [self.user_script] + self.user_args
        )
        host_run_script_file = _generate_run_script_rl(
            self.config,
            host,
            node_rank,
            cmd,
            background=True,
            with_test=with_test,
            resources=self.resources,
        )
        if host != "localhost":
            logging_config = self.config.system.logging
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

    def run(self, with_test=False, dryrun=False, monitor=False, interval=10):

        num_visible_devices = None
        runner_config = self.config.experiment.runner

        # If hostfile is provided, use the resources from the hostfile
        if self.resources is not None:
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)
            available_ip = list(self.resources.keys())[0]
            available_port = 6379
            num_processes = min(nnodes, _MAX_CPU_COUNT)
            self._run_each(
                'localhost',
                available_ip,
                available_port,
                1,
                0,
                0,
                with_test=with_test,
                dryrun=dryrun,
                cur_envs=self.user_envs,
            )

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

    def _stop_each(self, host, node_rank):
        host_stop_script_file = _generate_stop_script_rl(self.config, host, node_rank)
        logging_config = self.config.system.logging

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
        cmds_config = self.config.experiment.get("cmds", None)
        if cmds_config:
            before_start = cmds_config.get("before_start", "")
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = []
            for node_rank, (host, _) in enumerate(self.resources.items()):
                run_ssh_command(host, f"{before_start};ray stop")
