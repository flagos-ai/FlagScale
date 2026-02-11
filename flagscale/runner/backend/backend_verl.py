import os

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend.backend_base import BackendBase
from flagscale.runner.utils import flatten_dict_to_args_verl, logger, parse_hostfile


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


class VerlBackend(BackendBase):
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

    def generate_run_script(
        self, config, host, node_rank, cmd, background=True, with_test=False, resources=None
    ):
        system_config = config.system
        logging_config = config.system.logging

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
            if resources is not None:
                available_ip = next(iter(resources.keys()))
                ray_port = config.experiment.runner.get("ray_port", 6379)
                ray_dashboard_port = config.experiment.runner.get("ray_dashboard_port", 8265)
                for node_rank, (host, resource_info) in enumerate(resources.items()):
                    if node_rank == 0:
                        f.write(
                            f"ray start --head --port={ray_port} --dashboard-host=0.0.0.0 --dashboard-port={ray_dashboard_port} --num-gpus={resource_info['slots']}\n"
                        )
                    else:
                        f.write(
                            f'ssh -f -n {host} "{before_start};ray start --address={available_ip}:{ray_port} --num-gpus={resource_info["slots"]}"\n'
                        )

            f.write(f"mkdir -p {system_config.logging.log_dir}\n")
            f.write(f"mkdir -p {system_config.logging.pids_dir}\n")
            f.write("\n")
            f.write(f"cd {root_dir}\n")
            f.write("\n")
            f.write("export PYTHONPATH=${PYTHONPATH}\n")
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

        return host_run_script_file

    def generate_stop_script(self, config, host, node_rank):
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
