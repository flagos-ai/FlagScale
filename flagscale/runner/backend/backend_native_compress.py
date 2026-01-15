import os
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend.backend_base import BackendBase
from flagscale.runner.utils import logger, parse_hostfile


def _get_args_llmcompressor(config: DictConfig):
    # see the following link for more details
    # https://github.com/facebookresearch/hydra/discussions/2750
    # OmegaConf.set_struct(config, False)

    hydra_config = HydraConfig.get()
    output_dir = hydra_config.runtime.output_dir
    output_subdir = hydra_config.output_subdir
    config_path = os.path.join(output_dir, f"{output_subdir}/config.yaml")
    config_path = hydra.utils.to_absolute_path(config_path)

    args = []
    args.append(f"--config-path={config_path}")

    return args


def _update_config_compress(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)
    config = config.compress.system

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

    log_dir = os.path.join(exp_dir, "compress_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.logging.log_dir = log_dir
    config.logging.scripts_dir = scripts_dir
    config.logging.pids_dir = pids_dir
    config.logging.tensorboard_dir = tensorboard_dir
    config.logging.wandb_save_dir = wandb_dir

    OmegaConf.set_struct(config, True)


class NativeCompressBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "compress", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        _update_config_compress(self.config)
        self.user_args = _get_args_llmcompressor(self.config)
        self.rdzv_id = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        self.user_envs = self.config.experiment.get("envs", {})
        self.cur_envs = None  # current node envs
        self.user_script = self.config.experiment.task.entrypoint
        self.resources = parse_hostfile(self.config.experiment.runner.get("hostfile", None))
        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def generate_run_script(self, config, host, node_rank, cmd, background=True, with_test=False):
        system_config = config.compress.system
        logging_config = config.compress.system.logging

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
        compress_dir = os.path.join(root_dir, "compress")
        ### set megatron dir for dataset
        megtron_dir = os.path.join(root_dir, "megatron")
        cmds_config = config.experiment.get("cmds", None)
        if cmds_config:
            before_start = cmds_config.get("before_start", "")
        else:
            before_start = ""
        with open(host_run_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"{before_start}\n")
            f.write(f"mkdir -p {system_config.save_dir}\n")
            f.write(f"mkdir -p {system_config.logging.log_dir}\n")
            f.write(f"mkdir -p {system_config.logging.pids_dir}\n")
            f.write(f"mkdir -p {system_config.logging.tensorboard_dir}\n")
            f.write(f"mkdir -p {system_config.logging.wandb_save_dir}\n")
            f.write("\n")
            f.write(f"cd {root_dir}\n")
            f.write("\n")
            f.write(f"export PYTHONPATH={compress_dir}:{megtron_dir}:{root_dir}\n")
            f.write("\n")
            f.write(f'cmd="{cmd}"\n')
            f.write("\n")
            if with_test:
                f.write('bash -c "$cmd; sync" \n')
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

        return host_stop_script_file
