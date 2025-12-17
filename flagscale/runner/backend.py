import importlib
import os
import shlex
import sys

from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.runner_base import RunnerBase
from flagscale.runner.utils import (
    get_free_port,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
    run_scp_command,
    run_ssh_command,
)


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
    new_conf_file = os.path.join(logging_config.scripts_dir, f"inference.yaml")

    # step4: write the new yaml file to `outputs_dir/inference_logs/scripts/inference.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


def _update_config_inference(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if config.get("logging", None) is None:
        config.inference.logging = DictConfig({})

    log_dir = os.path.join(exp_dir, f"inference_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.inference.logging.log_dir = log_dir
    config.inference.logging.scripts_dir = scripts_dir
    config.inference.logging.pids_dir = pids_dir

    os.makedirs(config.inference.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


class BackendBase(ABC):

    @abstractmethod
    def generate_run_script(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_stop_script(self, *args, **kwargs):
        raise NotImplementedError


class MegatronBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class TorchrunBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class VllmBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "inference", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        _update_config_inference(self.config)
        self.user_args = _get_args_vllm(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_script = self.config.experiment.task.entrypoint
        self.resources = parse_hostfile(self.config.experiment.runner.get("hostfile", None))
        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def generate_run_script(self, config, host, node_rank, cmd, background=True, with_test=False):
        logging_config = config.inference.logging

        no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
        if no_shared_fs:
            host_output_file = os.path.join(logging_config.log_dir, f"host.output")
        else:
            host_output_file = os.path.join(
                logging_config.log_dir, f"host_{node_rank}_{host}.output"
            )
        host_run_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
        )
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            f.write(f"\n")
            f.write(f"cd {root_dir}\n")
            f.write(f"\n")
            f.write(f"export PYTHONPATH={root_dir}:${{PYTHONPATH}}\n")
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


class SglangBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class LlamallamaCppBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass


class CustomBackend(BackendBase):
    def generate_run_script(self, *args, **kwargs):
        pass

    def generate_stop_script(self, *args, **kwargs):
        pass
