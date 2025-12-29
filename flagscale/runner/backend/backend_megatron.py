import os

from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend import BackendBase
from flagscale.runner.runner_train import (
    _get_args_megatron,
    _get_args_pi0,
    _get_args_robotics,
    _update_config_train,
)
from flagscale.runner.utils import (
    flatten_dict_to_args,
    flatten_dict_to_args_verl,
    get_free_port,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
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


def _get_serve_engine(config):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    if serve_config and len(serve_config) > 1:
        logger.warning(f"Multiple 'serve' configurations found in task config: {serve_config}")

    engine = serve_config[0].get("engine", None)
    return engine


def _get_serve_engine_args(config, model="vllm_model"):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    engine_args = {}

    for item in serve_config:
        if item.get("serve_id", None) in ("vllm_model", "sglang_model"):
            engine_args = item.get("engine_args", {})
            break
    if not engine_args:
        raise ValueError(f"No 'engine_args' configuration found in task config: {serve_config}")

    return engine_args


def _get_profile_args(config, model="vllm_model"):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")

    profile_args = {}
    for item in serve_config:
        if item.get("serve_id", None) in ("vllm_model", "sglang_model"):
            profile_args = item.get("profile", {})
            break
    return profile_args


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
    new_conf_file = os.path.join(logging_config.scripts_dir, f"serve.yaml")

    # step4: write the new yaml file to `outputs_dir/serve_logs/scripts/serve.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


class MegatronBackend(BackendBase):
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

    def generate_run_script(
        self,
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
            host_output_file = os.path.join(
                logging_config.log_dir, f"host_{node_rank}_{host}.output"
            )
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
            f.write(f"export PYTHONPATH={root_dir}:{megatron_dir}:${{PYTHONPATH}}\n")
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
                f.write(f'  --host "{host}" \\\n')
                f.write(f'  --node-rank {node_rank} \\\n')
                f.write(f'  {"--no-shared-fs" if no_shared_fs else ""} \\\n')
                f.write(f'  --ssh-port {ssh_port} \\\n')
                f.write(f'  --interval 5 \\\n')
                f.write(f'  --enable-log-collection \\\n')
                f.write(f'  --enable-diagnostic \\\n')
                f.write(f'  > /tmp/monitor_output_{node_rank}_{host}.log 2>&1 &\n')
                f.write(
                    f'echo "Monitor service started in background for {host} (node {node_rank})"\n'
                )
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

    def generate_stop_script(self, host, node_rank):
        if getattr(self.config, "train", None):
            logging_config = self.config.train.system.logging
        else:
            logging_config = self.config.inference.system.logging

        host_stop_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        cmds_config = self.config.experiment.get("cmds", None)
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
