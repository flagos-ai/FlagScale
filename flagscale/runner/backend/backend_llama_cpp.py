import json
import os

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.backend.backend_base import BackendBase
from flagscale.runner.utils import get_free_port, logger, parse_hostfile


def _get_args_llamacpp(config: DictConfig):
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


class LlamaCppBackend(BackendBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self.user_script = "flagscale/serve/run_inference_engine.py"
        self._prepare()

    def _prepare(self):
        _update_config_serve(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_args = _get_args_llamacpp(self.config)

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

        logger.info("\n************** LlamaCpp Configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def generate_run_script(self, config, host, node_rank, cmd, background=True, with_test=False):
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
        if cmds_config:
            before_start_cmd = cmds_config.get("before_start", "")
        else:
            before_start_cmd = ""

        cmd += f" --log-dir={logging_config.log_dir}"

        envs = config.experiment.get("envs", {})

        with open(host_run_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("set -x\n")
            f.write("\n")
            f.write(f"{before_start_cmd}\n")
            f.write("\n")

            f.write('if [ -z "$PYTHONPATH" ]; then\n')
            f.write(f"    export PYTHONPATH={root_dir}\n")
            f.write("else\n")
            f.write(f'    export PYTHONPATH="$PYTHONPATH:{root_dir}"\n')
            f.write("fi\n")
            f.write("\n")

            envs_str = " && ".join(
                f"export {key}={value}" for key, value in envs.items() if key != "nodes_envs"
            )
            f.write(f"{envs_str}\n")

            f.write(f"mkdir -p {logging_config.log_dir}\n")
            f.write(f"mkdir -p {logging_config.pids_dir}\n")
            f.write("\n")
            f.write(f"cd {root_dir}\n")
            f.write("\n")
            f.write(f'cmd="{cmd}"\n')
            f.write("\n")
            f.write("echo '=========== launch task (LlamaCpp) ==========='\n")

            if with_test:
                f.write(f'bash -c "$cmd; sync" >> {host_output_file} \n')
            else:
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
        Refactored generic stop logic from old.txt.
        """
        logging_config = config.logging
        host_stop_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
        )
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        cmds_config = config.experiment.get("cmds", None)
        after_stop = cmds_config.get("after_stop", "") if cmds_config else ""
        before_start_cmd = cmds_config.get("before_start", "") if cmds_config else ""

        with open(host_stop_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("set -x\n")
            f.write(f"{before_start_cmd}\n\n")

            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    pkill -P $pid\n")
            f.write("    kill $pid\n")
            f.write("fi\n")

            f.write("pkill -f 'llama-server'\n")

            if after_stop:
                f.write(f"{after_stop}\n")

            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.chmod(host_stop_script_file, 0o755)
        return host_stop_script_file
