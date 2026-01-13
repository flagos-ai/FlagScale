import os
import shlex
import subprocess

from flagscale.runner.launcher.launcher_base import LauncherBase
from flagscale.runner.utils import (
    JobStatus,
    get_free_port,
    get_nproc_per_node,
    logger,
    run_local_command,
)


class CloudLauncher(LauncherBase):
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.user_args = self.backend.user_args
        self.user_envs = self.backend.user_envs
        self.user_script = self.backend.user_script
        self.host = None

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

        cmd = shlex.join(export_cmd + ["python"] + [self.user_script] + self.user_args)

        host_run_script_file = self.backend.generate_run_script(
            self.config, host, node_rank, cmd, background=True, with_test=with_test
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(
        self, with_test=False, dryrun=False, monitor=False, interval=10, enable_monitoring=None
    ):
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

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
        )
        self.host = available_addr
        return None

    def _stop_each(self, host, node_rank):
        host_stop_script_file = self.backend.generate_stop_script(self.config, host, node_rank)

        cmd = f"bash {host_stop_script_file}"
        logger.info(f"Run the local command: {cmd}")
        subprocess.run(
            cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

    def stop(self):
        self._stop_each("localhost", 0)
        return

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
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
                "    pid=$(ps aux | grep -E 'run_fs_serve_vllm|run_inference_engine' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file

    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        logging_config = self.config.logging
        result = ""
        try:
            result = run_local_command(f"bash {host_query_script_file}", query=True)
        except Exception as e:
            logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_status(self):
        "Query Job status."
        results = []
        result = self._query_each("localhost", 0)
        results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status

    def query(self, *args, **kwargs):
        return self._query_status()
