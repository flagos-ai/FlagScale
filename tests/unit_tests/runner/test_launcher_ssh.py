import subprocess
import sys
import types

import pytest
from omegaconf import OmegaConf

from flagscale.runner.launcher import launcher_ssh
from flagscale.runner.launcher.launcher_ssh import SshLauncher
from flagscale.runner.utils import JobStatus


class FakeBackend:
    def __init__(self, tmp_path):
        self.user_args = ["--use-cache", "--hetero-current-device-type", "old"]
        self.user_envs = {"CUDA_VISIBLE_DEVICES": "0,1"}
        self.user_script = "train.py"
        self.device_type_specific = {"A100": {"build_dir": str(tmp_path), "new_flag": "true"}}
        self.node_specific = {"localhost": {"use_cache": "false"}}
        self.generated = []

    def generate_run_script(self, config, host, node_rank, cmd, **kwargs):
        self.generated.append(
            {
                "config": config,
                "host": host,
                "node_rank": node_rank,
                "cmd": cmd,
                "kwargs": kwargs,
            }
        )
        return f"/tmp/host_{node_rank}_{host}_run.sh"

    def generate_stop_script(self, host, node_rank):
        return f"/tmp/host_{node_rank}_{host}_stop.sh"


def make_train_config(tmp_path, hostfile=None, no_shared_fs=False):
    return OmegaConf.create(
        {
            "experiment": {
                "task": {"type": "train", "backend": "native"},
                "runner": {
                    "type": "ssh",
                    "hostfile": hostfile,
                    "ssh_port": 2222,
                    "no_shared_fs": no_shared_fs,
                    "rdzv_id": "test-run",
                    "tee": "1",
                },
            },
            "train": {
                "system": {
                    "logging": {
                        "log_dir": str(tmp_path / "logs"),
                        "details_dir": str(tmp_path / "logs" / "details"),
                        "scripts_dir": str(tmp_path / "logs" / "scripts"),
                        "pids_dir": str(tmp_path / "logs" / "pids"),
                    }
                }
            },
        }
    )


def test_get_serve_engine_args_and_profile_select_first_serve_id():
    config = OmegaConf.create(
        {
            "serve": [
                {"engine_args": {"ignored": True}},
                {
                    "serve_id": "svc",
                    "engine_args": {"model": "qwen", "port": 8000},
                    "profile": {"num_prompts": 4},
                },
            ]
        }
    )

    assert launcher_ssh._get_serve_engine_args(config) == {
        "model": "qwen",
        "port": 8000,
    }
    assert launcher_ssh._get_profile_args(config) == {"num_prompts": 4}


def test_get_serve_engine_args_rejects_missing_config():
    with pytest.raises(ValueError, match="No 'serve' configuration"):
        launcher_ssh._get_serve_engine_args(OmegaConf.create({}))

    with pytest.raises(ValueError, match="No 'engine_args'"):
        launcher_ssh._get_serve_engine_args(
            OmegaConf.create({"serve": [{"serve_id": "svc", "profile": {}}]})
        )


def test_get_runner_cmd_train_filters_launcher_and_profiling_options(tmp_path):
    config = make_train_config(tmp_path)
    config.experiment.runner.nsys_bin_path = "/opt/nsys"
    config.experiment.runner.nsys_rep_file_path = "/tmp/nsys"
    config.experiment.runner.enable_gpu_health_check = True
    config.experiment.runner.deploy = {"ignored": True}

    cmd = launcher_ssh._get_runner_cmd_train("worker0", "10.0.0.1", 29500, 2, 1, 4, config)

    joined = " ".join(cmd)
    assert cmd[0] == "torchrun"
    assert "--nnodes 2" in joined
    assert "--node_rank 1" in joined
    assert "--nproc_per_node 4" in joined
    assert "--rdzv_endpoint 10.0.0.1:29500" in joined
    assert "--nsys_bin_path" not in cmd
    assert "--nsys_rep_file_path" not in cmd
    assert "--enable_gpu_health_check" not in cmd
    assert "--deploy" not in cmd


def test_run_each_train_generates_local_script_with_env_and_node_specific_updates(tmp_path, mocker):
    config = make_train_config(tmp_path)
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)
    run_local = mocker.patch("flagscale.runner.launcher.launcher_ssh.run_local_command")

    launcher._run_each(
        "localhost",
        "127.0.0.1",
        29500,
        1,
        0,
        2,
        device_type="A100",
        background=False,
        dryrun=True,
        cur_envs={"CUDA_VISIBLE_DEVICES": "0,1", "nodes_envs": "ignored"},
        enable_monitoring=True,
    )

    assert (
        backend.user_args[-2:] == ["--hetero-current-device-type", "A100"]
        or backend.user_args[backend.user_args.index("--hetero-current-device-type") + 1] == "A100"
    )
    generated = backend.generated[-1]
    cmd = generated["cmd"]
    assert "CUDA_VISIBLE_DEVICES=0,1" in cmd
    assert "nodes_envs" not in cmd
    assert "torchrun" in cmd
    assert "--hetero-current-device-type A100" in cmd
    assert "--use-cache" not in cmd
    assert "--new-flag" in cmd
    assert generated["kwargs"]["background"] is False
    assert generated["kwargs"]["pkg_dir"] == str(tmp_path)
    assert generated["kwargs"]["enable_monitoring"] is True
    run_local.assert_called_once_with("bash /tmp/host_0_localhost_run.sh", True, stream_output=True)


def test_run_each_remote_copies_script_when_no_shared_fs(tmp_path, mocker):
    config = make_train_config(tmp_path, no_shared_fs=True)
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)
    run_ssh = mocker.patch("flagscale.runner.launcher.launcher_ssh.run_ssh_command")
    run_scp = mocker.patch("flagscale.runner.launcher.launcher_ssh.run_scp_command")

    launcher._run_each(
        "worker0",
        "10.0.0.1",
        29500,
        1,
        0,
        2,
        device_type="A100",
        background=True,
        dryrun=True,
        cur_envs={"CUDA_VISIBLE_DEVICES": "0,1"},
    )

    assert run_ssh.call_args_list[0].args == (
        "worker0",
        f"mkdir -p {config.train.system.logging.scripts_dir}",
        2222,
        True,
    )
    run_scp.assert_called_once_with(
        "worker0",
        "/tmp/host_0_worker0_run.sh",
        config.train.system.logging.scripts_dir,
        2222,
        True,
    )
    assert run_ssh.call_args_list[1].args[:4] == (
        "worker0",
        "bash /tmp/host_0_worker0_run.sh",
        2222,
        True,
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["R", "S"], JobStatus.RUNNING),
        (["", "Z"], JobStatus.COMPLETED_OR_IDLE),
        (["R", ""], JobStatus.TRANSITIONAL),
    ],
)
def test_query_status_classifies_node_results(statuses, expected, mocker):
    launcher = object.__new__(SshLauncher)
    launcher.resources = {"host0": {}, "host1": {}}
    launcher.task_type = "train"
    mocker.patch.object(launcher, "_query_each", side_effect=statuses)

    assert launcher._query_status() == expected


def test_query_each_returns_stdout_for_local_command(tmp_path, mocker):
    config = make_train_config(tmp_path)
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)
    mocker.patch.object(launcher, "_generate_query_script", return_value="/tmp/query.sh")
    mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.run_local_command",
        return_value=subprocess.CompletedProcess("cmd", 0, stdout="R\n", stderr=""),
    )

    assert launcher._query_each("localhost", 0) == "R"


def test_run_uses_multiprocessing_pool_for_multinode_train(tmp_path, mocker):
    resources = {
        "worker0": {"slots": 8, "type": "A100"},
        "worker1": {"slots": 4, "type": "A100"},
    }
    parse = mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.parse_hostfile",
        return_value=resources,
    )
    config = make_train_config(tmp_path, hostfile="/tmp/hostfile")
    config.experiment.runner.nnodes = 2
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)
    get_free_port = mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.get_free_port", return_value=23456
    )

    class FakePool:
        instances = []

        def __init__(self, processes):
            self.processes = processes
            self.calls = []
            FakePool.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, tasks):
            self.calls.append((func, tasks))

    mocker.patch("flagscale.runner.launcher.launcher_ssh._MAX_CPU_COUNT", 16)
    mocker.patch("flagscale.runner.launcher.launcher_ssh.multiprocessing.Pool", FakePool)

    launcher.run(background=True, dryrun=True, enable_monitoring=True)

    parse.assert_called_once_with("/tmp/hostfile")
    get_free_port.assert_called_once()
    pool = FakePool.instances[-1]
    assert pool.processes == 2
    func, tasks = pool.calls[-1]
    assert func is launcher_ssh.run_node
    assert len(tasks) == 2
    assert tasks[0][0] == launcher._run_each
    assert tasks[0][2] == "worker0"
    assert tasks[0][6] == 2
    assert tasks[0][7] == "worker0"
    assert tasks[0][8] == 23456
    assert tasks[0][12] is True
    assert tasks[1][2] == "worker1"


def test_start_monitoring_service_starts_and_returns_service(tmp_path, mocker):
    config = make_train_config(tmp_path)
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)

    class FakeMonitorService:
        def __init__(self, config_arg, runner_arg, interval):
            self.config = config_arg
            self.runner = runner_arg
            self.interval = interval
            self.started = False

        def start_monitoring(self):
            self.started = True

    mocker.patch("flagscale.runner.launcher.launcher_ssh.MonitorService", FakeMonitorService)

    service = launcher.start_monitoring_service(interval=3)

    assert service.config is config
    assert service.runner is launcher
    assert service.interval == 3
    assert service.started is True


def test_run_gpu_health_check_on_node_local_single_process(tmp_path, mocker):
    config = make_train_config(tmp_path)
    config.train.system.tensor_model_parallel_size = 2
    config.train.system.pipeline_model_parallel_size = 1
    launcher = SshLauncher(config, FakeBackend(tmp_path))
    run = mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.subprocess.run",
        return_value=subprocess.CompletedProcess("cmd", 0),
    )

    assert launcher._run_gpu_health_check_on_node("localhost", 0, "localhost", 29500, 1, 1)

    cmd = run.call_args.args[0]
    assert cmd[:2] == ["python", launcher.gpu_health_check_path]
    assert "--tensor-model-parallel-size" in cmd
    assert "2" in cmd
    assert "--distributed-backend" in cmd
    assert "nccl" in cmd


def test_run_gpu_health_check_on_node_remote_distributed_uses_ssh(tmp_path, mocker):
    config = make_train_config(tmp_path)
    launcher = SshLauncher(config, FakeBackend(tmp_path))
    mocker.patch("shutil.which", return_value="/usr/bin/torchrun")
    run_ssh = mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.run_ssh_command",
        return_value=subprocess.CompletedProcess("ssh", 0, stdout="ok", stderr=""),
    )

    assert launcher._run_gpu_health_check_on_node("worker0", 1, "10.0.0.1", 29500, 2, 4)

    args = run_ssh.call_args.args
    assert args[0] == "worker0"
    assert "/usr/bin/torchrun" in args[1]
    assert "--nnodes=2" in args[1]
    assert "--node_rank=1" in args[1]
    assert "--nproc_per_node=4" in args[1]
    assert args[2] == 2222
    assert run_ssh.call_args.kwargs == {"query": True, "background": False}


def test_run_gpu_health_check_returns_false_when_script_missing(tmp_path, mocker):
    launcher = SshLauncher(make_train_config(tmp_path), FakeBackend(tmp_path))
    mocker.patch("flagscale.runner.launcher.launcher_ssh.os.path.exists", return_value=False)

    assert launcher._run_gpu_health_check() is False


def test_run_gpu_health_check_single_node_uses_visible_device_count(tmp_path, mocker):
    config = make_train_config(tmp_path)
    config.experiment.runner.nproc_per_node = 8
    backend = FakeBackend(tmp_path)
    backend.user_envs = {"CUDA_VISIBLE_DEVICES": "0,1"}
    launcher = SshLauncher(config, backend)
    mocker.patch("flagscale.runner.launcher.launcher_ssh.os.path.exists", return_value=True)
    run_node = mocker.patch.object(launcher, "_run_gpu_health_check_on_node", return_value=True)

    assert launcher._run_gpu_health_check() is True

    run_node.assert_called_once_with("localhost", 0, "localhost", 29500, 1, 2)


def test_run_gpu_health_check_multinode_aggregates_thread_results(tmp_path, mocker):
    resources = {
        "worker0": {"slots": 8, "type": "A100"},
        "worker1": {"slots": 4, "type": "A100"},
    }
    config = make_train_config(tmp_path)
    config.experiment.runner.nnodes = 2
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)
    launcher.resources = resources
    mocker.patch("flagscale.runner.launcher.launcher_ssh.os.path.exists", return_value=True)
    run_node = mocker.patch.object(
        launcher, "_run_gpu_health_check_on_node", side_effect=[True, False]
    )

    assert launcher._run_gpu_health_check() is False

    assert run_node.call_args_list[0].args == ("worker0", 0, "worker0", 29500, 2, 8)
    assert run_node.call_args_list[1].args == ("worker1", 1, "worker0", 29500, 2, 4)


def test_run_aborts_when_gpu_health_check_fails(tmp_path, mocker):
    config = make_train_config(tmp_path)
    backend = FakeBackend(tmp_path)
    launcher = SshLauncher(config, backend)
    health_check = mocker.patch.object(launcher, "_run_gpu_health_check", return_value=False)
    run_each = mocker.patch.object(launcher, "_run_each")

    assert launcher.run(dryrun=True, enable_gpu_health_check=True) is None

    health_check.assert_called_once()
    run_each.assert_not_called()


def test_profile_serve_uses_tokenizer_dummy_inputs_and_benchmark(monkeypatch, mocker):
    tokenizer_module = types.ModuleType("vllm.transformers_utils.tokenizer")
    tokenizer = object()
    tokenizer_module.get_tokenizer = mocker.Mock(return_value=tokenizer)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.transformers_utils",
        types.ModuleType("vllm.transformers_utils"),
    )
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer", tokenizer_module)

    dummy_inputs = [("prompt", 1, 2)]
    dummy_random_input = mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.dummy_random_input",
        return_value=dummy_inputs,
    )

    async def fake_benchmark(*args, **kwargs):
        return {"e2el": 1.23, "args": args, "kwargs": kwargs}

    benchmark = mocker.patch(
        "flagscale.runner.launcher.launcher_ssh.benchmark",
        side_effect=fake_benchmark,
    )
    launcher = object.__new__(SshLauncher)
    launcher.config = OmegaConf.create(
        {
            "serve": [
                {
                    "serve_id": "svc",
                    "engine_args": {
                        "model": "qwen",
                        "served_model_name": "alias",
                        "host": "127.0.0.1",
                        "port": 8000,
                        "trust_remote_code": True,
                    },
                    "profile": {
                        "prefix_len": 1,
                        "input_len": 16,
                        "output_len": 8,
                        "num_prompts": 2,
                        "range_ratio": 0.3,
                    },
                }
            ]
        }
    )

    result = launcher._profile_serve()

    tokenizer_module.get_tokenizer.assert_called_once_with(
        "qwen", tokenizer_mode="auto", trust_remote_code=True
    )
    dummy_random_input.assert_called_once_with(
        tokenizer=tokenizer,
        prefix_len=1,
        input_len=16,
        output_len=8,
        num_prompts=2,
        range_ratio=0.3,
    )
    assert benchmark.call_args.args[0] == "http://127.0.0.1:8000/v1/chat/completions"
    assert benchmark.call_args.kwargs["model"] == "qwen"
    assert benchmark.call_args.kwargs["served_model_name"] == "alias"
    assert benchmark.call_args.kwargs["input_requests"] == dummy_inputs
    assert result["e2el"] == 1.23
