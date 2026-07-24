import collections
import subprocess

import pytest
from omegaconf import OmegaConf

from flagscale.runner import utils


def test_flatten_dict_to_args_handles_nested_lists_bools_and_ignore_keys():
    config = {
        "model_name": "qwen",
        "enable_feature": True,
        "disabled_feature": False,
        "layers": [1, 2],
        "nested": {"inner_value": "x"},
        "ignored": "skip",
    }

    args = utils.flatten_dict_to_args(config, ignore_keys=["ignored"])

    assert args == [
        "--model-name",
        "qwen",
        "--enable-feature",
        "--layers",
        "1",
        "2",
        "--inner-value",
        "x",
    ]


def test_flatten_dict_to_args_can_preserve_underscores():
    args = utils.flatten_dict_to_args({"model_name": "qwen"}, do_dash_replace=False)

    assert args == ["--model_name", "qwen"]


def test_flatten_dict_to_args_verl_handles_config_path_and_append_kwargs():
    config = {
        "config-path": "conf",
        "config-name": "ppo",
        "trainer": {"n_gpus": 8, "enabled": True},
        "append_kargs": {"data.train_files": ["a", "b"]},
    }

    args = utils.flatten_dict_to_args_verl(config)

    assert "--config-path=conf" in args
    assert "--config-name=ppo" in args
    assert "trainer.n_gpus=8" in args
    assert "trainer.enabled=True" in args
    assert '+data.train_files=["a", "b"]' in args


@pytest.mark.parametrize(
    ("hostfile", "args", "expected"),
    [
        (2, None, 2),
        (None, "4", 4),
        (3, "4", 3),
        (8, "2:8", 2),
    ],
)
def test_get_nnodes(hostfile, args, expected):
    assert utils.get_nnodes(hostfile, args) == expected


@pytest.mark.parametrize(
    ("hostfile", "args", "visible", "expected"),
    [
        (8, 4, None, 4),
        (8, 4, 2, 2),
        (8, None, 4, 4),
        (None, "8", 2, 2),
        (None, None, 3, 3),
        (None, None, None, 1),
    ],
)
def test_get_nproc_per_node(hostfile, args, visible, expected):
    assert utils.get_nproc_per_node(hostfile, args, visible) == expected


def test_update_nodes_envs_applies_device_and_node_specific_overrides():
    env_config = {
        "BASE": "1",
        "device_type_specific": {"A100": {"CUDA_VISIBLE_DEVICES": "0,1"}},
        "node_specific": {"worker0": {"NODE_ONLY": "yes"}},
    }

    result = utils.update_nodes_envs(env_config, "worker0", {"type": "A100"})

    assert result == {
        "BASE": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NODE_ONLY": "yes",
    }


def test_update_nodes_envs_supports_omegaconf_and_warns_without_overrides(mocker):
    warn = mocker.patch("flagscale.runner.utils.logger.warning")
    env_config = OmegaConf.create({"BASE": "1"})

    result = utils.update_nodes_envs(env_config, "worker0", {"type": None})

    assert result == {"BASE": "1"}
    warn.assert_called_once()


def test_add_decive_extra_config_merges_matching_device_and_preserves_other_dicts():
    config = {
        "A100": {"batch_size": 8},
        "H100": {"batch_size": 16},
        "common": {"seed": 42},
    }

    assert utils.add_decive_extra_config(config, "A100") == {
        "batch_size": 8,
        "H100": {"batch_size": 16},
        "common": {"seed": 42},
    }


def test_add_decive_extra_config_without_device_returns_full_config():
    config = OmegaConf.create({"common": {"seed": 42}})

    assert utils.add_decive_extra_config(config, None) == {"common": {"seed": 42}}


def test_update_cmd_with_node_specific_config_updates_existing_flags_and_adds_true_flags():
    cmd = "torchrun train.py --use-cache false --enable-old --keep value"
    result = utils.update_cmd_with_node_specific_config(
        cmd,
        {
            "use_cache": "true",
            "enable_old": "false",
            "new_flag": "true",
            "ignored_value": "123",
        },
    )

    assert result == "torchrun train.py --use-cache --keep value --new-flag"


def test_update_cmd_with_node_specific_config_empty_returns_original():
    cmd = "torchrun train.py --foo bar"
    assert utils.update_cmd_with_node_specific_config(cmd, None) == cmd
    assert utils.update_cmd_with_node_specific_config(cmd, {}) == cmd


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1", True),
        ("255.255.255.255", True),
        ("999.1.1.1", False),
        (None, False),
    ],
)
def test_is_ip_addr(value, expected):
    assert utils.is_ip_addr(value) is expected


def test_find_latest_stdout_log_uses_latest_attempt_and_highest_rank(tmp_path):
    root = tmp_path / "logs" / "details" / "host_0_localhost"
    older = root / "20260101_000000" / "default_a" / "attempt_0" / "0"
    latest_low_rank = root / "20260102_000000" / "default_b" / "attempt_1" / "0"
    latest_high_rank = root / "20260102_000000" / "default_b" / "attempt_1" / "7"
    for path in (older, latest_low_rank, latest_high_rank):
        path.mkdir(parents=True)
        (path / "stdout.log").write_text(str(path))

    assert utils.find_latest_stdout_log(str(root)) == str(latest_high_rank / "stdout.log")


def test_find_latest_stdout_log_returns_none_for_missing_path(tmp_path):
    assert utils.find_latest_stdout_log(str(tmp_path / "missing")) is None


def test_get_node0_log_file_handles_shared_and_non_shared_fs():
    logging_config = OmegaConf.create({"log_dir": "/tmp/logs"})
    resources = collections.OrderedDict([("worker0", {"slots": 8})])

    assert utils.get_node0_log_file(logging_config, True, resources) == "/tmp/logs/host.output"
    assert (
        utils.get_node0_log_file(logging_config, False, resources)
        == "/tmp/logs/host_0_worker0.output"
    )
    assert (
        utils.get_node0_log_file(logging_config, False, None) == "/tmp/logs/host_0_localhost.output"
    )


def test_run_local_command_dryrun_does_not_execute(mocker):
    run = mocker.patch("flagscale.runner.utils.subprocess.run")

    assert utils.run_local_command("echo hi", dryrun=True) is None
    run.assert_not_called()


def test_run_local_command_query_returns_completed_process(mocker):
    completed = subprocess.CompletedProcess("cmd", 0, stdout="ok", stderr="")
    run = mocker.patch("flagscale.runner.utils.subprocess.run", return_value=completed)

    assert utils.run_local_command("cmd", query=True) is completed
    run.assert_called_once()


def test_run_local_command_stream_output_exits_on_failure(mocker):
    mocker.patch(
        "flagscale.runner.utils.subprocess.run",
        return_value=subprocess.CompletedProcess("cmd", 7),
    )

    with pytest.raises(SystemExit) as exc_info:
        utils.run_local_command("cmd", stream_output=True)

    assert exc_info.value.code == 7


def test_run_ssh_command_builds_port_and_returns_query_result(mocker):
    completed = subprocess.CompletedProcess("ssh", 0, stdout="ok", stderr="")
    run = mocker.patch("flagscale.runner.utils.subprocess.run", return_value=completed)

    assert utils.run_ssh_command("host", "echo hi", port=2222, query=True) is completed

    ssh_cmd = run.call_args.args[0]
    assert "ssh -f -n -p 2222 host 'echo hi'" == ssh_cmd


def test_run_ssh_command_stream_output_exits_on_failure(mocker):
    mocker.patch(
        "flagscale.runner.utils.subprocess.run",
        return_value=subprocess.CompletedProcess("ssh", 9),
    )

    with pytest.raises(SystemExit) as exc_info:
        utils.run_ssh_command("host", "bad", stream_output=True)

    assert exc_info.value.code == 9


def test_remote_file_helpers_return_values_and_failure(mocker):
    mocker.patch(
        "flagscale.runner.utils.subprocess.run",
        side_effect=[
            subprocess.CompletedProcess("ssh", 0, stdout="123\n", stderr=""),
            subprocess.CompletedProcess("ssh", 0, stdout="456\n", stderr=""),
            subprocess.CalledProcessError(1, "ssh"),
        ],
    )

    assert utils.get_remote_file_mtime("host", "/tmp/file") == 123
    assert utils.get_remote_file_size("host", "/tmp/file") == 456
    assert utils.get_remote_file_mtime("host", "/tmp/missing") == -1
