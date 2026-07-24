import os

from omegaconf import OmegaConf

from flagscale.runner import runner_inference, runner_train


def make_train_config(tmp_path, runner_extra=None, data=None):
    runner = {
        "type": "ssh",
        "backend": "torchrun",
        "hostfile": "/tmp/hostfile",
        "ssh_port": 2222,
        "master_addr": "ignored-master",
        "master_port": 23456,
        "enable_monitoring": True,
        "enable_gpu_health_check": True,
        "nsys_bin_path": "/opt/nsys",
        "nsys_rep_file_path": "/tmp/nsys",
        "deploy": {"ignored": True},
        "tee": "1",
    }
    if runner_extra:
        runner.update(runner_extra)

    config = {
        "experiment": {
            "exp_dir": str(tmp_path / "exp"),
            "task": {"type": "train", "backend": "megatron"},
            "runner": runner,
        },
        "train": {
            "system": {
                "checkpoint": {},
                "logging": {},
                "micro_batch_size": 2,
            },
            "model": {"num_layers": 2, "use_flash_attn": True},
            "data": data or {},
        },
    }
    return OmegaConf.create(config)


def test_get_args_megatron_flattens_train_sections_and_ignores_log_paths(tmp_path):
    config = make_train_config(tmp_path)
    config.train.system.logging = {
        "log_dir": "skip",
        "details_dir": "skip",
        "scripts_dir": "skip",
        "pids_dir": "skip",
    }

    args = runner_train._get_args_megatron(config)

    assert "--micro-batch-size" in args
    assert "2" in args
    assert "--num-layers" in args
    assert "--use-flash-attn" in args
    assert "--log-dir" not in args
    assert "--details-dir" not in args


def test_update_config_train_sets_default_dirs_and_resolves_tokenizer_files(tmp_path):
    vocab = tmp_path / "vocab.json"
    merge = tmp_path / "merge.txt"
    vocab.write_text("{}")
    merge.write_text("#merge")
    config = make_train_config(
        tmp_path,
        data={"tokenizer": {"vocab_file": str(vocab), "merge_file": str(merge)}},
    )

    runner_train._update_config_train(config)

    exp_dir = str(tmp_path / "exp")
    assert os.path.isdir(exp_dir)
    assert config.train.system.checkpoint.save == os.path.join(exp_dir, "checkpoints")
    assert config.train.system.checkpoint.load == os.path.join(exp_dir, "checkpoints")
    assert config.train.system.logging.log_dir == os.path.join(exp_dir, "logs")
    assert config.train.system.logging.details_dir == os.path.join(exp_dir, "logs", "details")
    assert config.train.system.logging.tensorboard_dir == os.path.join(exp_dir, "tensorboard")
    assert config.train.system.logging.wandb_save_dir == os.path.join(exp_dir, "wandb")
    assert config.train.data.tokenizer.vocab_file == str(vocab.resolve())
    assert config.train.data.tokenizer.merge_file == str(merge.resolve())


def test_get_runner_cmd_train_filters_launcher_only_options_and_sets_distributed_args(
    tmp_path,
):
    config = make_train_config(tmp_path, runner_extra={"rdzv_id": "run-1"})
    config.train.system.logging.details_dir = str(tmp_path / "details")

    cmd = runner_train._get_runner_cmd_train("worker0", "10.0.0.1", 29500, 2, 1, 4, config)

    joined = " ".join(cmd)
    assert cmd[0] == "torchrun"
    assert "--nnodes 2" in joined
    assert "--node_rank 1" in joined
    assert "--nproc_per_node 4" in joined
    assert "--rdzv_endpoint 10.0.0.1:29500" in joined
    assert "host_1_worker0" in joined
    assert "--type" not in cmd
    assert "--hostfile" not in cmd
    assert "--ssh_port" not in cmd
    assert "--enable_monitoring" not in cmd


def test_get_runner_cmd_train_per_node_task_overrides_topology(tmp_path):
    config = make_train_config(tmp_path, runner_extra={"per_node_task": True})
    config.train.system.logging.details_dir = str(tmp_path / "details")

    cmd = runner_train._get_runner_cmd_train("worker0", "10.0.0.1", 29500, 8, 7, 4, config)
    joined = " ".join(cmd)

    assert "--nnodes 1" in joined
    assert "--node_rank 0" in joined
    assert "--rdzv_endpoint localhost:29500" in joined


def make_inference_config(tmp_path):
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "infer-exp"),
                "task": {"type": "inference", "backend": "vllm"},
                "runner": {"type": "ssh"},
            },
            "inference": {
                "logging": {"log_dir": "", "scripts_dir": "", "pids_dir": ""},
                "model": "qwen",
                "server": {"port": 8000},
            },
        }
    )


def test_update_config_inference_sets_log_dirs(tmp_path):
    config = make_inference_config(tmp_path)

    runner_inference._update_config_inference(config)

    exp_dir = str(tmp_path / "infer-exp")
    assert config.inference.logging.log_dir == os.path.join(exp_dir, "inference_logs")
    assert config.inference.logging.scripts_dir == os.path.join(
        exp_dir, "inference_logs", "scripts"
    )
    assert config.inference.logging.pids_dir == os.path.join(exp_dir, "inference_logs", "pids")
    assert os.path.isdir(config.inference.logging.scripts_dir)


def test_get_args_vllm_writes_sanitized_yaml(tmp_path):
    config = make_inference_config(tmp_path)
    runner_inference._update_config_inference(config)

    args = runner_inference._get_args_vllm(config)

    assert len(args) == 1
    assert args[0].startswith("--config-path=")
    config_path = args[0].split("=", 1)[1]
    assert os.path.exists(config_path)
    written = OmegaConf.load(config_path)
    assert written.model == "qwen"
    assert written.server.port == 8000
    assert "logging" not in written
