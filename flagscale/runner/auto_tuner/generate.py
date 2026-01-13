import copy
import os

from omegaconf import OmegaConf


class Generator:
    def __init__(self, config):
        self.config = config
        # TODO: Just a temporary solution, need to be configured by user
        if "args_mapping" in config.experiment.auto_tuner:
            self.args_mapping = config.experiment.auto_tuner.args_mapping
        else:
            self.args_mapping = {
                "data_parallel_size": "data_parallel_size",
                "use_distributed_optimizer": "use_distributed_optimizer",
                "tensor_model_parallel_size": "tensor_model_parallel_size",
                "sequence_parallel": "sequence_parallel",
                "pipeline_model_parallel_size": "pipeline_model_parallel_size",
                "num_layers_per_virtual_pipeline_stage": "num_layers_per_virtual_pipeline_stage",
                "recompute_method": "recompute_method",
                "recompute_granularity": "recompute_granularity",
                "recompute_num_layers": "recompute_num_layers",
                "micro_batch_size": "micro_batch_size",
                "context_parallel_size": "context_parallel_size",
                "expert_model_parallel_size": "expert_model_parallel_size",
                "decoder_first_pipeline_num_layers": "decoder_first_pipeline_num_layers",
                "decoder_last_pipeline_num_layers": "decoder_last_pipeline_num_layers",
            }

    def _set_value(self, strategy, config):
        for key, value in self.args_mapping.items():
            if key in ["micro_batch_size"]:
                config.train.model[value] = strategy[key]
            elif key in ["data_parallel_size"]:
                continue
            else:
                if strategy[key] is None:
                    if value in config.train.system:
                        del config.train.system[value]
                    continue
                config.train.system[value] = strategy[key]

    def gen(self, strategy):
        config = copy.deepcopy(self.config)
        self._set_value(strategy, config)

        # Logging interval should be 1
        config.train.system.logging.log_interval = 1

        # Set redirect and tee
        config.experiment.runner.tee = 3
        config.experiment.runner.redirects = 3

        # auto_tune should be true, it will not save ckpt when train ended and report memory every iteration
        config.train.system.auto_tune = True

        # Del lr_warmup_samples and train_samples to run megatron.
        assert "optimizer" in config.train.model
        assert "lr_scheduler" in config.train.model.optimizer
        if "lr_warmup_samples" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_warmup_samples
        # Del lr_decay_samples and train_samples to run megatron.
        if "lr_decay_samples" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_decay_samples
        # Del rampup_batch_size and train_samples to run megatron.
        if "rampup_batch_size" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.rampup_batch_size
        # Del train_samples to run megatron.
        if "train_samples" in config.train.model:
            del config.train.model.train_samples

        # Del checkpoint load
        if "checkpoint" in config.train.system:
            if "load" in config.train.system.checkpoint:
                del config.train.system.checkpoint.load
            if "save_interval" in config.train.system.checkpoint:
                config.train.system.checkpoint.save_interval = 2000

        # Set train_iters of each task
        if "control" in config.experiment.auto_tuner:
            config.train.model.train_iters = config.experiment.auto_tuner.control.get(
                "train_iters", 5
            )
        else:
            config.train.model.train_iters = 5

        # log dir
        config.experiment.exp_dir = os.path.join(
            config.experiment.exp_dir, "auto_tuner", f"task_{strategy['idx']}"
        )

        return config

    def gen_best_task(self, strategy, config):
        self._set_value(strategy, config)
        return config


class ServeGenerator(Generator):
    def __init__(self, config):
        self.config = config
        if "args_mapping" in config.experiment.auto_tuner:
            self.args_mapping = config.experiment.auto_tuner.args_mapping
        else:
            self.args_mapping = {
                "tensor_model_parallel_size": "tensor_parallel_size",
                "pipeline_model_parallel_size": "pipeline_parallel_size",
                "instance": "num_replicas",
                "block_size": "block_size",
                "max_num_batched_tokens": "max_num_batched_tokens",
                "max_num_seqs": "max_num_seqs",
                "swap_space": "swap_space",
                "n_gpu_layers": "n_gpu_layers",
                "parallel": "parallel",
                "threads": "threads",
                "chunked_prefill_size": "chunked_prefill_size",
                "max_prefill_tokens": "max_prefill_tokens",
                "page_size": "page_size",
                "max_running_requests": "max_running_requests",
            }

    def _set_value(self, strategy, config):
        serve_config = config.serve
        model_config = None
        for item in serve_config:
            if item.get("serve_id", None) is not None:
                model_config = item
                break
            else:
                raise ValueError(
                    f"No 'serve_id' configuration found in task config: {serve_config}"
                )

        if not model_config.get("resources", None):
            model_config["resources"] = {}
        if model_config is None:
            raise ValueError(f"No valid configuration found in task config: {serve_config}")

        backend_value = config.get("experiment", {}).get("task", {}).get("backend")
        if backend_value is None:
            engine = model_config.get("engine", None)
        else:
            engine = backend_value

        if "engine" in strategy:
            engine = model_config.engine = strategy["engine"]

        for key, value in self.args_mapping.items():
            if key not in strategy:
                continue
            if key == "instance":
                if strategy[key] is None:
                    if value in model_config.resources:
                        del model_config.resources[value]
                    continue
                if value not in model_config.engine_args_specific:
                    model_config.resources = OmegaConf.merge(
                        model_config.resources, {value: strategy[key]}
                    )
                else:
                    model_config.resources[value] = strategy[key]
            else:
                if strategy[key] is None:
                    if value in model_config.engine_args_specific[engine]:
                        del model_config.engine_args_specific[engine][value]
                    continue
                if value not in model_config.engine_args_specific[engine]:
                    model_config.engine_args_specific[engine] = OmegaConf.merge(
                        model_config.engine_args_specific[engine], {value: strategy[key]}
                    )
                else:
                    model_config.engine_args_specific[engine][value] = strategy[key]
        current_tp = model_config.engine_args_specific[engine].get("tensor_parallel_size", 1)
        current_pp = model_config.engine_args_specific[engine].get("pipeline_parallel_size", 1)
        model_config.resources["num_gpus"] = current_tp * current_pp

        if not config.experiment.get("runner", {}).get("deploy", {}).get("use_fs_serve", True):
            del model_config["resources"]

    def gen(self, strategy):
        config = copy.deepcopy(self.config)
        self._set_value(strategy, config)

        # log dir
        config.experiment.exp_dir = os.path.join(
            config.experiment.exp_dir, "auto_tuner", f"task_{strategy['idx']}"
        )

        return config

    def gen_best_task(self, strategy, config):
        self._set_value(strategy, config)
        return config
