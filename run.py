import os
import warnings

import hydra

from omegaconf import DictConfig, OmegaConf

from flagscale.logger import logger
from flagscale.runner.autotuner_factory import AutotunerFactory
from flagscale.runner.runner_base import Runner
from flagscale.runner.runner_inference import SSHInferenceRunner
from flagscale.runner.runner_serve import CloudServeRunner, SSHServeRunner
from flagscale.runner.runner_train import CloudTrainRunner, SSHTrainRunner
from flagscale.runner.utils import is_master

# To accommodate the scenario where the before_start field is used to switch to the actual environment during program execution,
# we have placed the import statements inside the function body rather than at the beginning of the file.


FLAGSCALE_USE_V1 = os.environ.get("FLAGSCALE_USE_V1", "1").lower() in ("1", "true")

VALID_TASKS = {"train", "inference", "compress", "serve", "rl"}

LEGACY_RUNNER_MAP = {
    "train": SSHTrainRunner,
    "inference": SSHInferenceRunner,
    "serve": SSHServeRunner,
}

# task_type -> allowed actions
TASK_ACTIONS = {
    "train": {"run", "dryrun", "test", "stop", "query", "auto_tune"},
    "inference": {"run", "dryrun", "test", "stop"},
    "serve": {"run", "test", "stop", "auto_tune"},
    "compress": {"run", "dryrun", "stop"},
    "rl": {"run", "dryrun", "test", "stop"},
}


def check_and_reset_deploy_config(config: DictConfig) -> None:
    if config.experiment.get("deploy", {}):
        OmegaConf.set_struct(config.experiment.runner, False)
        config.experiment.runner.deploy = config.experiment.deploy
        del config.experiment.deploy
        warnings.warn(
            "'config.experiment.deploy' has been moved to 'config.experiment.runner.deploy'. "
            "Support for the old location will be removed in a future release."
        )
        OmegaConf.set_struct(config.experiment.runner, True)


def validate_task(task_type: str, action: str) -> None:
    if task_type not in VALID_TASKS:
        raise ValueError(f"Invalid task_type '{task_type}', must be one of {sorted(VALID_TASKS)}")

    allowed_actions = TASK_ACTIONS[task_type]
    if action not in allowed_actions:
        raise ValueError(
            f"Action '{action}' is not allowed for task_type '{task_type}'. "
            f"Allowed actions: {sorted(allowed_actions)}"
        )


def get_runner(config: DictConfig, task_type: str):
    runner_type = config.experiment.runner.get("type", "ssh")

    if runner_type == "cloud":
        if task_type == "train":
            return CloudTrainRunner(config)
        elif task_type == "serve":
            if FLAGSCALE_USE_V1:
                return Runner(config)
            else:
                return CloudServeRunner(config)
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not supported by cloud runner")

    if FLAGSCALE_USE_V1 and config.experiment.task.get("backend", None) not in {
        "pi0",
        "robotics",
        "lerobot",
    }:
        return Runner(config)

    logger.warning(
        "Using legacy runner, which will be removed in future. Please use new runner instead."
    )

    assert (
        task_type in LEGACY_RUNNER_MAP
    ), f"Task type '{task_type}' is not supported by legacy runner"
    return LEGACY_RUNNER_MAP[task_type](config)


def handle_auto_tune(config: DictConfig, task_type: str) -> None:
    if task_type not in {"serve", "train"}:
        raise NotImplementedError(f"Auto tune is not implemented for task type '{task_type}'")

    # Only one autotuner process for MPI-based runs
    if task_type == "train" and not is_master(config):
        return

    AutoTuner = AutotunerFactory.get_autotuner(task_type)
    AutoTuner(config).tune()


def execute_action(runner, action: str, task_type: str, config: DictConfig) -> None:
    if action == "run":
        if task_type == "train":
            enable_monitoring = config.experiment.runner.get("enable_monitoring", False)
            runner.run(enable_monitoring=enable_monitoring)

            if enable_monitoring:
                logger.info("Monitor service will be started automatically when training begins.")
        else:
            runner.run()
    elif action == "dryrun":
        runner.run(dryrun=True)
    elif action == "test":
        runner.run(with_test=True)
    elif action == "stop":
        runner.stop()
    elif action == "query":
        runner.query()
    else:
        raise ValueError(f"Unknown action '{action}'")


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    check_and_reset_deploy_config(config)

    task_type = config.experiment.task.get("type", None)
    action = config.action
    validate_task(task_type, action)

    # auto_tune invokes the runner internally
    if action == "auto_tune":
        handle_auto_tune(config, task_type)
        return

    runner = get_runner(config, task_type)
    execute_action(runner, action, task_type, config)
    return


if __name__ == "__main__":
    main()
