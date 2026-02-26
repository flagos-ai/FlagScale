#!/usr/bin/env python
"""Backward compatibility wrapper for flagscale.run.

This file is kept for backward compatibility with existing scripts and documentation.
New code should use: python -m flagscale.run

Note: This file has its own @hydra.main decorator to ensure config paths are resolved
relative to the current working directory, not relative to the flagscale package.
"""

import hydra
from omegaconf import DictConfig

from flagscale.run import (
    check_and_reset_deploy_config,
    execute_action,
    get_runner,
    handle_auto_tune,
    validate_task,
)


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    check_and_reset_deploy_config(config)

    task_type = config.experiment.task.get("type", None)
    action = config.action
    validate_task(task_type, action)

    if action == "auto_tune":
        handle_auto_tune(config, task_type)
        return

    runner = get_runner(config, task_type)
    execute_action(runner, action, task_type, config)


if __name__ == "__main__":
    main()
