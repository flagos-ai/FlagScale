import pytest
from omegaconf import OmegaConf

from flagscale.run import (
    TASK_ACTIONS,
    VALID_TASKS,
    check_and_reset_deploy_config,
    validate_task,
)


class TestValidateTask:
    """Tests for validate_task() function"""

    @pytest.mark.parametrize(
        "task,action",
        [
            ("train", "run"),
            ("train", "dryrun"),
            ("train", "test"),
            ("train", "stop"),
            ("train", "query"),
            ("train", "auto_tune"),
            ("inference", "run"),
            ("inference", "dryrun"),
            ("inference", "test"),
            ("inference", "stop"),
            ("serve", "run"),
            ("serve", "test"),
            ("serve", "stop"),
            ("serve", "auto_tune"),
            ("compress", "run"),
            ("compress", "dryrun"),
            ("compress", "stop"),
            ("rl", "run"),
            ("rl", "dryrun"),
            ("rl", "test"),
            ("rl", "stop"),
        ],
    )
    def test_valid_combinations(self, task, action):
        """Valid task/action pairs should not raise"""
        validate_task(task, action)  # Should not raise

    def test_invalid_task(self):
        """Invalid task raises ValueError"""
        with pytest.raises(ValueError, match="Invalid task_type"):
            validate_task("invalid_task", "run")

    def test_invalid_task_empty_string(self):
        """Empty string task raises ValueError"""
        with pytest.raises(ValueError, match="Invalid task_type"):
            validate_task("", "run")

    def test_invalid_action_for_train(self):
        """Invalid action for train raises ValueError"""
        with pytest.raises(ValueError, match="not allowed for task_type"):
            validate_task("train", "invalid_action")

    def test_invalid_action_query_for_inference(self):
        """Query action not allowed for inference"""
        with pytest.raises(ValueError, match="not allowed for task_type"):
            validate_task("inference", "query")

    def test_invalid_action_query_for_serve(self):
        """Query action not allowed for serve"""
        with pytest.raises(ValueError, match="not allowed for task_type"):
            validate_task("serve", "query")

    def test_invalid_action_dryrun_for_serve(self):
        """Dryrun action not allowed for serve"""
        with pytest.raises(ValueError, match="not allowed for task_type"):
            validate_task("serve", "dryrun")

    def test_invalid_action_auto_tune_for_compress(self):
        """Auto_tune action not allowed for compress"""
        with pytest.raises(ValueError, match="not allowed for task_type"):
            validate_task("compress", "auto_tune")


class TestValidTasks:
    """Tests for VALID_TASKS and TASK_ACTIONS constants"""

    def test_valid_tasks_contains_expected(self):
        """VALID_TASKS contains all expected task types"""
        expected = {"train", "inference", "compress", "serve", "rl"}
        assert VALID_TASKS == expected

    def test_task_actions_keys_match_valid_tasks(self):
        """TASK_ACTIONS keys match VALID_TASKS"""
        assert set(TASK_ACTIONS.keys()) == VALID_TASKS

    def test_all_tasks_have_run_action(self):
        """All tasks support the 'run' action"""
        for task in VALID_TASKS:
            assert "run" in TASK_ACTIONS[task], f"Task '{task}' should support 'run' action"

    def test_all_tasks_have_stop_action(self):
        """All tasks support the 'stop' action"""
        for task in VALID_TASKS:
            assert "stop" in TASK_ACTIONS[task], f"Task '{task}' should support 'stop' action"


class TestCheckAndResetDeployConfig:
    """Tests for check_and_reset_deploy_config() function"""

    def test_migrates_deploy_to_runner(self):
        """Deploy config moves from experiment.deploy to experiment.runner.deploy"""
        config = OmegaConf.create(
            {"experiment": {"deploy": {"key": "value", "nested": {"a": 1}}, "runner": {}}}
        )

        check_and_reset_deploy_config(config)

        assert config.experiment.runner.deploy.key == "value"
        assert config.experiment.runner.deploy.nested.a == 1
        assert "deploy" not in config.experiment

    def test_no_change_when_no_deploy(self):
        """No change when no deploy section exists"""
        config = OmegaConf.create({"experiment": {"runner": {"other_key": "value"}}})

        check_and_reset_deploy_config(config)

        assert "deploy" not in config.experiment
        assert "deploy" not in config.experiment.runner
        assert config.experiment.runner.other_key == "value"

    def test_no_change_when_deploy_is_empty(self):
        """No migration when deploy is empty dict"""
        config = OmegaConf.create({"experiment": {"deploy": {}, "runner": {}}})

        check_and_reset_deploy_config(config)

        # Empty dict is falsy, so no migration should happen
        assert "deploy" in config.experiment or "deploy" not in config.experiment.runner

    def test_preserves_existing_runner_config(self):
        """Migration preserves existing runner configuration"""
        config = OmegaConf.create(
            {
                "experiment": {
                    "deploy": {"deploy_key": "deploy_value"},
                    "runner": {"existing_key": "existing_value"},
                }
            }
        )

        check_and_reset_deploy_config(config)

        assert config.experiment.runner.existing_key == "existing_value"
        assert config.experiment.runner.deploy.deploy_key == "deploy_value"

    def test_warns_about_deprecated_location(self, mocker):
        """Function warns about deprecated deploy location"""
        mock_warn = mocker.patch("warnings.warn")

        config = OmegaConf.create({"experiment": {"deploy": {"key": "value"}, "runner": {}}})

        check_and_reset_deploy_config(config)

        mock_warn.assert_called_once()
        warning_msg = mock_warn.call_args[0][0]
        assert "moved" in warning_msg.lower() or "deprecated" in warning_msg.lower()
