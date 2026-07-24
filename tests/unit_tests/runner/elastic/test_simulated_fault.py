from unittest.mock import patch

import pytest

from flagscale.runner.elastic import simulated_fault


def test_simulated_fault_loop_writes_explicit_error_keys(tmp_path):
    log_file = tmp_path / "output.log"

    with patch("flagscale.runner.elastic.simulated_fault.time.sleep") as sleep:
        simulated_fault.simulated_fault_loop(
            log_file=str(log_file),
            error_keys=["cuda error", "permission denied"],
            interval=3,
            iterations=2,
            mode="a",
        )

    content = log_file.read_text()
    assert content.count("--- Simulated log at") == 2
    assert content.count("cuda error") == 2
    assert content.count("permission denied") == 2
    assert sleep.call_count == 2
    sleep.assert_called_with(3)


def test_simulated_fault_loop_uses_random_error_when_no_errors(tmp_path):
    log_file = tmp_path / "output.log"

    with (
        patch(
            "flagscale.runner.elastic.simulated_fault.random.choice",
            return_value="fatal error",
        ) as choice,
        patch("flagscale.runner.elastic.simulated_fault.time.sleep"),
    ):
        simulated_fault.simulated_fault_loop(
            log_file=str(log_file), error_keys=None, interval=0, iterations=1, mode="a"
        )

    assert "fatal error" in log_file.read_text()
    choice.assert_called_once_with(simulated_fault.error_keys_list)


def test_simulated_fault_loop_rejects_non_list_error_keys(tmp_path):
    with pytest.raises(ValueError, match="error_keys must be a list"):
        simulated_fault.simulated_fault_loop(
            log_file=str(tmp_path / "output.log"), error_keys="cuda error"
        )


def test_main_parses_arguments_and_invokes_loop(tmp_path):
    log_file = tmp_path / "fault.log"
    argv = [
        "simulated_fault.py",
        "--log_file",
        str(log_file),
        "--errors",
        "out of memory",
        "cuda error",
        "--interval",
        "9",
        "--iterations",
        "4",
        "--mode",
        "w",
    ]

    with (
        patch("sys.argv", argv),
        patch("flagscale.runner.elastic.simulated_fault.simulated_fault_loop") as loop,
    ):
        simulated_fault.main()

    loop.assert_called_once_with(
        log_file=str(log_file),
        error_keys=["out of memory", "cuda error"],
        interval=9,
        iterations=4,
        mode="w",
    )


def test_main_uses_default_arguments():
    with (
        patch("sys.argv", ["simulated_fault.py"]),
        patch("flagscale.runner.elastic.simulated_fault.simulated_fault_loop") as loop,
    ):
        simulated_fault.main()

    loop.assert_called_once_with(
        log_file="output.log",
        error_keys=None,
        interval=5,
        iterations=1,
        mode="a",
    )
