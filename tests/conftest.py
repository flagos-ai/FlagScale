import pytest


def pytest_addoption(parser):
    """Register pytest options for test configuration and environment."""
    opts = [
        ("--path", "path", "Base directory path for test cases"),
        ("--task", "task", "Task type (train/inference/hetero_train/rl/serve)"),
        ("--model", "model", "Model name (aquila/deepseek/mixtral)"),
        ("--case", "case", "Specific test case configuration"),
        (
            "--platform",
            "platform",
            "Platform type (cuda, etc.) - see tests/test_utils/config/platforms/",
        ),
        ("--device", "device", "Device type (a100/a800/etc.)"),
    ]
    for opt, name, help_text in opts:
        parser.addoption(opt, action="store", default="none", help=help_text)


@pytest.fixture
def path(request):
    return request.config.getoption("--path")


@pytest.fixture
def task(request):
    return request.config.getoption("--task")


@pytest.fixture
def model(request):
    return request.config.getoption("--model")


@pytest.fixture
def case(request):
    return request.config.getoption("--case")


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
