import pytest


def pytest_addoption(parser):
    """Register pytest options for test configuration and environment."""
    opts = [
        ("--test_path", "test_path", "Base directory path for test cases"),
        ("--test_type", "test_type", "Test type (train/inference/hetero_train/rl/serve)"),
        ("--test_task", "test_task", "Task/model name (aquila/deepseek/mixtral)"),
        ("--test_case", "test_case", "Specific test case configuration"),
        ("--platform", "platform", "Platform type (default/a100)"),
        ("--device", "device", "Device type (generic/a100/a800/h100)"),
    ]
    for opt, name, help_text in opts:
        parser.addoption(opt, action="store", default="none", help=help_text)


@pytest.fixture
def test_path(request):
    return request.config.getoption("--test_path")


@pytest.fixture
def test_type(request):
    return request.config.getoption("--test_type")


@pytest.fixture
def test_task(request):
    return request.config.getoption("--test_task")


@pytest.fixture
def test_case(request):
    return request.config.getoption("--test_case")


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
