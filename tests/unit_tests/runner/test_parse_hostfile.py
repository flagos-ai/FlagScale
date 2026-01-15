import pytest

from flagscale.runner.utils import parse_hostfile


@pytest.fixture
def mock_os_path_isfile(mocker):
    return mocker.patch("os.path.isfile", return_value=True)


@pytest.fixture
def mock_open(mocker):
    return mocker.patch("builtins.open", mocker.mock_open())


def test_parse_hostfile_non_existing_file(mocker):
    mocker.patch("os.path.isfile", return_value=False)
    result = parse_hostfile("/path/to/non_existing_hostfile.txt")
    assert result is None


def test_parse_hostfile_correctly_formatted(mock_os_path_isfile, mock_open):
    hostfile_content = [
        "worker0 slots=16 type=A100",
        "worker1 slots=8 type=V100",
        "worker2 slots=32",
        "# comment line",
        "worker3 slots=16 type=A100",
        "# comment line",
    ]
    # expected_result = {
    #     "worker0": {"slots": 16, "type": "A100"},
    #     "worker1": {"slots": 8, "type": "V100"},
    #     "worker2": {"slots": 32, "type": None},
    #     "worker3": {"slots": 16, "type": "A100"},
    # }

    with pytest.raises(
        AssertionError, match="All hosts must have the a machine type or no machine type specified."
    ):
        mock_open.return_value.readlines.return_value = hostfile_content
        parse_hostfile("/path/to/hostfile.txt")


def test_parse_hostfile_incorrectly_formatted(mock_os_path_isfile, mock_open):
    hostfile_content = ["worker0 slots=16 type=A100", "invalid line", "# comment line"]

    mock_open.return_value.readlines.return_value = hostfile_content
    with pytest.raises(ValueError):
        parse_hostfile("/path/to/hostfile.txt")


def test_parse_hostfile_empty(mock_os_path_isfile, mock_open):
    hostfile_content = ""

    mock_open.return_value.readlines.return_value = hostfile_content
    with pytest.raises(ValueError):
        parse_hostfile("/path/to/hostfile.txt")
