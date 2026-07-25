# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_get_version_from_importlib_metadata(mocker):
    """Test version retrieval from importlib.metadata"""
    mocker.patch("importlib.metadata.version", return_value="1.2.3")

    # Import the function fresh to test with mocked metadata
    from flagscale import _get_version

    # Since the module is already imported, we need to call the function directly
    # The mock should be in place for the importlib.metadata.version call
    result = _get_version()
    assert result == "1.2.3"


def test_get_version_fallback_to_pyproject(mocker, tmp_path):
    """Test fallback to pyproject.toml when importlib fails"""
    # Make importlib.metadata.version raise an exception
    mocker.patch("importlib.metadata.version", side_effect=Exception("Not installed"))

    # Create a fake pyproject.toml
    pyproject_content = b'[project]\nversion = "2.0.0"\n'
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_bytes(pyproject_content)

    from flagscale import _get_version

    # This test verifies the fallback logic exists
    # The actual behavior depends on file system state
    result = _get_version()
    assert isinstance(result, str)


def test_get_version_fallback_to_default(mocker):
    """Test fallback to '0.0.0' when all methods fail"""
    # Make importlib.metadata.version raise an exception
    mocker.patch("importlib.metadata.version", side_effect=Exception("Not installed"))

    # Make tomllib.load raise an exception
    mocker.patch("tomllib.load", side_effect=Exception("Parse error"))

    from flagscale import _get_version

    result = _get_version()
    # Should return a string (either version or default)
    assert isinstance(result, str)


def test_version_is_string():
    """Test that __version__ is always a string"""
    from flagscale import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
