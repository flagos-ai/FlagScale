import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from flagscale.models.utils.hub_utils import _get_lock, resolve_model_path, use_modelscope


class TestUseModelscope(unittest.TestCase):
    def test_default_is_false(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(use_modelscope())

    def test_true(self):
        with patch.dict(os.environ, {"FLAGSCALE_USE_MODELSCOPE": "true"}):
            self.assertTrue(use_modelscope())

    def test_true_case_insensitive(self):
        with patch.dict(os.environ, {"FLAGSCALE_USE_MODELSCOPE": "True"}):
            self.assertTrue(use_modelscope())

    def test_false_explicit(self):
        with patch.dict(os.environ, {"FLAGSCALE_USE_MODELSCOPE": "false"}):
            self.assertFalse(use_modelscope())

    def test_other_value_is_false(self):
        with patch.dict(os.environ, {"FLAGSCALE_USE_MODELSCOPE": "1"}):
            self.assertFalse(use_modelscope())


class TestGetLock(unittest.TestCase):
    def test_lock_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = _get_lock("org/model", cache_dir=tmpdir)
            self.assertTrue(lock.lock_file.startswith(tmpdir))
            self.assertTrue(lock.lock_file.endswith(".lock"))

    def test_different_models_different_locks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock1 = _get_lock("org/model-a", cache_dir=tmpdir)
            lock2 = _get_lock("org/model-b", cache_dir=tmpdir)
            self.assertNotEqual(lock1.lock_file, lock2.lock_file)


class TestResolveModelPath(unittest.TestCase):
    def test_local_directory_returned_as_is(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = resolve_model_path(tmpdir)
            self.assertEqual(result, tmpdir)

    @patch("flagscale.models.utils.hub_utils.use_modelscope", return_value=False)
    def test_hf_download(self, _mock_use_ms):
        with patch("huggingface_hub.snapshot_download", return_value="/cache/model") as mock_dl:
            result = resolve_model_path("org/model", revision="main", cache_dir="/tmp/cache")
            mock_dl.assert_called_once_with(
                "org/model",
                repo_type="model",
                revision="main",
                cache_dir="/tmp/cache",
                allow_patterns=None,
                ignore_patterns=None,
            )
            self.assertEqual(result, "/cache/model")

    @patch("flagscale.models.utils.hub_utils.use_modelscope", return_value=True)
    def test_modelscope_download(self, _mock_use_ms):
        mock_ms_module = MagicMock()
        mock_ms_module.snapshot_download.return_value = "/cache/ms_model"
        with patch.dict("sys.modules", {"modelscope.hub.snapshot_download": mock_ms_module}):
            result = resolve_model_path("org/model", revision="v1", cache_dir="/tmp/ms")
            mock_ms_module.snapshot_download.assert_called_once_with(
                model_id="org/model",
                cache_dir="/tmp/ms",
                revision="v1",
                ignore_file_pattern=None,
                allow_patterns=None,
            )
            self.assertEqual(result, "/cache/ms_model")

    @patch("flagscale.models.utils.hub_utils.use_modelscope", return_value=False)
    def test_raises_on_empty_download_result(self, _mock_use_ms):
        with (
            patch("huggingface_hub.snapshot_download", return_value=None),
            self.assertRaises(RuntimeError),
        ):
            resolve_model_path("org/model")

    def test_allow_and_ignore_patterns(self):
        with (
            patch("flagscale.models.utils.hub_utils.use_modelscope", return_value=False),
            patch("huggingface_hub.snapshot_download", return_value="/cache/m") as mock_dl,
        ):
            resolve_model_path(
                "org/model",
                allow_patterns=["*.safetensors"],
                ignore_patterns=["*.bin"],
            )
            _, kwargs = mock_dl.call_args
            self.assertEqual(kwargs["allow_patterns"], ["*.safetensors"])
            self.assertEqual(kwargs["ignore_patterns"], ["*.bin"])


if __name__ == "__main__":
    unittest.main()
