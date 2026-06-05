import numpy as np
import pytest

from flagscale.serve.processor.image_resize_processor import ImageResizeProcessorStep


@pytest.fixture
def step():
    return ImageResizeProcessorStep(image_size=[128, 128])


def _make_obs(**image_kwargs):
    obs = {"observation.state": np.array([1.0, 2.0])}
    for key, img in image_kwargs.items():
        obs[f"observation.images.{key}"] = img
    return obs


def test_resizes_hwc_uint8(step):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs = _make_obs(image=img)
    result = step.observation(obs)
    assert result["observation.images.image"].shape == (128, 128, 3)
    assert result["observation.images.image"].dtype == np.uint8


def test_preserves_non_image_keys(step):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs = _make_obs(image=img)
    result = step.observation(obs)
    np.testing.assert_array_equal(result["observation.state"], np.array([1.0, 2.0]))


def test_multiple_images(step):
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    obs = _make_obs(image=img1, wrist_image=img2)
    result = step.observation(obs)
    assert result["observation.images.image"].shape == (128, 128, 3)
    assert result["observation.images.wrist_image"].shape == (128, 128, 3)


def test_skips_none_values(step):
    obs = _make_obs(image=None)
    result = step.observation(obs)
    assert result["observation.images.image"] is None


def test_skips_non_ndarray(step):
    obs = _make_obs(image="not_an_image")
    result = step.observation(obs)
    assert result["observation.images.image"] == "not_an_image"


def test_warns_on_wrong_ndim(step, caplog):
    import logging

    fs_logger = logging.getLogger("FlagScale")
    fs_logger.propagate = True
    try:
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        obs = _make_obs(image=img)
        with caplog.at_level(logging.WARNING, logger="FlagScale"):
            result = step.observation(obs)
        assert result["observation.images.image"].shape == (480, 640)
        assert "ndim=2" in caplog.text
    finally:
        fs_logger.propagate = False


def test_noop_when_already_target_size(step):
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    obs = _make_obs(image=img)
    result = step.observation(obs)
    assert result["observation.images.image"].shape == (128, 128, 3)


def test_default_image_size():
    step = ImageResizeProcessorStep()
    assert step.image_size == [224, 224]


def test_get_config(step):
    assert step.get_config() == {"image_size": [128, 128]}


def test_ignores_non_image_observation_keys(step):
    obs = {
        "observation.state": np.array([1.0]),
        "observation.images.cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "task": "pick up the cup",
    }
    result = step.observation(obs)
    assert result["observation.images.cam"].shape == (128, 128, 3)
    assert result["task"] == "pick up the cup"
    np.testing.assert_array_equal(result["observation.state"], np.array([1.0]))
