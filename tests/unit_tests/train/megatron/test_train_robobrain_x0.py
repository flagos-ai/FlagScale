"""
Unit tests for train_robobrain_x0.py dataloader consistency.

This test verifies that lerobot and energon dataloaders produce consistent data
when initialized with equivalent mock datasets.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

_mock_modules = [
    # megatron modules
    "megatron",
    "megatron.core",
    "megatron.core.config",
    "megatron.core.parallel_state",
    "megatron.core.datasets",
    "megatron.core.datasets.blended_megatron_dataset_builder",
    "megatron.core.datasets.gpt_dataset",
    "megatron.core.enums",
    "megatron.core.energy_monitor",
    "megatron.core.models",
    "megatron.core.models.gpt",
    "megatron.core.models.gpt.gpt_layer_specs",
    "megatron.core.models.gpt.heterogeneous",
    "megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs",
    "megatron.core.rerun_state_machine",
    "megatron.core.transformer",
    "megatron.core.transformer.spec_utils",
    "megatron.core.utils",
    "megatron.core.num_microbatches_calculator",
    "megatron.core.mpu",
    "megatron.energon",
    "megatron.legacy",
    "megatron.legacy.model",
    "megatron.plugin",
    "megatron.plugin.utils",
    "megatron.training",
    "megatron.training.arguments",
    "megatron.training.checkpointing",
    "megatron.training.dist_signal_handler",
    "megatron.training.global_vars",
    "megatron.training.spiky_loss",
    "megatron.training.tokenizer",
    "megatron.training.tokenizer.tokenizer",
    "megatron.training.training",
    "megatron.training.utils",
    "megatron.training.yaml_arguments",
    # external dependencies
    "webdataset",
    "webdataset.autodecode",
    # tools modules
    "tools",
    "tools.datasets",
    "tools.datasets.vla",
    "tools.datasets.vla.data",
    "tools.datasets.vla.data.dataset_helpers_vlm",
    "tools.datasets.vla.data.energon",
    "tools.datasets.vla.data.energon.chatml",
    # flagscale internal modules
    "flagscale.models.megatron.qwen2_5_vl",
    "flagscale.models.megatron.qwen2_5_vl.layer_specs",
    "flagscale.models.megatron.qwen2_5_vl.qwen2_5_vl_model",
    "flagscale.models.megatron.qwen2_5_vl.tensor_parallel",
    "flagscale.models.megatron.qwen2_5_vl.transformer_config",
    "flagscale.train.datasets.lerobot_dataset",
]

# Store original modules for cleanup
_original_modules = {}

for mod in _mock_modules:
    _original_modules[mod] = sys.modules.get(mod)
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from flagscale.train.megatron.train_robobrain_x0 import (
    EnergonDataloader,
    LeRobotDataloader,
    LeRobotDatasetWrapper,
    lerobot_collate_fn,
)


@pytest.fixture(scope="module", autouse=True)
def cleanup_mocked_modules():
    """Cleanup mocked modules after all tests in this module complete."""
    yield
    # Restore original modules or remove mocks
    for mod in _mock_modules:
        if _original_modules[mod] is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = _original_modules[mod]


class MockArgs:
    max_padding_length = 128
    temporal_patch_size = 2
    spatial_merge_size = 2
    patch_size = 14
    tensor_model_parallel_size = 1
    context_parallel_size = 1
    sequence_parallel = False
    micro_batch_size = 2
    num_workers = 0
    enable_variable_seq_lengths = False
    transformer_pipeline_model_parallel_size = 0
    video_backend = "pyav"


class MockTokenizer:
    pad_token_id = 0
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    vocab = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "user": 151646,
        "assistant": 151647,
        "system": 151648,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
        "<boa>": 151665,
        "<EOA>": 151666,
        "<action_split>": 151667,
    }
    # Add action tokens to vocab
    for i in range(2048):
        vocab[f"<action_token_{i}>"] = 149595 + i
    processor = MagicMock()

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text[:10]]


mock_args = MockArgs()
mock_tokenizer = MockTokenizer()


def create_mock_batch_data(batch_size, seq_len, num_images=1, seed=42):
    """Create mock batch data for both lerobot and energon formats."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    img_dim = 3 * 2 * 14 * 14
    num_patches = num_images * 4

    return {
        "text": np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64),
        "target": np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int64),
        "imgs": np.random.randn(num_patches, img_dim).astype(np.float32),
        "videos": np.empty([0, img_dim], dtype=np.float32),
        "image_thw_grids": np.array([[1, 2, 2]] * num_images, dtype=np.int64),
        "video_thw_grids": np.empty([0, 3], dtype=np.int64),
        "image_input_mask": np.zeros((batch_size, seq_len), dtype=bool),
        "video_input_mask": np.zeros((batch_size, seq_len), dtype=bool),
        "second_per_grid_ts": np.empty([0], dtype=np.float32),
    }


def to_lerobot_samples(data, batch_size):
    """Convert batch data to individual lerobot-style samples."""
    return [
        {
            "text": data["text"][i],
            "target": data["target"][i],
            "imgs": data["imgs"]
            if i == 0
            else np.empty([0, data["imgs"].shape[1]], dtype=np.float32),
            "videos": data["videos"],
            "image_thw_grids": data["image_thw_grids"]
            if i == 0
            else np.empty([0, 3], dtype=np.int64),
            "video_thw_grids": data["video_thw_grids"],
            "image_input_mask": data["image_input_mask"][i],
            "video_input_mask": data["video_input_mask"][i],
            "second_per_grid_ts": data["second_per_grid_ts"],
        }
        for i in range(batch_size)
    ]


def to_energon_batch(data):
    """Convert data to energon batch format (torch tensors)."""
    return {k: torch.from_numpy(v) for k, v in data.items()}


class TestDataloaderConsistency:
    """Test data consistency between lerobot and energon dataloaders."""

    @pytest.fixture
    def batch_data(self):
        return create_mock_batch_data(
            mock_args.micro_batch_size, mock_args.max_padding_length, num_images=2
        )

    def _get_lerobot_batch(self, data):
        samples = to_lerobot_samples(data, mock_args.micro_batch_size)
        with (
            patch("flagscale.train.megatron.train_robobrain_x0.get_args", return_value=mock_args),
            patch(
                "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
                return_value=mock_tokenizer,
            ),
        ):
            return lerobot_collate_fn(samples)

    def test_output_keys_match(self, batch_data):
        """Verify both formats output the same keys."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert set(lerobot_batch.keys()) == set(energon_batch.keys())

    def test_text_consistency(self, batch_data):
        """Verify text tokens match."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.allclose(lerobot_batch["text"].float(), energon_batch["text"].float())

    def test_target_consistency(self, batch_data):
        """Verify target tokens match."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.allclose(lerobot_batch["target"].float(), energon_batch["target"].float())

    def test_image_mask_consistency(self, batch_data):
        """Verify image masks match."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.equal(lerobot_batch["image_input_mask"], energon_batch["image_input_mask"])

    def test_grid_consistency(self, batch_data):
        """Verify grid info matches."""
        lerobot_batch = self._get_lerobot_batch(batch_data)
        energon_batch = to_energon_batch(batch_data)
        assert torch.equal(lerobot_batch["image_thw_grids"], energon_batch["image_thw_grids"])
        assert torch.equal(lerobot_batch["video_thw_grids"], energon_batch["video_thw_grids"])

    def test_multiple_seeds(self):
        """Verify consistency across multiple random seeds."""
        for seed in [42, 123, 456]:
            data = create_mock_batch_data(
                mock_args.micro_batch_size, mock_args.max_padding_length, seed=seed
            )
            lerobot_batch = self._get_lerobot_batch(data)
            energon_batch = to_energon_batch(data)
            assert torch.allclose(lerobot_batch["text"].float(), energon_batch["text"].float())

    def test_dataloader_iteration_consistency(self, batch_data):
        """Verify data consistency when iterating through both dataloaders."""
        samples = to_lerobot_samples(batch_data, mock_args.micro_batch_size)

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, d):
                self.d = d

            def __len__(self):
                return len(self.d)

            def __getitem__(self, i):
                return self.d[i]

        def collate_fn(x):
            with (
                patch(
                    "flagscale.train.megatron.train_robobrain_x0.get_args", return_value=mock_args
                ),
                patch(
                    "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
                    return_value=mock_tokenizer,
                ),
            ):
                return lerobot_collate_fn(x)

        lerobot_loader = LeRobotDataloader(
            torch.utils.data.DataLoader(
                SimpleDataset(samples), batch_size=len(samples), collate_fn=collate_fn
            )
        )

        class MockEnergonIter:
            def __init__(self, b):
                self.b = b

            def __iter__(self):
                return self

            def __next__(self):
                return self.b

        energon_loader = EnergonDataloader(MockEnergonIter(to_energon_batch(batch_data)))

        lerobot_batch = next(lerobot_loader)
        energon_batch = next(energon_loader)

        assert torch.allclose(lerobot_batch["text"].float(), energon_batch["text"].float())
        assert torch.equal(lerobot_batch["image_input_mask"], energon_batch["image_input_mask"])


class MockLeRobotMeta:
    """Mock metadata for LeRobotDataset."""

    def __init__(self, camera_keys=None, features=None):
        self.camera_keys = camera_keys or ["observation.image"]
        self.features = features or {"action": {"shape": [7]}}


class MockLeRobotDataset:
    """Mock LeRobotDataset for testing LeRobotDatasetWrapper."""

    def __init__(self, samples=None, camera_keys=None, action_dim=7):
        self.samples = samples or []
        self.meta = MockLeRobotMeta(
            camera_keys=camera_keys, features={"action": {"shape": [action_dim]}}
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TestLeRobotDatasetWrapper:
    """Test LeRobotDatasetWrapper functionality."""

    @pytest.fixture
    def mock_lerobot_dataset(self):
        """Create a mock LeRobotDataset with sample data."""
        samples = [
            {
                "observation.image": torch.rand(3, 224, 224),
                "action": torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),
                "task": "Pick up the object.",
            },
            {
                "observation.image": torch.rand(3, 224, 224),
                "action": torch.tensor([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]),
                "task": "Place the object on the table.",
            },
        ]
        return MockLeRobotDataset(samples=samples, camera_keys=["observation.image"])

    @pytest.fixture
    def wrapper(self, mock_lerobot_dataset):
        """Create a LeRobotDatasetWrapper instance."""
        with patch(
            "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            return LeRobotDatasetWrapper(mock_lerobot_dataset, mock_args)

    def test_init_action_dim_from_features(self):
        """Test that action_dim is correctly extracted from dataset features."""
        dataset = MockLeRobotDataset(samples=[], action_dim=10)
        with patch(
            "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            wrapper = LeRobotDatasetWrapper(dataset, mock_args)
            assert wrapper.action_dim == 10

    def test_init_action_dim_default(self):
        """Test that action_dim defaults to 7 when not in features."""
        dataset = MockLeRobotDataset(samples=[])
        dataset.meta.features = {}  # Remove action from features
        with patch(
            "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            wrapper = LeRobotDatasetWrapper(dataset, mock_args)
            assert wrapper.action_dim == 7

    def test_len(self, wrapper, mock_lerobot_dataset):
        """Test __len__ returns correct dataset length."""
        assert len(wrapper) == len(mock_lerobot_dataset)
        assert len(wrapper) == 2

    def test_discretize_action_range(self, wrapper):
        """Test _discretize_action produces values in valid range."""
        action = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75])
        result = wrapper._discretize_action(action)
        assert all(0 <= v < 2048 for v in result)

    def test_discretize_action_boundary_values(self, wrapper):
        """Test _discretize_action handles boundary values correctly."""
        # Action at -1 should map to 0
        action_min = np.array([-1.0])
        result_min = wrapper._discretize_action(action_min)
        assert result_min[0] == 0

        # Action at 1 should map to 2047
        action_max = np.array([1.0])
        result_max = wrapper._discretize_action(action_max)
        assert result_max[0] == 2047

        # Action at 0 should map to middle (~1023)
        action_mid = np.array([0.0])
        result_mid = wrapper._discretize_action(action_mid)
        assert 1020 <= result_mid[0] <= 1025

    def test_discretize_action_clipping(self, wrapper):
        """Test _discretize_action clips out-of-range values."""
        # Values outside [-1, 1] should be clipped
        action = np.array([-2.0, 2.0])
        result = wrapper._discretize_action(action)
        assert result[0] == 0  # Clipped to min
        assert result[1] == 2047  # Clipped to max

    def test_build_token_cache(self, wrapper):
        """Test _build_token_cache creates expected keys."""
        expected_keys = [
            "im_start",
            "im_end",
            "user",
            "assistant",
            "system",
            "vision_start",
            "vision_end",
            "image_pad",
            "video_pad",
            "newline",
            "space",
            "boa",
            "EOA",
            "action_split",
        ]
        for key in expected_keys:
            assert key in wrapper._token_cache

    def test_build_action_token_cache(self, wrapper):
        """Test _build_action_token_cache creates valid action token mappings."""
        # Should have entries for action tokens 0-2047
        assert len(wrapper._action_token_cache) > 0
        # Check that token IDs are in expected range
        for action_id, token_id in wrapper._action_token_cache.items():
            assert 0 <= action_id < 2048
            assert (
                LeRobotDatasetWrapper.ACTION_TOKEN_START_ID
                <= token_id
                < LeRobotDatasetWrapper.ACTION_TOKEN_END_ID
            )

    def test_find_turn_end(self, wrapper):
        """Test _find_turn_end finds correct position."""
        im_end_id = wrapper._token_cache["im_end"]
        # Create a simple token sequence with im_end at position 5
        input_ids = np.array([1, 2, 3, 4, 5, im_end_id, 7, 8])
        result = wrapper._find_turn_end(input_ids, 0)
        assert result == 5

    def test_find_turn_end_from_offset(self, wrapper):
        """Test _find_turn_end respects start_idx."""
        im_end_id = wrapper._token_cache["im_end"]
        # Create a sequence with multiple im_end tokens
        input_ids = np.array([1, im_end_id, 3, 4, 5, im_end_id, 7, 8])
        # Starting from index 2, should find the second im_end at position 5
        result = wrapper._find_turn_end(input_ids, 2)
        assert result == 5

    def test_find_turn_end_not_found(self, wrapper):
        """Test _find_turn_end returns last index when im_end not found."""
        input_ids = np.array([1, 2, 3, 4, 5])
        result = wrapper._find_turn_end(input_ids, 0)
        assert result == len(input_ids) - 1

    def test_build_conversation_tokens_structure(self, wrapper):
        """Test _build_conversation_tokens produces valid token sequence."""
        conversation = [
            {"role": "system", "content": "You are a robot."},
            {"role": "user", "content": "Do task."},
            {"role": "assistant", "content": ""},
        ]
        action_tokens_list = [[], [], [0, 100, 200]]
        result = wrapper._build_conversation_tokens(conversation, action_tokens_list)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        # Should contain im_start and im_end tokens
        im_start_id = wrapper._token_cache["im_start"]
        im_end_id = wrapper._token_cache["im_end"]
        assert im_start_id in result
        assert im_end_id in result

    def test_build_conversation_tokens_with_image(self, wrapper):
        """Test _build_conversation_tokens handles image content."""
        conversation = [
            {"role": "system", "content": "You are a robot."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "0"},
                    {"type": "text", "text": "What do you see?"},
                ],
            },
            {"role": "assistant", "content": ""},
        ]
        action_tokens_list = [[], [], []]
        result = wrapper._build_conversation_tokens(conversation, action_tokens_list)

        # Should contain image_pad token
        image_pad_id = wrapper._token_cache["image_pad"]
        assert image_pad_id in result

    def test_getitem_returns_expected_keys(self, wrapper):
        """Test __getitem__ returns dict with expected keys."""
        with patch.object(wrapper.tokenizer, "processor", create=True) as mock_processor:
            mock_processor.image_processor.return_value = {
                "pixel_values": np.random.randn(4, 3 * 2 * 14 * 14).astype(np.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
            result = wrapper[0]

        expected_keys = [
            "text",
            "target",
            "imgs",
            "videos",
            "image_thw_grids",
            "video_thw_grids",
            "image_input_mask",
            "video_input_mask",
            "second_per_grid_ts",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_getitem_without_action(self):
        """Test __getitem__ handles samples without action."""
        samples = [
            {
                "observation.image": torch.rand(3, 224, 224),
                "task": "Observe the scene.",
            }
        ]
        dataset = MockLeRobotDataset(samples=samples)
        with patch(
            "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            wrapper = LeRobotDatasetWrapper(dataset, mock_args)
            with patch.object(wrapper.tokenizer, "processor", create=True) as mock_processor:
                mock_processor.image_processor.return_value = {
                    "pixel_values": np.random.randn(4, 3 * 2 * 14 * 14).astype(np.float32),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                }
                result = wrapper[0]

        # Should still return valid result
        assert "text" in result
        assert "target" in result

    def test_getitem_without_camera(self):
        """Test __getitem__ handles samples without camera images."""
        samples = [
            {
                "action": torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),
                "task": "Execute blind task.",
            }
        ]
        dataset = MockLeRobotDataset(samples=samples, camera_keys=["observation.image"])
        with patch(
            "flagscale.train.megatron.train_robobrain_x0.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            wrapper = LeRobotDatasetWrapper(dataset, mock_args)
            result = wrapper[0]

        # Should return empty image arrays
        assert result["imgs"].shape[0] == 0
        assert result["image_thw_grids"].shape[0] == 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
