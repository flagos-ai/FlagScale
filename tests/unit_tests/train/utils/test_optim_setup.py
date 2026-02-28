"""Unit tests for optimizer setup utilities."""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from flagscale.train.utils.optim_setup import (
    apply_freeze_config,
    build_optim_param_groups,
    freeze_and_get_trainable_params,
    log_trainable_params,
    print_param_names,
    setup_optimizer,
    setup_optimizer_and_scheduler,
    setup_scheduler,
)


class SimpleModel(nn.Module):
    """Simple model for testing freeze patterns."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        self.head = nn.Linear(10, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class NestedModel(nn.Module):
    """Model with nested structure similar to QwenGR00T."""

    def __init__(self):
        super().__init__()
        self.vlm = nn.ModuleDict(
            {
                "visual": nn.Sequential(
                    nn.Linear(10, 20),
                    nn.Linear(20, 10),
                ),
                "language": nn.ModuleDict(
                    {
                        "layers": nn.ModuleList([nn.Linear(10, 10) for _ in range(5)]),
                        "embed": nn.Embedding(100, 10),
                    }
                ),
            }
        )
        self.action_model = nn.ModuleDict(
            {
                "encoder": nn.Linear(10, 20),
                "decoder": nn.Linear(20, 10),
                "transformer_blocks": nn.ModuleList([nn.Linear(10, 10) for _ in range(4)]),
            }
        )

    def forward(self, x):
        return x


class TestFreezeAndGetTrainableParams(unittest.TestCase):
    """Test freeze_and_get_trainable_params function."""

    def setUp(self):
        self.model = SimpleModel()

    def test_no_patterns_all_trainable(self):
        """Without patterns, all params should be trainable."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=None,
                keep_patterns=None,
            )
        )

        all_params = list(self.model.parameters())
        self.assertEqual(len(params), len(all_params))

        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

    def test_freeze_single_module(self):
        """Test freezing a single module by pattern."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=None,
            )
        )

        # Check encoder is frozen
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
            else:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

        # Returned params should only be trainable ones
        for param in params:
            self.assertTrue(param.requires_grad)

    def test_freeze_multiple_modules(self):
        """Test freezing multiple modules."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*", "decoder\\..*"],
                keep_patterns=None,
            )
        )

        # Only head should be trainable
        for name, param in self.model.named_parameters():
            if name.startswith("head"):
                self.assertTrue(param.requires_grad)
            else:
                self.assertFalse(param.requires_grad)

        # Returned params should only be head params
        head_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("head")
        )
        self.assertEqual(len(params), head_param_count)

    def test_freeze_all_pattern(self):
        """Test freezing everything with '.*' pattern."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=[".*"],
                keep_patterns=None,
            )
        )

        self.assertEqual(len(params), 0)
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_keep_patterns_override_freeze(self):
        """Test that keep_patterns override freeze_patterns."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=[".*"],  # Freeze everything
                keep_patterns=["head\\..*"],  # But keep head trainable
            )
        )

        # Only head should be trainable
        for name, param in self.model.named_parameters():
            if name.startswith("head"):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            else:
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Should only return head params
        self.assertEqual(len(params), 2)  # head.weight and head.bias

    def test_partial_pattern_match(self):
        """Test that patterns use search (partial match)."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["weight"],  # Matches all weights
                keep_patterns=None,
            )
        )

        # Only biases should be trainable
        for name, param in self.model.named_parameters():
            if "weight" in name:
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)

        # Returned params should only be biases
        bias_param_count = sum(
            1 for name, _ in self.model.named_parameters() if "weight" not in name
        )
        self.assertEqual(len(params), bias_param_count)


class TestFreezeWithNestedModel(unittest.TestCase):
    """Test freeze patterns with nested model structure."""

    def setUp(self):
        self.model = NestedModel()

    def test_freeze_vlm_module(self):
        """Test freezing entire VLM module."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["vlm\\..*"],
                keep_patterns=None,
            )
        )

        for name, param in self.model.named_parameters():
            if name.startswith("vlm"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
            else:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

        # Returned params should only be action_model params
        action_model_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("action_model")
        )
        self.assertEqual(len(params), action_model_param_count)

    def test_freeze_specific_layers(self):
        """Test freezing specific layers by index."""
        # Freeze layers 0-2
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["vlm\\.language\\.layers\\.[0-2]\\..*"],
                keep_patterns=None,
            )
        )

        for name, param in self.model.named_parameters():
            if (
                "vlm.language.layers.0" in name
                or "vlm.language.layers.1" in name
                or "vlm.language.layers.2" in name
            ):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Layers 3-4 should still be trainable
        for name, param in self.model.named_parameters():
            if "vlm.language.layers.3" in name or "vlm.language.layers.4" in name:
                self.assertTrue(param.requires_grad, f"{name} should be trainable")

        # Returned params should exclude frozen layers
        trainable_param_count = sum(
            1 for name, param in self.model.named_parameters() if param.requires_grad
        )
        self.assertEqual(len(params), trainable_param_count)

    def test_freeze_vlm_keep_visual(self):
        """Test freezing VLM but keeping visual encoder trainable."""
        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["vlm\\..*"],
                keep_patterns=["vlm\\.visual\\..*"],
            )
        )

        for name, param in self.model.named_parameters():
            if name.startswith("vlm.visual"):
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            elif name.startswith("vlm"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")

        # Returned params should include visual and action_model params
        trainable_param_count = sum(
            1 for name, param in self.model.named_parameters() if param.requires_grad
        )
        self.assertEqual(len(params), trainable_param_count)


class TestApplyFreezeConfig(unittest.TestCase):
    """Test apply_freeze_config function."""

    def setUp(self):
        self.model = SimpleModel()

    def test_none_config_returns_all_params(self):
        """With None config, should return all parameters."""
        params = apply_freeze_config(self.model, None)

        all_params = list(self.model.parameters())
        self.assertEqual(len(params), len(all_params))

    def test_with_freeze_config(self):
        """Test with a FreezeConfig-like object."""
        freeze_config = MagicMock()
        freeze_config.freeze_patterns = ["encoder\\..*"]
        freeze_config.keep_patterns = None

        params = apply_freeze_config(self.model, freeze_config)

        # Should only return non-encoder params
        encoder_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("encoder")
        )
        total_param_count = sum(1 for _ in self.model.parameters())

        self.assertEqual(len(params), total_param_count - encoder_param_count)


class TestLogTrainableParams(unittest.TestCase):
    """Test log_trainable_params function."""

    def setUp(self):
        self.model = SimpleModel()

    def test_all_trainable(self):
        """Test logging when all params are trainable."""
        result = log_trainable_params(self.model)

        self.assertIn("trainable", result)
        self.assertIn("frozen", result)
        self.assertIn("encoder", result["trainable"])
        self.assertIn("decoder", result["trainable"])
        self.assertIn("head", result["trainable"])

    def test_partial_frozen(self):
        """Test logging with some frozen params."""
        # Freeze encoder
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False

        result = log_trainable_params(self.model)

        self.assertIn("encoder", result["frozen"])
        self.assertIn("decoder", result["trainable"])
        self.assertIn("head", result["trainable"])
        self.assertGreater(result["frozen"]["encoder"], 0)


class TestUnusedPatternWarnings(unittest.TestCase):
    """Test that unused patterns trigger warnings."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_warns_on_unused_freeze_pattern(self, mock_logger):
        """Should warn when freeze pattern matches nothing."""
        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["nonexistent_module\\..*"],
                keep_patterns=None,
            )
        )

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Freeze patterns matched nothing", warning_call)

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_warns_on_unused_keep_pattern(self, mock_logger):
        """Should warn when keep pattern matches nothing."""
        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=["nonexistent_module\\..*"],
            )
        )

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("Keep patterns matched nothing", warning_call)


class TestPrintParamNames(unittest.TestCase):
    """Test print_param_names debug helper."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("builtins.print")
    def test_prints_all_params(self, mock_print):
        """Should print all params when no pattern given."""
        print_param_names(self.model)

        self.assertGreater(mock_print.call_count, 0)

    @patch("builtins.print")
    def test_filters_by_pattern(self, mock_print):
        """Should only print params matching pattern."""
        print_param_names(self.model, pattern="encoder")

        # Should only print encoder params
        for call in mock_print.call_args_list:
            self.assertIn("encoder", call[0][0])


class TestParameterCounts(unittest.TestCase):
    """Test that parameter counts are correctly reported."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_parameter_count_logging(self, mock_logger):
        """Verify correct parameter counts are logged."""
        # Count total params
        total_params = sum(p.numel() for p in self.model.parameters())

        # Count encoder params
        encoder_params = sum(
            p.numel() for name, p in self.model.named_parameters() if name.startswith("encoder")
        )

        # Freeze encoder
        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=None,
            )
        )

        # Check that info was logged with correct counts
        mock_logger.info.assert_called()
        info_call = mock_logger.info.call_args[0][0]
        self.assertIn(f"trainable={total_params - encoder_params:,}", info_call)
        self.assertIn(f"frozen={encoder_params:,}", info_call)


class TestBuildOptimParamGroups(unittest.TestCase):
    """Test build_optim_param_groups function (NeMo-style per-module config)."""

    def setUp(self):
        self.model = SimpleModel()

    def test_none_config_returns_single_group(self):
        """With None config, should return single group with all params."""
        param_groups = build_optim_param_groups(self.model, None)

        self.assertEqual(len(param_groups), 1)
        all_params = list(self.model.parameters())
        self.assertEqual(len(param_groups[0]["params"]), len(all_params))

    def test_single_module_config(self):
        """Test with config for single module."""
        config = {"encoder": {"lr": 1e-5}}
        param_groups = build_optim_param_groups(self.model, config)

        # Should have 2 groups: default + encoder
        self.assertEqual(len(param_groups), 2)

        # Find encoder group
        encoder_group = next(g for g in param_groups if g.get("name") == "encoder")
        self.assertEqual(encoder_group["lr"], 1e-5)

        # Encoder params count
        encoder_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("encoder")
        )
        self.assertEqual(len(encoder_group["params"]), encoder_param_count)

    def test_multiple_module_config(self):
        """Test with config for multiple modules."""
        config = {
            "encoder": {"lr": 1e-5, "weight_decay": 0.01},
            "decoder": {"lr": 2e-5},
        }
        param_groups = build_optim_param_groups(self.model, config)

        # Should have 3 groups: default + encoder + decoder
        self.assertEqual(len(param_groups), 3)

        encoder_group = next(g for g in param_groups if g.get("name") == "encoder")
        decoder_group = next(g for g in param_groups if g.get("name") == "decoder")

        self.assertEqual(encoder_group["lr"], 1e-5)
        self.assertEqual(encoder_group["weight_decay"], 0.01)
        self.assertEqual(decoder_group["lr"], 2e-5)

    def test_default_group_contains_remaining_params(self):
        """Default group should contain params not in other groups."""
        config = {"encoder": {"lr": 1e-5}}
        param_groups = build_optim_param_groups(self.model, config)

        default_group = next(g for g in param_groups if g.get("name") == "default")

        # Default should contain decoder + head params
        non_encoder_count = sum(
            1 for name, _ in self.model.named_parameters() if not name.startswith("encoder")
        )
        self.assertEqual(len(default_group["params"]), non_encoder_count)

    def test_respects_requires_grad(self):
        """Should only include trainable params."""
        # Freeze encoder
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False

        config = {"encoder": {"lr": 1e-5}}
        param_groups = build_optim_param_groups(self.model, config)

        # Encoder group should not be added when it has no trainable params
        encoder_groups = [g for g in param_groups if g.get("name") == "encoder"]
        self.assertEqual(
            len(encoder_groups),
            0,
            "Encoder group should not exist when all params are frozen",
        )

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_warns_on_nonexistent_module(self, mock_logger):
        """Should warn when module doesn't exist."""
        config = {"nonexistent": {"lr": 1e-5}}
        build_optim_param_groups(self.model, config)

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("nonexistent", warning_call)


class TestBuildOptimParamGroupsNested(unittest.TestCase):
    """Test build_optim_param_groups with nested model structure."""

    def setUp(self):
        self.model = NestedModel()

    def test_nested_module_path(self):
        """Test accessing nested modules via dot path."""
        config = {"vlm.visual": {"lr": 1e-5}}
        param_groups = build_optim_param_groups(self.model, config)

        visual_group = next(g for g in param_groups if g.get("name") == "vlm.visual")
        self.assertEqual(visual_group["lr"], 1e-5)

        # Count visual params
        visual_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("vlm.visual")
        )
        self.assertEqual(len(visual_group["params"]), visual_param_count)

    def test_multiple_nested_paths(self):
        """Test multiple nested module configs."""
        config = {
            "vlm.visual": {"lr": 1e-5},
            "vlm.language": {"lr": 2e-5},
            "action_model": {"lr": 1e-4},
        }
        param_groups = build_optim_param_groups(self.model, config)

        # 3 configured groups + default (though default may be empty)
        groups_with_params = [g for g in param_groups if len(g["params"]) > 0]
        self.assertGreaterEqual(len(groups_with_params), 3)


class TestSetupScheduler(unittest.TestCase):
    """Test setup_scheduler function."""

    def setUp(self):
        self.model = SimpleModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def test_cosine_scheduler(self):
        """Test creating a cosine scheduler."""
        scheduler_config = MagicMock()
        scheduler_config.name = "cosine"
        scheduler_config.warmup_steps = 100
        scheduler_config.scheduler_kwargs = None

        scheduler = setup_scheduler(self.optimizer, scheduler_config, num_training_steps=1000)

        self.assertIsNotNone(scheduler)
        self.assertTrue(hasattr(scheduler, "step"))

    def test_linear_scheduler(self):
        """Test creating a linear scheduler."""
        scheduler_config = MagicMock()
        scheduler_config.name = "linear"
        scheduler_config.warmup_steps = 50
        scheduler_config.scheduler_kwargs = None

        scheduler = setup_scheduler(self.optimizer, scheduler_config, num_training_steps=500)

        self.assertIsNotNone(scheduler)

    def test_constant_with_warmup_scheduler(self):
        """Test creating a constant_with_warmup scheduler."""
        scheduler_config = MagicMock()
        scheduler_config.name = "constant_with_warmup"
        scheduler_config.warmup_steps = 100
        scheduler_config.scheduler_kwargs = None

        scheduler = setup_scheduler(self.optimizer, scheduler_config, num_training_steps=1000)

        self.assertIsNotNone(scheduler)

    def test_cosine_with_min_lr(self):
        """Test creating a cosine scheduler with min_lr."""
        scheduler_config = MagicMock()
        scheduler_config.name = "cosine_with_min_lr"
        scheduler_config.warmup_steps = 100
        scheduler_config.scheduler_kwargs = {"min_lr": 1e-6}

        scheduler = setup_scheduler(self.optimizer, scheduler_config, num_training_steps=1000)

        self.assertIsNotNone(scheduler)

    def test_raises_error_when_name_is_none(self):
        """Should raise ValueError when scheduler name is None."""
        scheduler_config = MagicMock()
        scheduler_config.name = None
        scheduler_config.warmup_steps = 100
        scheduler_config.scheduler_kwargs = None

        with self.assertRaises(ValueError) as context:
            setup_scheduler(self.optimizer, scheduler_config, num_training_steps=1000)

        self.assertIn("name must be specified", str(context.exception))

    def test_scheduler_step_updates_lr(self):
        """Test that scheduler step updates learning rate."""
        scheduler_config = MagicMock()
        scheduler_config.name = "linear"
        scheduler_config.warmup_steps = 10
        scheduler_config.scheduler_kwargs = None

        scheduler = setup_scheduler(self.optimizer, scheduler_config, num_training_steps=100)

        initial_lr = self.optimizer.param_groups[0]["lr"]
        for _ in range(50):
            scheduler.step()
        final_lr = self.optimizer.param_groups[0]["lr"]

        self.assertNotEqual(initial_lr, final_lr)

    def test_warmup_phase(self):
        """Test that warmup phase increases lr."""
        scheduler_config = MagicMock()
        scheduler_config.name = "linear"
        scheduler_config.warmup_steps = 100
        scheduler_config.scheduler_kwargs = None

        scheduler = setup_scheduler(self.optimizer, scheduler_config, num_training_steps=1000)

        lrs = []
        for _ in range(50):
            lrs.append(self.optimizer.param_groups[0]["lr"])
            scheduler.step()

        # During warmup, LR should generally increase
        self.assertLess(lrs[0], lrs[-1])


class TestSetupOptimizerAndScheduler(unittest.TestCase):
    """Test setup_optimizer_and_scheduler function."""

    def setUp(self):
        self.model = SimpleModel()

    def _make_train_config(self, freeze_patterns=None, keep_patterns=None):
        """Helper to create a mock TrainConfig."""
        train_config = MagicMock()
        # System config
        train_config.system = MagicMock()
        train_config.system.train_steps = 1000
        # Model config with optimizer, scheduler, and freeze
        train_config.model = MagicMock()
        train_config.model.optimizer = MagicMock()
        train_config.model.optimizer.name = "AdamW"
        train_config.model.optimizer.lr = 1e-4
        train_config.model.optimizer.param_groups = None
        train_config.model.optimizer.get_optimizer_kwargs.return_value = {"lr": 1e-4}
        train_config.model.optimizer.scheduler = MagicMock()
        train_config.model.optimizer.scheduler.name = "cosine"
        train_config.model.optimizer.scheduler.warmup_steps = 100
        train_config.model.optimizer.scheduler.scheduler_kwargs = None
        if freeze_patterns is not None:
            train_config.model.freeze = MagicMock()
            train_config.model.freeze.freeze_patterns = freeze_patterns
            train_config.model.freeze.keep_patterns = keep_patterns
        else:
            train_config.model.freeze = None
        return train_config

    def test_returns_optimizer_and_scheduler(self):
        """Test that function returns both optimizer and scheduler."""
        train_config = self._make_train_config()

        optimizer, scheduler = setup_optimizer_and_scheduler(self.model, train_config)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNotNone(scheduler)
        self.assertTrue(hasattr(scheduler, "step"))

    def test_with_freeze_config(self):
        """Test with freeze config applied."""
        train_config = self._make_train_config(freeze_patterns=["encoder\\..*"])
        train_config.model.optimizer.scheduler.name = "linear"
        train_config.model.optimizer.scheduler.warmup_steps = 50
        train_config.system.train_steps = 500

        optimizer, scheduler = setup_optimizer_and_scheduler(self.model, train_config)

        # Encoder should be frozen
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                self.assertFalse(param.requires_grad)
            else:
                self.assertTrue(param.requires_grad)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNotNone(scheduler)

    def test_scheduler_uses_train_steps(self):
        """Test that scheduler uses train_steps from TrainConfig."""
        train_config = self._make_train_config()
        train_config.model.optimizer.scheduler.name = "linear"
        train_config.model.optimizer.scheduler.warmup_steps = 10
        train_config.system.train_steps = 100

        optimizer, scheduler = setup_optimizer_and_scheduler(self.model, train_config)

        # Step through warmup first
        for _ in range(15):
            optimizer.step()
            scheduler.step()
        peak_lr = optimizer.param_groups[0]["lr"]

        # Step through decay phase
        for _ in range(80):
            optimizer.step()
            scheduler.step()
        final_lr = optimizer.param_groups[0]["lr"]

        # After decay, LR should be less than peak
        self.assertLess(final_lr, peak_lr)


class TestFreezeRequiresGradPreservation(unittest.TestCase):
    """Test that freeze logic correctly preserves or overrides requires_grad."""

    def setUp(self):
        self.model = SimpleModel()

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_no_freeze_patterns_preserves_requires_grad(self, mock_logger):
        """Params with requires_grad=False should stay frozen when no freeze patterns provided."""
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False

        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=None,
                keep_patterns=None,
            )
        )

        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                self.assertFalse(param.requires_grad, f"{name} should remain frozen")

        encoder_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("encoder")
        )
        total_count = sum(1 for _ in self.model.parameters())
        self.assertEqual(len(params), total_count - encoder_count)

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_freeze_patterns_forces_unmatched_trainable(self, mock_logger):
        """Params not matching freeze patterns become trainable even if originally frozen."""
        for param in self.model.parameters():
            param.requires_grad = False

        params = list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=None,
            )
        )

        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
            else:
                self.assertTrue(param.requires_grad, f"{name} should be forced trainable")

        encoder_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("encoder")
        )
        total_count = sum(1 for _ in self.model.parameters())
        self.assertEqual(len(params), total_count - encoder_count)

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_warns_when_unfreezing_previously_frozen(self, mock_logger):
        """Should warn about params that were frozen but are being made trainable."""
        for name, param in self.model.named_parameters():
            if name.startswith("decoder"):
                param.requires_grad = False

        list(
            freeze_and_get_trainable_params(
                self.model.named_parameters(),
                freeze_patterns=["encoder\\..*"],
                keep_patterns=None,
            )
        )

        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        summary_warnings = [w for w in warning_calls if "already frozen" in w]
        self.assertEqual(len(summary_warnings), 1)

        per_param_warnings = [w for w in warning_calls if "unfrozen:" in w]
        decoder_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("decoder")
        )
        self.assertEqual(len(per_param_warnings), decoder_param_count)
        for w in per_param_warnings:
            self.assertIn("decoder", w)


class TestBuildOptimParamGroupsOverlap(unittest.TestCase):
    """Test build_optim_param_groups with overlapping module paths."""

    def setUp(self):
        self.model = NestedModel()

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_parent_child_overlap_dedup(self, mock_logger):
        """Parent module listed before child: child group gets skipped."""
        config = {
            "vlm": {"lr": 1e-5},
            "vlm.visual": {"lr": 2e-5},
        }
        param_groups = build_optim_param_groups(self.model, config)

        vlm_group = next(g for g in param_groups if g.get("name") == "vlm")
        visual_groups = [g for g in param_groups if g.get("name") == "vlm.visual"]

        vlm_params = [p for p in self.model.vlm.parameters() if p.requires_grad]
        self.assertEqual(len(vlm_group["params"]), len(vlm_params))
        self.assertEqual(len(visual_groups), 0)

        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        overlap_warnings = [w for w in warning_calls if "already assigned" in w]
        self.assertGreater(len(overlap_warnings), 0)

    @patch("flagscale.train.utils.optim_setup.logger")
    def test_child_parent_overlap_partial(self, mock_logger):
        """Child module listed before parent: parent group excludes child's params."""
        config = {
            "vlm.visual": {"lr": 2e-5},
            "vlm": {"lr": 1e-5},
        }
        param_groups = build_optim_param_groups(self.model, config)

        visual_group = next(g for g in param_groups if g.get("name") == "vlm.visual")
        vlm_group = next(g for g in param_groups if g.get("name") == "vlm")

        visual_param_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("vlm.visual")
        )
        all_vlm_count = sum(
            1 for name, _ in self.model.named_parameters() if name.startswith("vlm")
        )
        self.assertEqual(len(visual_group["params"]), visual_param_count)
        self.assertEqual(len(vlm_group["params"]), all_vlm_count - visual_param_count)

    def test_no_duplicate_params_across_groups(self):
        """No parameter should appear in more than one group."""
        config = {
            "vlm.visual": {"lr": 2e-5},
            "vlm": {"lr": 1e-5},
            "action_model": {"lr": 1e-4},
        }
        param_groups = build_optim_param_groups(self.model, config)

        all_param_ids = []
        for group in param_groups:
            all_param_ids.extend(id(p) for p in group["params"])
        self.assertEqual(len(all_param_ids), len(set(all_param_ids)))


class TestSetupOptimizerEmptyParamGroups(unittest.TestCase):
    """Test setup_optimizer raises ValueError when all params are frozen."""

    def setUp(self):
        self.model = SimpleModel()

    def test_all_frozen_raises_value_error(self):
        freeze_config = MagicMock()
        freeze_config.freeze_patterns = [".*"]
        freeze_config.keep_patterns = None

        optimizer_config = MagicMock()
        optimizer_config.name = "AdamW"
        optimizer_config.param_groups = None
        optimizer_config.get_optimizer_kwargs.return_value = {"lr": 1e-4}

        with self.assertRaises(ValueError) as context:
            setup_optimizer(self.model, optimizer_config, freeze_config=freeze_config)

        self.assertIn("No trainable parameters found", str(context.exception))
