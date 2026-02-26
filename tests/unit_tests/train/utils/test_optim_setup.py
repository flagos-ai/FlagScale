import unittest

import torch
import torch.nn as nn

from flagscale.train.train_config import (
    FreezeConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from flagscale.train.utils.optim_setup import (
    PatternMatcher,
    apply_freeze_config,
    build_optim_param_groups,
    freeze_and_get_trainable_params,
    setup_optimizer,
    setup_scheduler,
)


class _DummyModel(nn.Module):
    """Simple model with named submodules for testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 8)
        self.decoder = nn.Linear(8, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))


class TestPatternMatcher(unittest.TestCase):
    def test_exact_match(self):
        m = PatternMatcher(["encoder"])
        self.assertTrue(m.matches("encoder.weight"))
        self.assertFalse(m.matches("decoder.weight"))

    def test_regex_match(self):
        m = PatternMatcher([r"encoder\..*"])
        self.assertTrue(m.matches("encoder.weight"))
        self.assertTrue(m.matches("encoder.bias"))
        self.assertFalse(m.matches("decoder.weight"))

    def test_unused_patterns(self):
        m = PatternMatcher(["encoder", "nonexistent"])
        m.matches("encoder.weight")
        self.assertEqual(m.get_unused_patterns(), ["nonexistent"])

    def test_multiple_patterns(self):
        m = PatternMatcher(["encoder", "decoder"])
        self.assertTrue(m.matches("encoder.weight"))
        self.assertTrue(m.matches("decoder.bias"))
        self.assertFalse(m.matches("head.weight"))


class TestFreezeAndGetTrainableParams(unittest.TestCase):
    def test_no_patterns_all_trainable(self):
        model = _DummyModel()
        params = list(freeze_and_get_trainable_params(model.named_parameters()))
        all_params = list(model.parameters())
        self.assertEqual(len(params), len(all_params))

    def test_freeze_encoder(self):
        model = _DummyModel()
        _ = list(
            freeze_and_get_trainable_params(model.named_parameters(), freeze_patterns=["encoder"])
        )
        # encoder has weight + bias = 2 params frozen
        for p in model.encoder.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.decoder.parameters():
            self.assertTrue(p.requires_grad)
        for p in model.head.parameters():
            self.assertTrue(p.requires_grad)

    def test_keep_overrides_freeze(self):
        model = _DummyModel()
        _ = list(
            freeze_and_get_trainable_params(
                model.named_parameters(),
                freeze_patterns=["encoder", "decoder"],
                keep_patterns=["decoder"],
            )
        )
        for p in model.encoder.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.decoder.parameters():
            self.assertTrue(p.requires_grad)


class TestApplyFreezeConfig(unittest.TestCase):
    def test_none_config_returns_all(self):
        model = _DummyModel()
        params = apply_freeze_config(model, None)
        self.assertEqual(len(params), len(list(model.parameters())))

    def test_freeze_config(self):
        model = _DummyModel()
        freeze_cfg = FreezeConfig(freeze_patterns=["head"])
        params = apply_freeze_config(model, freeze_cfg)
        # head params should be excluded
        head_param_ids = {id(p) for p in model.head.parameters()}
        for p in params:
            self.assertNotIn(id(p), head_param_ids)


class TestBuildOptimParamGroups(unittest.TestCase):
    def test_no_config_single_group(self):
        model = _DummyModel()
        groups = build_optim_param_groups(model)
        self.assertEqual(len(groups), 1)

    def test_named_groups(self):
        model = _DummyModel()
        groups = build_optim_param_groups(
            model,
            optim_param_groups_config={
                "encoder": {"lr": 1e-5},
                "decoder": {"lr": 1e-3},
            },
        )
        # Should have encoder, decoder, and default (head) groups
        names = {g.get("name") for g in groups}
        self.assertIn("encoder", names)
        self.assertIn("decoder", names)
        self.assertIn("default", names)

    def test_nonexistent_module_skipped(self):
        model = _DummyModel()
        groups = build_optim_param_groups(
            model,
            optim_param_groups_config={"nonexistent": {"lr": 1e-5}},
        )
        # Only default group
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["name"], "default")

    def test_no_double_counting(self):
        model = _DummyModel()
        groups = build_optim_param_groups(
            model,
            optim_param_groups_config={"encoder": {"lr": 1e-5}},
        )
        all_param_ids = set()
        for g in groups:
            for p in g["params"]:
                self.assertNotIn(id(p), all_param_ids)
                all_param_ids.add(id(p))


class TestSetupOptimizer(unittest.TestCase):
    def test_creates_adamw(self):
        model = _DummyModel()
        config = OptimizerConfig(name="AdamW", lr=1e-4, weight_decay=0.01)
        optimizer = setup_optimizer(model, config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_with_freeze(self):
        model = _DummyModel()
        config = OptimizerConfig(name="AdamW", lr=1e-4)
        freeze_cfg = FreezeConfig(freeze_patterns=["encoder"])
        optimizer = setup_optimizer(model, config, freeze_config=freeze_cfg)
        # Encoder params should not be in optimizer
        encoder_numel = sum(p.numel() for p in model.encoder.parameters())
        optimizer_numel = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        total_numel = sum(p.numel() for p in model.parameters())
        self.assertEqual(optimizer_numel, total_numel - encoder_numel)

    def test_unsupported_optimizer_raises(self):
        model = _DummyModel()
        config = OptimizerConfig(name="UnsupportedOpt", lr=1e-4)
        with self.assertRaises(ValueError):
            setup_optimizer(model, config)

    def test_with_param_groups(self):
        model = _DummyModel()
        config = OptimizerConfig(
            name="AdamW",
            lr=1e-4,
            param_groups={"encoder": {"lr": 1e-5}},
        )
        optimizer = setup_optimizer(model, config)
        # Should have default group + encoder group
        self.assertEqual(len(optimizer.param_groups), 2)


class TestSetupScheduler(unittest.TestCase):
    def test_cosine_scheduler(self):
        model = _DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = SchedulerConfig(name="cosine", warmup_steps=100)
        scheduler = setup_scheduler(optimizer, config, num_training_steps=1000)
        self.assertIsNotNone(scheduler)

    def test_linear_scheduler(self):
        model = _DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = SchedulerConfig(name="linear", warmup_steps=50)
        scheduler = setup_scheduler(optimizer, config, num_training_steps=500)
        # Step once and verify it doesn't crash
        scheduler.step()

    def test_none_name_raises(self):
        model = _DummyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = SchedulerConfig(name=None)
        with self.assertRaises(ValueError):
            setup_scheduler(optimizer, config, num_training_steps=1000)
