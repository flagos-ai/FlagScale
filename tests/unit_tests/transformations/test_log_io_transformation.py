from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from flagscale.transformations.hook import ModuleHookRegistry
from flagscale.transformations.log_io_transformation import (
    LogIOHook,
    LogIOTransformation,
    _shape_of,
)


class EchoModule(nn.Module):
    def forward(self, x, scale=1):
        return x * scale


class TupleModule(nn.Module):
    def forward(self, x):
        return x, x + 1


def test_shape_of_tensor_and_non_tensor_values():
    tensor = torch.zeros(2, 3)

    assert _shape_of(tensor) == "N/A: torch.Size([2, 3])"
    assert _shape_of(5, "value") == "value: int"


def test_log_io_hook_rejects_invalid_log_level():
    with pytest.raises(ValueError, match="Invalid log level"):
        LogIOHook(log_level="not_a_level")


def test_log_io_hook_logs_inputs_and_single_output():
    log = MagicMock()
    with patch("flagscale.transformations.log_io_transformation.logger") as logger:
        logger.debug = log
        hook = LogIOHook(log_level="debug")

    module = EchoModule()
    x = torch.ones(1, 2)
    args, kwargs = hook.pre_forward(module, x, scale=3)
    output = hook.post_forward(module, module(*args, **kwargs))

    assert torch.equal(output, x * 3)
    assert "input shapes" in log.call_args_list[0].args[0]
    assert "scale: int" in log.call_args_list[0].args[0]
    assert "output shape" in log.call_args_list[1].args[0]


def test_log_io_hook_logs_tuple_outputs():
    log = MagicMock()
    with patch("flagscale.transformations.log_io_transformation.logger") as logger:
        logger.info = log
        hook = LogIOHook(log_level="info")

    x = torch.ones(1, 2)
    output = TupleModule()(x)
    assert hook.post_forward(TupleModule(), output) is output
    assert "output shape" in log.call_args.args[0]
    assert "torch.Size([1, 2])" in log.call_args.args[0]


def test_log_io_transformation_applies_hook_and_duplicate_apply_fails():
    module = EchoModule()
    transform = LogIOTransformation(log_level="debug")

    assert transform.apply(module) is True
    reg = ModuleHookRegistry.get_registry_if_present(module)
    assert isinstance(reg.get_hook("log_io"), LogIOHook)

    with pytest.raises(ValueError, match="Hook with name log_io already exists"):
        transform.apply(module)


def test_log_io_transformation_hook_runs_during_forward():
    module = EchoModule()
    log = MagicMock()
    with patch("flagscale.transformations.log_io_transformation.logger") as logger:
        logger.info = log
        LogIOTransformation(log_level="info").apply(module)
        result = module(torch.ones(1, 2), scale=2)

    assert torch.equal(result, torch.full((1, 2), 2.0))
    assert log.call_count == 2
    assert "EchoModule input shapes" in log.call_args_list[0].args[0]
    assert "EchoModule output shape" in log.call_args_list[1].args[0]
