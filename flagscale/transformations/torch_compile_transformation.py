import importlib

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx as fx

from omegaconf import DictConfig
from packaging import version
from packaging.version import Version
from torch import nn
from torch._dynamo.eval_frame import OptimizedModule

from flagscale.compilation.inductor_pass import InductorPass
from flagscale.runner.utils import logger
from flagscale.transformations.transformation import Selector, Transformation, build_selector


# Copied from https://github.com/vllm-project/vllm/blob/3f5a4b6473ad1948ec49d62abf2af48c7dc1c5c6/vllm/utils/torch_utils.py#L504
# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


@dataclass
class CompileOptions:
    """
    Options for the compilation.
    """

    mode: Optional[str] = "default"
    dynamic: Optional[bool] = True
    fullgraph: Optional[bool] = False
    disable: Optional[bool] = True


class PassManager(InductorPass):
    def __init__(self, passes: Iterable[Callable[[fx.Graph], None]]):
        self.passes: List[Callable[[fx.Graph], None]] = list(passes)

    def __call__(self, graph: torch.fx.Graph) -> None:
        for p in self.passes:
            p(graph)

    def uuid(self) -> str:
        state = {f"{type(p).__name__}_{idx}": p.uuid() for idx, p in enumerate(self.passes)}
        return InductorPass.hash_dict(state)

    def empty(self) -> bool:
        return len(self.passes) == 0


def init_backend(passes: List[InductorPass]):
    pass_manager = PassManager(passes)

    def backend(
        gm: fx.GraphModule,
        example_inputs: Sequence[Optional[Union[torch.Tensor, int, torch.SymInt]]],
    ) -> Any:
        previous = torch._inductor.config.post_grad_custom_post_pass

        if not pass_manager.empty():
            torch._inductor.config.post_grad_custom_post_pass = pass_manager
        try:
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)
        finally:
            torch._inductor.config.post_grad_custom_post_pass = previous

    return backend


# TODO(yupu): Check if we need to support multiple modules with different passes.
class TorchCompileTransformation(Transformation):
    def __init__(
        self,
        options: Optional[Dict[str, Any]] = None,
        passes: Optional[DictConfig] = None,
        targets: Optional[DictConfig] = None,
    ):
        super().__init__()
        self._options: CompileOptions = CompileOptions(**(options or {}))
        self._selector: Selector = build_selector(targets)

        from flagscale.transformations import create_passes_from_config

        self._passes: List[InductorPass] = create_passes_from_config(passes)

    def preflight(self) -> bool:
        if not is_torch_equal_or_newer("2.6.0"):
            logger.error(
                f"TorchCompileTransformation requires PyTorch >= 2.6, but got {torch.__version__}"
            )
            return False
        return True

    def targets(self, scope: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        return self._selector(scope)

    def apply(self, module: nn.Module) -> bool:
        logger.debug(f"Applying torch compile to module: {module.__class__.__name__}")

        if self._options.disable or isinstance(module, OptimizedModule):
            logger.info(
                f"Skipping torch compile for {module.__class__.__name__}: disabled or already compiled"
            )
            return True

        module.compile(
            backend=init_backend(self._passes),
            mode=self._options.mode,
            dynamic=self._options.dynamic,
            fullgraph=self._options.fullgraph,
        )

        return True
