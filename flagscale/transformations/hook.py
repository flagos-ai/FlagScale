# Modified from
# https://github.com/huggingface/diffusers/blob/4a7556eaecc9872dea50ce161301edfa6392693c/src/diffusers/hooks/hooks.py

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from flagscale.transformations.state_store import StateStore


# Copied from
# https://github.com/huggingface/diffusers/blob/4a7556eaecc9872dea50ce161301edfa6392693c/src/diffusers/utils/torch_utils.py
def is_compiled_module(module) -> bool:
    """Check whether the module was compiled with torch.compile()"""
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def unwrap_module(module):
    """Unwraps a module if it was compiled with torch.compile()"""
    return module._orig_mod if is_compiled_module(module) else module


class ModelHook:
    """
    Hook applied to a module.
    """

    def __init__(self) -> None:
        self.fn_ref: HookFunctionReference = None
        # A list of `StateStore`s that the hook has access to.
        # The stores could be shared across multiple hooks from the same transform.
        # The transform should call `register_stateful` to register the stores.
        self._stateful: list[StateStore[Any]] = []

    def on_attach(self, module: nn.Module) -> nn.Module:
        """
        Called when the hook is attached to a module.
        """
        return module

    def on_detach(self, module: nn.Module) -> nn.Module:
        """
        Called when the hook is detached from a module.
        """
        return module

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple[Any], dict[str, Any]]:
        """
        Called before the module's forward pass.
        """
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """
        Called after the module's forward pass.
        """
        return output

    def custom_forward(self, module: nn.Module, *args, **kwargs) -> Any:
        """
        Optional: Override the module's forward.

        Override this method in subclasses to replace the module's forward. The signature must be
        `custom_forward(module, *args, **kwargs)` and return the module output.
        """
        # Default behavior: delegate to the original forward. Subclasses may override.
        return module.forward(*args, **kwargs)

    def register_stateful(self, state_store: StateStore[Any]) -> None:
        """
        Register a `StateStore` for the hook.
        """
        self._stateful.append(state_store)

    def set_state_scope(self, name: str | None = None) -> None:
        """
        Set the state scope for the hook.
        """
        for state_store in self._stateful:
            state_store.set_scope(name)

    def reset_stateful(self) -> None:
        """
        Reset all the state_stores for the hook.
        """
        for state_store in self._stateful:
            state_store.reset()


@dataclass
class HookFunctionReference:
    """
    Mutable references for spliceable chains
    """

    pre_forward: Callable[..., Any] | None = None
    post_forward: Callable[..., Any] | None = None
    forward: Callable[..., Any] | None = None
    original_forward: Callable[..., Any] | None = None


class ModuleHookRegistry:
    """Per-module registry. Attaching a hook immediately wraps module's forward"""

    def __init__(self, module_ref: nn.Module) -> None:
        # A reference to the module
        self._module_ref = module_ref
        # name -> hook
        self._hooks: dict[str, ModelHook] = {}
        # The order of hooks registered
        self._order: list[str] = []
        # A ordered list of applied forward functions for each hook
        self._fn_refs: list[HookFunctionReference] = []

    @classmethod
    def get_or_create_registry(cls, module: nn.Module) -> "ModuleHookRegistry":
        """
        Get or create a registry for the module.

        Args:
            module: The module to get or create a registry for.

        Returns:
            The registry for the module.
        """
        reg = ModuleHookRegistry.get_registry_if_present(module)
        if reg is None:
            reg = cls(module)
            module._flagscale_hooks = reg
        return reg

    @classmethod
    def get_registry_if_present(cls, module: nn.Module) -> Optional["ModuleHookRegistry"]:
        """Return existing registry if present, without creating a new one."""
        reg = vars(module).get("_flagscale_hooks")
        return reg if isinstance(reg, ModuleHookRegistry) else None

    def register_hook(self, hook: ModelHook, name: str) -> None:
        """Register a hook on the module.

        Args:
            hook: The hook to register.
            name: The name of the hook, which should be unique within the registry.
        """
        if name in self._hooks:
            raise ValueError(
                f"Hook with name {name} already exists in the registry. Please use a different name."
            )

        # Let hook adjust module if needed
        self._module_ref = hook.on_attach(self._module_ref)

        fn_ref = HookFunctionReference(
            pre_forward=hook.pre_forward,
            post_forward=hook.post_forward,
            forward=self._module_ref.forward,
        )

        # If custom_forward is overridden by subclass, use it to replace forward
        if type(hook).custom_forward is not ModelHook.custom_forward:
            fn_ref.original_forward = self._module_ref.forward
            custom = functools.update_wrapper(
                functools.partial(hook.custom_forward, self._module_ref), hook.custom_forward
            )
            fn_ref.forward = custom

        # Build the wrapped forward for this hook
        def make_wrapped(fn_ref: HookFunctionReference):
            def wrapped(module: nn.Module, *args, **kwargs):
                args, kwargs = fn_ref.pre_forward(module, *args, **kwargs)
                output = fn_ref.forward(*args, **kwargs)
                return fn_ref.post_forward(module, output)

            return wrapped

        rewritten = make_wrapped(fn_ref)
        new_forward = functools.update_wrapper(
            functools.partial(rewritten, self._module_ref), rewritten
        )
        self._module_ref.forward = new_forward

        # Track ordering and links
        hook.fn_ref = fn_ref
        fn_ref._name = name
        self._hooks[name] = hook
        self._order.append(name)
        self._fn_refs.append(fn_ref)

    def get_hook(self, name: str) -> ModelHook | None:
        """Get a hook by name.

        Args:
            name: The name of the hook.

        Returns:
            The hook, or None if the hook is not found.
        """
        return self._hooks.get(name, None)

    def remove_hook(self, name: str, recursive: bool = True) -> None:
        """Remove a hook by name.

        Args:
            name: The name of the hook.
            recursive: Whether to remove the hook recursively.
        """
        if name not in self._hooks:
            return

        hook = self._hooks[name]
        idx = self._order.index(name)
        fn_ref = self._fn_refs[idx]

        # Determine which forward to restore at this splice point
        old_forward = (
            fn_ref.original_forward if fn_ref.original_forward is not None else fn_ref.forward
        )

        # Splice: if removing the last link, restore module.forward,
        # otherwise point the next link's forward to the restored function
        if idx == len(self._fn_refs) - 1:
            self._module_ref.forward = old_forward
        else:
            self._fn_refs[idx + 1].forward = old_forward

        # Allow hook to clean up and possibly adjust module
        self._module_ref = hook.on_detach(self._module_ref)

        # Drop bookkeeping
        self._hooks.pop(name, None)
        self._order.pop(idx)
        self._fn_refs.pop(idx)

        if recursive:
            # This happens before torch.compile, so calling `unwrap_module` is not necessary.
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                reg = ModuleHookRegistry.get_registry_if_present(module)
                if reg is not None:
                    reg.remove_hook(name, recursive=False)

    def reset_stateful_hooks(self, recursive: bool = True) -> None:
        """
        Reset all the stateful hooks for the registry.
        """
        for hook_name in reversed(self._order):
            hook = self._hooks[hook_name]
            hook.reset_stateful()

        if recursive:
            for module_name, module in unwrap_module(self._module_ref).named_modules():
                if module_name == "":
                    continue
                module = unwrap_module(module)
                reg = ModuleHookRegistry.get_registry_if_present(module)
                if reg is not None:
                    reg.reset_stateful_hooks(recursive=False)

    def set_state_scope(self, name: str | None = None) -> None:
        # TODO(yupu): Does the order matter?
        for hook_name in reversed(self._order):
            self._hooks[hook_name].set_state_scope(name)

        for module_name, module in unwrap_module(self._module_ref).named_modules():
            if module_name == "":
                continue
            reg = ModuleHookRegistry.get_registry_if_present(module)
            if reg is not None:
                reg.set_state_scope(name)

    def __repr__(self) -> str:
        registry_repr = ""
        for i, hook_name in enumerate(self._order):
            if self._hooks[hook_name].__class__.__repr__ is not object.__repr__:
                hook_repr = self._hooks[hook_name].__repr__()
            else:
                hook_repr = self._hooks[hook_name].__class__.__name__
            registry_repr += f"  ({i}) {hook_name} - {hook_repr}"
            if i < len(self._order) - 1:
                registry_repr += "\n"
        return f"ModuleHookRegistry(\n{registry_repr}\n)"
