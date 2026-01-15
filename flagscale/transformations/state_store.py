# Modified from
# https://github.com/huggingface/diffusers/blob/4a7556eaecc9872dea50ce161301edfa6392693c/src/diffusers/hooks/hooks.py

from typing import Generic, TypeVar

S = TypeVar("S")


class BaseState:
    """Base class for states."""

    def reset(self, *args, **kwargs):
        """Reset the state."""
        raise NotImplementedError(
            "BaseState::reset is not implemented. Please implement this method in the derived class."
        )


class StateStore(Generic[S]):
    """A state store that manages states for different contexts."""

    def __init__(self, state_cls: type[S], init_args=None, init_kwargs=None) -> None:
        self._state_cls: type[S] = state_cls
        # TODO(yupu): Fixed args/kwargs?
        self._init_args = init_args if init_args is not None else ()
        self._init_kwargs = init_kwargs if init_kwargs is not None else {}
        # Mapping: state scope name -> state.
        # State can be shared across modules or by a single module.
        self._state_by_scope: dict[str, S] = {}
        # The state scope name that is currently in use.
        self._active_scope: str | None = None

    def get_or_create_state(self) -> S:
        """Get or create a state for the current scope.

        Returns:
            The state for the current scope.

        Raises:
            ValueError: If no state scope is currently in use.
        """
        if self._active_scope is None:
            raise ValueError("No state scope is currently in use. Please set a state scope first.")
        if self._active_scope not in self._state_by_scope:
            self._state_by_scope[self._active_scope] = self._state_cls(
                *self._init_args, **self._init_kwargs
            )
        return self._state_by_scope[self._active_scope]

    def set_scope(self, name: str):
        """Set the current state scope.

        Args:
            name: The name of the state scope to set.
        """
        self._active_scope = name

    def reset(self, *args, **kwargs):
        """Reset the state store."""
        for scope, state in list(self._state_by_scope.items()):
            state.reset(*args, **kwargs)
            self._state_by_scope.pop(scope)
        self._active_scope = None
