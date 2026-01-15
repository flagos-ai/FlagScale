import contextlib
import contextvars
import itertools
from collections.abc import Callable
from typing import Optional

import torch

# The current context in use
_current_ctx: contextvars.ContextVar["RuntimeContext | None"] = contextvars.ContextVar(
    "flagscale_runtime_ctx", default=None
)


class RuntimeContext:
    """Context shared across a model's forward pass.
    `Transformation`s could write/read information from this context.
    """

    def __init__(self, state_scopes: list[str] | None = None):
        if state_scopes is not None and len(state_scopes) > 0:
            # A provider of state scope names.
            # If set, the scope names would be used by `Transformation`s to access
            # different streams of the `StateStore`. The scope names will be cycled
            # through in order indefinitely.
            provider = itertools.cycle(state_scopes).__next__
            self.state_scope_provider: Callable[[], str | None] | None = provider
        else:
            self.state_scope_provider: Callable[[], str | None] | None = None

        # ==========================================
        #           DIFFUSION MODEL SETTINGS
        # ==========================================
        # The current timestep retrieved from the root module's input
        self.timestep: float = -1
        # The current timestep index. Range from 0 to num_inference_steps - 1
        self.timestep_index: int = -1

        # Callbacks invoked at session exit (LIFO order)
        self._on_exit_callbacks: list[Callable[[], None]] = []

    @contextlib.contextmanager
    def session(self) -> contextlib.AbstractContextManager["RuntimeContext"]:
        """Activate this context for the current call stack (process-local).
        Use once around the model call in each worker/rank.
        """

        token = _current_ctx.set(self)
        try:
            # Reset per-session runtime signals
            self.reset_session_state()
            yield self
        finally:
            # Run registered exit callbacks in reverse registration order
            for cb in reversed(self._on_exit_callbacks):
                try:
                    cb()
                except Exception:
                    # Best-effort cleanup; swallow errors to avoid masking user exceptions
                    pass
            self._on_exit_callbacks.clear()
            _current_ctx.reset(token)

    @classmethod
    def current(cls) -> Optional["RuntimeContext"]:
        """Get the current active context.

        Returns:
            Optional[RuntimeContext]: The current active context, or None if no context is active.
        """

        return _current_ctx.get()

    @property
    def state_scope(self) -> str | None:
        """Get the current state context name.

        Returns:
            Optional[str]: The current state context name, or None if no provider is set.
        """

        p = self.state_scope_provider
        return p() if callable(p) else None

    def update_timestep(self, t: torch.Tensor | int | float) -> None:
        """Update the current timestep only when a new timestep arrives.

        Args:
            t (Union[torch.Tensor, int, float]): The new timestep.
        """

        scalar_t: float | None = None
        if isinstance(t, torch.Tensor):
            if t.ndim == 0:
                scalar_t = float(t.item())
            else:
                scalar_t = float(t.flatten()[0].item())
        elif isinstance(t, (int, float)):
            scalar_t = float(t)
        else:
            raise ValueError(f"Unexpected timestep type: {type(t)}")

        # In case there are multiple model forward calls in the same timestep
        if self.timestep != scalar_t:
            self.timestep = scalar_t
            self.timestep_index += 1

    def reset_session_state(self) -> None:
        """Reset per-session fields to their initial values."""
        self.timestep = -1
        self.timestep_index = -1

    def add_on_exit(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when the session exits.

        Callbacks run in LIFO order. Errors in callbacks are swallowed to avoid
        masking exceptions from the user code.
        """
        self._on_exit_callbacks.append(callback)


def current_ctx() -> RuntimeContext | None:
    """Optional module-level alias for convenience."""

    return RuntimeContext.current()
