# Copyright (c) 2026, BAAI. All rights reserved.

"""Per-model grid communicator contracts and their registry (model-agnostic).

``build_grid_multimodule_communicator`` (``bridge.training``) needs three
model-specific values to construct the MCore
``MultiModulePipelineCommunicator``:

- the module dependency graph (which module feeds which),
- the schedule-tensor axis layout (where s/b/h live),
- each module's output dimensionality.

Those values are properties of the *model*, not of the grid machinery, so
they are registered here by the model's provider module (import-time
registration) instead of being hardcoded in ``bridge.training``.  Adding a
new model to the grid path means registering one contract from its own
providers/recipe module; this file and ``bridge.training`` stay untouched.

This module is a stdlib-only leaf: it must not import ``bridge.training``,
the providers, or torch, so both sides can import it without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GridCommunicatorContract",
    "SBH_DIM_MAPPING",
    "register_grid_communicator_contract",
    "get_grid_communicator_contract",
]


@dataclass(frozen=True)
class GridCommunicatorContract:
    """Communicator wiring description for one model family.

    Attributes:
        topology: module dependency graph as ``{module_name: [consumer, ...]}``
            (outgoing edges; sinks map to an empty list).  Must cover every
            module in the run's ``module_to_grid_map``.
        dim_mapping: axis positions of the schedule tensor's sequence, batch
            and hidden dims, e.g. ``{"s": 0, "b": 1, "h": 2}`` for SBH
            (seq-first) layout.
        module_output_ndim: output dimensionality per module.  Patch-packed
            encoders emit flat ``[tokens, H]`` tensors (2, fan-in/out on dim
            0); the language module emits 3D hidden states (3).
    """

    topology: dict[str, list[str]]
    dim_mapping: dict[str, int]
    module_output_ndim: dict[str, int]


#: Megatron-wide default axis layout for seq-first (SBH) schedule tensors;
#: model contracts should reuse this unless their tensors are laid out
#: differently.
SBH_DIM_MAPPING = {"s": 0, "b": 1, "h": 2}

_GRID_COMMUNICATOR_CONTRACTS: dict[str, GridCommunicatorContract] = {}


def register_grid_communicator_contract(key: str, contract: GridCommunicatorContract) -> None:
    """Register ``contract`` under ``key`` (called at provider import time).

    Raises:
        TypeError: if ``contract`` is not a :class:`GridCommunicatorContract`.
        ValueError: if ``key`` is already registered - duplicate registration
            almost always means a copy-paste error in the registering module.
    """
    if not isinstance(contract, GridCommunicatorContract):
        raise TypeError(
            f"grid communicator contract must be a GridCommunicatorContract, "
            f"got {type(contract).__name__}"
        )
    if key in _GRID_COMMUNICATOR_CONTRACTS:
        raise ValueError(
            f"grid communicator contract '{key}' is already registered; refusing to overwrite"
        )
    _GRID_COMMUNICATOR_CONTRACTS[key] = contract


def get_grid_communicator_contract(
    key: str | None = None,
) -> GridCommunicatorContract:
    """Look up a registered contract.

    With ``key=None`` (the common case: one model per process) the single
    registered contract is returned.  Lookup fails fast with an actionable
    message when nothing is registered (the model's providers module was not
    imported) or when several contracts are registered and no key is given.

    Raises:
        KeyError: ``key`` given but not registered.
        RuntimeError: zero contracts registered, or multiple registered with
            ``key=None``.
    """
    if key is not None:
        try:
            return _GRID_COMMUNICATOR_CONTRACTS[key]
        except KeyError:
            registered = sorted(_GRID_COMMUNICATOR_CONTRACTS) or ["<none>"]
            raise KeyError(
                f"no grid communicator contract registered under '{key}' "
                f"(registered: {registered}); import the model's providers "
                f"module to register it"
            ) from None
    if not _GRID_COMMUNICATOR_CONTRACTS:
        raise RuntimeError(
            "no grid communicator contract registered; import the model's "
            "providers module (e.g. bridge.providers.<model>) so it registers "
            "its contract before building the grid communicator"
        )
    if len(_GRID_COMMUNICATOR_CONTRACTS) > 1:
        raise RuntimeError(
            f"multiple grid communicator contracts registered "
            f"({sorted(_GRID_COMMUNICATOR_CONTRACTS)}); pass an explicit key"
        )
    return next(iter(_GRID_COMMUNICATOR_CONTRACTS.values()))
