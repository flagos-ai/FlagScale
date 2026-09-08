# Copyright (c) 2026, BAAI. All rights reserved.

"""MIMO component parallelism layout configuration (FlagScale-native).

Standalone, dependency-free (stdlib-only, no torch/Megatron) configuration
for MIMO layouts: in the non-colocated layout each module (``vision``,
``language``, ...) owns a disjoint, contiguous rank span given by
``rank_offset`` plus ``total_ranks``, and the modules tile ``[0, world_size)``
exactly; the colocated layout has every module span the full world.  Mirrors
the API shape of the Megatron-Bridge ``megatron_mimo_config`` (same field
names and derived sizing helpers) in FlagScale style, with immutable configs
- all validation happens at construction / ``finalize`` time, no post-hoc
mutation.

Key API: :class:`ModuleParallelismConfig` (frozen, self-validating per-module
sizing plus rank/world helpers), :class:`MIMOLayout` (``COLOCATED`` /
``NON_COLOCATED`` / ``AUTO``), :class:`MIMOParallelismConfig` (one config per
module; ``finalize(world_size, layout=...)`` validates layout invariants),
:func:`classify_layout`, and :func:`parse_module_parallelism` /
:func:`parse_module_parallelisms` (parsers for repeatable spec strings such
as ``language=tp=4,pp=2,dp=2,rank_offset=0``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Mandatory language module name; kept in sync with the MIMO scheduler and
# Bridge communicator assumptions.
LANGUAGE_MODULE_NAME = "language"

_SIZING_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "data_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
)

# Accepted keys in spec strings, with short aliases (tp/cp/pp/dp/ep/etp).
_MODULE_KEY_ALIASES = {
    "tp": "tensor_model_parallel_size",
    "tensor_model_parallel_size": "tensor_model_parallel_size",
    "cp": "context_parallel_size",
    "context_parallel_size": "context_parallel_size",
    "pp": "pipeline_model_parallel_size",
    "pipeline_model_parallel_size": "pipeline_model_parallel_size",
    "dp": "data_parallel_size",
    "data_parallel_size": "data_parallel_size",
    "ep": "expert_model_parallel_size",
    "expert_model_parallel_size": "expert_model_parallel_size",
    "etp": "expert_tensor_parallel_size",
    "expert_tensor_parallel_size": "expert_tensor_parallel_size",
    "rank_offset": "rank_offset",
}

_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ModuleParallelismConfig:
    """Immutable, validated parallelism sizing for a single MIMO module.

    All sizes are positive integers and ``rank_offset`` is non-negative;
    validation runs in ``__post_init__``, so an invalid config can never be
    constructed.  Expert parallelism (``ep``/``etp``) subdivides the dense
    token domain as ``tp * cp * dp = ep * etp * expert_dp`` and therefore
    does not appear in the module's rank span product.
    """

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    rank_offset: int = 0

    def __post_init__(self) -> None:
        for field_name in _SIZING_FIELDS:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
        rank_offset = self.rank_offset
        if isinstance(rank_offset, bool) or not isinstance(rank_offset, int) or rank_offset < 0:
            raise ValueError(f"rank_offset must be a non-negative integer, got {rank_offset!r}.")
        self._validate_expert_factorization()

    @property
    def dense_model_parallel_size(self) -> int:
        """Dense/token model-parallel size: TP * CP * PP."""
        return (
            self.tensor_model_parallel_size
            * self.context_parallel_size
            * self.pipeline_model_parallel_size
        )

    @property
    def total_model_parallel_size(self) -> int:
        """Backward-compatible alias for the dense/token model-parallel size."""
        return self.dense_model_parallel_size

    @property
    def total_ranks(self) -> int:
        """Number of ranks this module spans: dense MP size * DP."""
        return self.dense_model_parallel_size * self.data_parallel_size

    @property
    def rank_span(self) -> int:
        """Alias of ``total_ranks`` (the module's span in the world)."""
        return self.total_ranks

    @property
    def rank_end(self) -> int:
        """Exclusive end of this module's rank range."""
        return self.rank_offset + self.total_ranks

    @property
    def rank_range(self) -> tuple[int, int]:
        """Half-open ``[rank_offset, rank_end)`` rank range of this module."""
        return (self.rank_offset, self.rank_end)

    @property
    def expert_model_parallel_span(self) -> int:
        """Rank span consumed by expert parallelism: ETP * EP * PP."""
        return (
            self.expert_tensor_parallel_size
            * self.expert_model_parallel_size
            * self.pipeline_model_parallel_size
        )

    @property
    def expert_data_parallel_size(self) -> int:
        """Derived expert data-parallel size over this module's rank span."""
        span = self.expert_model_parallel_span
        if self.total_ranks % span != 0:
            raise ValueError(
                "total_ranks must be divisible by expert_tensor_parallel_size * "
                "expert_model_parallel_size * pipeline_model_parallel_size; got "
                f"total_ranks={self.total_ranks}, span={span}."
            )
        return self.total_ranks // span

    def _validate_expert_factorization(self) -> None:
        dense_token_span = (
            self.tensor_model_parallel_size * self.context_parallel_size * self.data_parallel_size
        )
        expert_span = self.expert_tensor_parallel_size * self.expert_model_parallel_size
        if dense_token_span % expert_span != 0:
            raise ValueError(
                "TP * CP * DP must be divisible by expert_tensor_parallel_size * "
                "expert_model_parallel_size; got "
                f"TP={self.tensor_model_parallel_size}, "
                f"CP={self.context_parallel_size}, "
                f"DP={self.data_parallel_size}, "
                f"ETP={self.expert_tensor_parallel_size}, "
                f"EP={self.expert_model_parallel_size}."
            )


class MIMOLayout(Enum):
    """Deployment layout of MIMO modules over the distributed world."""

    COLOCATED = "colocated"
    NON_COLOCATED = "non_colocated"
    AUTO = "auto"


def classify_layout(
    module_parallelisms: Mapping[str, ModuleParallelismConfig], world_size: int
) -> MIMOLayout:
    """Auto-classify a module set as colocated or non-colocated.

    Colocated when every module spans the full world (``rank_offset == 0``
    and ``total_ranks == world_size``), non-colocated otherwise.  Only a
    heuristic - the exact tiling / full-world invariants are enforced by
    :meth:`MIMOParallelismConfig.finalize`.
    """
    if not module_parallelisms:
        raise ValueError("classify_layout requires at least one module.")
    all_full_world = all(
        parallelism.rank_offset == 0 and parallelism.total_ranks == world_size
        for parallelism in module_parallelisms.values()
    )
    return MIMOLayout.COLOCATED if all_full_world else MIMOLayout.NON_COLOCATED


@dataclass(frozen=True)
class MIMOParallelismConfig:
    """Container of per-module parallelism configs for a MIMO deployment.

    Holds one :class:`ModuleParallelismConfig` per module (the language
    module is mandatory).  Per-module sizing is validated at construction of
    the module configs; :meth:`finalize` validates the cross-module
    invariants for the requested (or auto-classified) :class:`MIMOLayout`:
    ``NON_COLOCATED`` requires module rank ranges to tile ``[0, world_size)``
    exactly (no gaps, no overlaps); ``COLOCATED`` requires every module to
    span the full world.  Both layouts additionally require TP powers of
    two, pairwise-divisible DP sizes, and dense (EP == ETP == 1) modality
    modules for cross-module communication compatibility.
    """

    module_parallelisms: Mapping[str, ModuleParallelismConfig]
    layout: MIMOLayout = MIMOLayout.AUTO

    def __post_init__(self) -> None:
        if not self.module_parallelisms:
            raise ValueError("module_parallelisms must contain at least one module.")
        for name, parallelism in self.module_parallelisms.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"module name must be a non-empty string, got {name!r}.")
            if not isinstance(parallelism, ModuleParallelismConfig):
                raise TypeError(
                    f"module '{name}' must be a ModuleParallelismConfig, got "
                    f"{type(parallelism).__name__}."
                )
        object.__setattr__(
            self, "module_parallelisms", MappingProxyType(dict(self.module_parallelisms))
        )
        object.__setattr__(self, "layout", self._coerce_layout(self.layout))

    @classmethod
    def from_specs(
        cls,
        specs: str | Iterable[str],
        layout: MIMOLayout | str | None = None,
    ) -> MIMOParallelismConfig:
        """Build a config from repeatable spec strings (see :func:`parse_module_parallelisms`)."""
        return cls(
            module_parallelisms=parse_module_parallelisms(specs),
            layout=layout or MIMOLayout.AUTO,
        )

    @staticmethod
    def _coerce_layout(layout: MIMOLayout | str) -> MIMOLayout:
        if isinstance(layout, MIMOLayout):
            return layout
        if isinstance(layout, str):
            normalized = layout.strip().lower().replace("-", "_")
            for member in MIMOLayout:
                if member.value == normalized or member.name.lower() == normalized:
                    return member
            raise ValueError(
                f"unknown MIMO layout {layout!r}; expected one of "
                f"{[member.value for member in MIMOLayout]}."
            )
        raise TypeError(f"layout must be a MIMOLayout or str, got {type(layout).__name__}.")

    @property
    def module_names(self) -> list:
        return list(self.module_parallelisms.keys())

    @property
    def total_world_size(self) -> int:
        """Covered world size: max rank end across modules (0 if empty)."""
        ends = [parallelism.rank_end for parallelism in self.module_parallelisms.values()]
        return max(ends) if ends else 0

    @property
    def rank_ranges(self) -> list:
        """Sorted ``(start, end, name)`` rank ranges of all modules."""
        ranges = [
            (parallelism.rank_offset, parallelism.rank_end, name)
            for name, parallelism in self.module_parallelisms.items()
        ]
        ranges.sort(key=lambda item: item[0])
        return ranges

    def get_parallelism(self, module_name: str) -> ModuleParallelismConfig:
        return self.module_parallelisms[module_name]

    def finalize(
        self,
        world_size: int,
        layout: MIMOLayout | str | None = None,
    ) -> None:
        """Validate the full layout against ``world_size``.

        Args:
            world_size: total ranks; must be a positive integer (MIMO
                requires a distributed environment).
            layout: optional override of the container layout; ``AUTO``
                classifies via :func:`classify_layout`.
        """
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError(f"world_size must be a positive integer, got {world_size!r}.")
        if LANGUAGE_MODULE_NAME not in self.module_parallelisms:
            raise ValueError(
                f"Language module '{LANGUAGE_MODULE_NAME}' must be in module_parallelisms. "
                f"Found modules: {self.module_names}"
            )

        resolved = self._coerce_layout(self.layout if layout is None else layout)
        if resolved is MIMOLayout.AUTO:
            resolved = classify_layout(self.module_parallelisms, world_size)

        self._validate_parallelism_constraints()
        self._validate_encoder_expert_parallelism()
        if resolved is MIMOLayout.COLOCATED:
            self._validate_colocated_full_world(world_size)
        else:
            self._validate_non_colocated_tiling(world_size)

    def _validate_non_colocated_tiling(self, world_size: int) -> None:
        """Ranges must tile ``[0, world_size)`` exactly: no gaps, no overlaps."""
        expected_start = 0
        for start, end, name in self.rank_ranges:
            if start < expected_start:
                raise ValueError(
                    "module rank ranges must tile the world with no overlaps: "
                    f"module '{name}' starts at rank {start}, but the previous "
                    f"module already covers up to rank {expected_start}."
                )
            if start > expected_start:
                raise ValueError(
                    "module rank ranges must tile the world with no gaps: "
                    f"expected module '{name}' to start at rank {expected_start}, "
                    f"got {start}."
                )
            expected_start = end
        if expected_start != world_size:
            raise ValueError(
                "module rank ranges must tile the world with no gaps: covered "
                f"[0, {expected_start}), but world_size is {world_size}."
            )

    def _validate_colocated_full_world(self, world_size: int) -> None:
        """Every module must span the full world."""
        for name, parallelism in self.module_parallelisms.items():
            if parallelism.rank_offset != 0:
                raise ValueError(
                    "colocated MIMO requires every module to span the full world: "
                    f"module '{name}' has rank_offset={parallelism.rank_offset}, "
                    "must be 0."
                )
            if parallelism.total_ranks != world_size:
                raise ValueError(
                    "colocated MIMO requires every module to span the full world: "
                    f"module '{name}' spans {parallelism.total_ranks} ranks, "
                    f"world_size is {world_size}."
                )

    def _validate_encoder_expert_parallelism(self) -> None:
        """Modality (non-language) modules must remain dense.

        The mcore MIMO MoE machinery only supports expert parallelism on the
        language module; a modality module with EP/ETP > 1 passes the
        per-module factorization algebra but is unsupported at wiring time
        and would fail obscurely.  Mirrors the Megatron-Bridge
        ``MegatronMIMOParallelismConfig._validate_encoder_expert_parallelism``
        until EP > 1 modality support lands.
        """
        for name, parallelism in self.module_parallelisms.items():
            if name == LANGUAGE_MODULE_NAME:
                continue
            if (
                parallelism.expert_model_parallel_size != 1
                or parallelism.expert_tensor_parallel_size != 1
            ):
                raise ValueError(
                    f"Module '{name}' is not the language module and must remain "
                    "dense for MIMO: expert_model_parallel_size and "
                    "expert_tensor_parallel_size must both be 1, got "
                    f"EP={parallelism.expert_model_parallel_size}, "
                    f"ETP={parallelism.expert_tensor_parallel_size}."
                )

    def _validate_parallelism_constraints(self) -> None:
        """Cross-module communication compatibility constraints.

        TP sizes must be powers of 2; DP sizes must be pairwise divisible
        (one divides the other).
        """

        def is_power_of_two(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0

        for name, parallelism in self.module_parallelisms.items():
            tp = parallelism.tensor_model_parallel_size
            if not is_power_of_two(tp):
                raise ValueError(
                    f"Module '{name}' has TP={tp}, but TP size must be a power "
                    "of 2 (1, 2, 4, 8, ...) for cross-module communication "
                    "compatibility."
                )

        module_names = list(self.module_parallelisms.keys())
        for idx, name1 in enumerate(module_names):
            for name2 in module_names[idx + 1 :]:
                dp1 = self.module_parallelisms[name1].data_parallel_size
                dp2 = self.module_parallelisms[name2].data_parallel_size
                if dp1 % dp2 != 0 and dp2 % dp1 != 0:
                    raise ValueError(
                        "DP sizes must be pairwise divisible between modules; "
                        f"module '{name1}' has DP={dp1}, module '{name2}' has "
                        f"DP={dp2} - one must divide the other for "
                        "cross-module communication."
                    )


def parse_module_parallelism(spec: str) -> tuple[str, ModuleParallelismConfig]:
    """Parse one module spec into ``(name, config)``.

    Spec format: ``name=tp=4,pp=2,dp=2,rank_offset=0`` - the module name
    followed by comma-separated ``key=value`` pairs.  Keys accept short
    aliases (``tp``/``cp``/``pp``/``dp``/``ep``/``etp``) or the full field
    names; unspecified sizes default to 1 and ``rank_offset`` defaults to 0.
    """
    spec = spec.strip()
    name, sep, body = spec.partition("=")
    name = name.strip()
    if not sep or not body:
        raise ValueError(
            f"invalid module spec {spec!r}; expected 'name=key=value,...', e.g. "
            "'language=tp=4,pp=2,dp=2,rank_offset=0'."
        )
    if not _MODULE_NAME_RE.match(name):
        raise ValueError(
            f"invalid module name {name!r} in spec {spec!r}; names must match "
            f"{_MODULE_NAME_RE.pattern}."
        )

    kwargs = {}
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, raw_value = part.partition("=")
        if not sep:
            raise ValueError(
                f"invalid parallelism pair {part!r} in spec {spec!r}; expected 'key=value'."
            )
        field_name = _MODULE_KEY_ALIASES.get(key.strip())
        if field_name is None:
            raise ValueError(
                f"unknown parallelism key {key.strip()!r} in spec {spec!r}; "
                f"expected one of {sorted(_MODULE_KEY_ALIASES)}."
            )
        if field_name in kwargs:
            raise ValueError(f"duplicate parallelism key {key.strip()!r} in spec {spec!r}.")
        try:
            kwargs[field_name] = int(raw_value.strip())
        except ValueError:
            raise ValueError(
                f"parallelism value {raw_value.strip()!r} for key "
                f"{key.strip()!r} in spec {spec!r} must be an integer."
            ) from None

    return name, ModuleParallelismConfig(**kwargs)


def parse_module_parallelisms(specs: str | Iterable[str]) -> dict:
    """Parse repeatable module spec strings into ``{name: config}``.

    ``specs`` may be a single ``";"``-separated string (whitespace around
    separators is tolerated) or an iterable of spec strings, e.g.::

        "vision=tp=1,dp=4; language=tp=4,pp=2,dp=2,rank_offset=4"
    """
    if isinstance(specs, str):
        parts = [part.strip() for part in specs.split(";") if part.strip()]
    else:
        parts = [part.strip() for part in specs if part.strip()]

    result = {}
    for part in parts:
        name, config = parse_module_parallelism(part)
        if name in result:
            raise ValueError(f"duplicate module name {name!r} in specs {specs!r}.")
        result[name] = config
    return result
