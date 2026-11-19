# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small, rule-driven HF / MCore distributed-checkpoint conversion tools."""


class MappingError(RuntimeError):
    """A declared mapping cannot be executed on the supplied checkpoint."""
