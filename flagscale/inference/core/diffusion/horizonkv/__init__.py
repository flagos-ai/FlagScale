# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FlagScale's self-contained HorizonKV video inference integration.

The package keeps the public FlagScale adapters at this level.  The
``algorithm`` package contains sparse KV selection, cache/offload and overlap
execution, while ``wan`` contains the causal Wan model components.
CUDA extensions are loaded lazily only after the worker preflight succeeds.
"""
