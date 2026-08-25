#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

# The inference image already owns the complete Hygon serving runtime.
python - <<'PY'
import importlib.metadata as metadata

assert metadata.version("vllm").startswith("0.20.2")
assert metadata.version("vllm-plugin-fl")
PY
