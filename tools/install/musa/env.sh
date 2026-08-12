#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# FlagScale MUSA Environment Variables
# =============================================================================

: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${MPI_HOME:=/usr/local/openmpi}"
: "${MUSA_HOME:=/usr/local/musa}"

export FLAGSCALE_HOME FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export MPI_HOME MUSA_HOME

export PATH="$MUSA_HOME/bin:$MPI_HOME/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MPI_HOME/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
