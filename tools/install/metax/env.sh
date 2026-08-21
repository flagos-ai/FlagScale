#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${FLAGSCALE_CONDA:=/opt/conda}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${UV_PROJECT_ENVIRONMENT:=$FLAGSCALE_HOME/venv}"
: "${MPI_HOME:=/usr/local/mpi}"
: "${MACA_HOME:=/opt/maca}"
: "${CUDA_HOME:=$MACA_HOME}"
: "${TE_FL_SKIP_CUDA:=1}"
: "${NVTE_WITH_MACA:=1}"

export FLAGSCALE_HOME FLAGSCALE_CONDA FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export UV_PROJECT_ENVIRONMENT MPI_HOME MACA_HOME CUDA_HOME
export TE_FL_SKIP_CUDA NVTE_WITH_MACA
export PATH="$FLAGSCALE_CONDA/bin:$HOME/.local/bin:$MPI_HOME/bin:$MACA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib64:$MPI_HOME/lib:$MACA_HOME/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
