#!/bin/bash
# =============================================================================
# FlagScale CUDA Environment Variables
# =============================================================================
#
# Self-contained environment setup for CUDA platform.
# Includes all common + CUDA-specific variables.
#
# Usage:
#   - Development: source tools/install/cuda/env.sh
#   - Docker: Sourced via /etc/profile.d/flagscale-env.sh
#
# Variables can be overridden by setting them before sourcing this file.
#
# FLAGSCALE_HOME is the root directory for all FlagScale installations:
#   - $FLAGSCALE_HOME/miniconda3  - Conda installation
#   - $FLAGSCALE_HOME/venv        - UV virtual environment
#   - $FLAGSCALE_HOME/deps        - Source dependencies (Megatron, etc.)
#   - $FLAGSCALE_HOME/downloads   - Cached downloads (miniconda, etc.)
# =============================================================================

# -----------------------------------------------------------------------------
# Root Installation Directory (single source of truth)
# -----------------------------------------------------------------------------
: "${FLAGSCALE_HOME:=/opt/flagscale}"

# -----------------------------------------------------------------------------
# Derived Paths (from FLAGSCALE_HOME)
# -----------------------------------------------------------------------------
: "${UV_PROJECT_ENVIRONMENT:=$FLAGSCALE_HOME/venv}"
: "${FLAGSCALE_CONDA:=$FLAGSCALE_HOME/miniconda3}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${MPI_HOME:=/usr/local/mpi}"

# -----------------------------------------------------------------------------
# CUDA Configuration
# -----------------------------------------------------------------------------
: "${CUDA_HOME:=/usr/local/cuda}"

# -----------------------------------------------------------------------------
# UV Configuration
# -----------------------------------------------------------------------------
: "${UV_HTTP_TIMEOUT:=500}"
: "${UV_INDEX_STRATEGY:=unsafe-best-match}"
: "${UV_LINK_MODE:=copy}"

# -----------------------------------------------------------------------------
# Export Variables
# -----------------------------------------------------------------------------
export FLAGSCALE_HOME FLAGSCALE_CONDA FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export UV_PROJECT_ENVIRONMENT MPI_HOME CUDA_HOME
export UV_HTTP_TIMEOUT UV_INDEX_STRATEGY UV_LINK_MODE
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

# -----------------------------------------------------------------------------
# PATH Configuration
# -----------------------------------------------------------------------------
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$FLAGSCALE_CONDA/bin:$HOME/.local/bin:$MPI_HOME/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib64:$MPI_HOME/lib:/usr/local/lib:$LD_LIBRARY_PATH"
