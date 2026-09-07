#!/bin/bash
# =============================================================================
# FlagScale Kunlunxin Environment Variables
# =============================================================================

: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${UV_PROJECT_ENVIRONMENT:=$FLAGSCALE_HOME/venv}"
: "${FLAGSCALE_CONDA:=/root/miniconda}"
: "${FLAGSCALE_ENV_NAME:=python310_torch29_cuda}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${FLAGSCALE_DOWNLOADS:=$FLAGSCALE_HOME/downloads}"
: "${MPI_HOME:=/usr/local/mpi}"
: "${KLX_HOME:=/opt/xccl_Linux_x86_64}"
: "${FLAGSCALE_MEGATRON_PATH:=$FLAGSCALE_DEPS/Megatron-LM-FL}"

# Kunlunxin XRE runtime libraries are required by the FlagCX/XCCL training
# communication stack. The directory is versioned and has no stable symlink,
# so discover the installed path at runtime.
if [ -z "${XRE_HOME:-}" ]; then
    for _d in /opt/xre-Linux-x86_64-*; do
        [ -d "$_d/so" ] && XRE_HOME="$_d" && break
    done
fi
# FlagCX may be installed editable or under /opt/FlagCX, depending on the
# training base image. Add its source directory to PYTHONPATH so the training
# stack and image verification import the same wrapper. A caller-set value wins.
if [ -z "${FLAGCX_PATH:-}" ]; then
    for _el in \
        "$FLAGSCALE_CONDA"/envs/"$FLAGSCALE_ENV_NAME"/lib/python*/site-packages/flagcx.egg-link \
        /usr/lib/python*/site-packages/flagcx.egg-link; do
        [ -f "$_el" ] || continue
        _dir=$(head -1 "$_el" 2>/dev/null)
        [ -n "$_dir" ] && [ -d "$_dir" ] && FLAGCX_PATH="$_dir" && break
    done
fi
: "${FLAGCX_PATH:=/opt/FlagCX}"

: "${UV_HTTP_TIMEOUT:=500}"
: "${UV_INDEX_STRATEGY:=unsafe-best-match}"
: "${UV_LINK_MODE:=copy}"
: "${TE_FL_SKIP_CUDA:=1}"

export FLAGSCALE_HOME FLAGSCALE_CONDA FLAGSCALE_ENV_NAME FLAGSCALE_DEPS FLAGSCALE_DOWNLOADS
export UV_PROJECT_ENVIRONMENT MPI_HOME KLX_HOME XRE_HOME FLAGCX_PATH FLAGSCALE_MEGATRON_PATH TE_FL_SKIP_CUDA
export UV_HTTP_TIMEOUT UV_INDEX_STRATEGY UV_LINK_MODE
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

export PATH="$FLAGSCALE_CONDA/envs/$FLAGSCALE_ENV_NAME/bin:$UV_PROJECT_ENVIRONMENT/bin:$FLAGSCALE_CONDA/bin:$HOME/.local/bin:$MPI_HOME/bin:$KLX_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${XRE_HOME:+$XRE_HOME/so:}$KLX_HOME/so:$MPI_HOME/lib64:$MPI_HOME/lib:/usr/local/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$FLAGSCALE_MEGATRON_PATH:$FLAGCX_PATH:${PYTHONPATH:-}"
