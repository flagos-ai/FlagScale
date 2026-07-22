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

# Prepare the Python/runtime environment for training tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PLATFORM=""
PKG_MGR="uv"
ENV_NAME=""
ENV_PATH="/opt/venv"
INSTALL_COMMON_DEPS=false
INSTALL_CLI=false
INSTALL_PLATFORM_DEPS=true
COPY_DATA=false


while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --pkg-mgr) PKG_MGR="$2"; shift 2 ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --env-path) ENV_PATH="$2"; shift 2 ;;
        --install-common-deps) INSTALL_COMMON_DEPS=true; shift ;;
        --install-cli) INSTALL_CLI=true; shift ;;
        --skip-platform-deps) INSTALL_PLATFORM_DEPS=false; shift ;;
        --copy-data) COPY_DATA=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -n "$PLATFORM" ] || { echo "platform is required" >&2; exit 1; }

cd "$PROJECT_ROOT"
source ./tools/install/utils/pyenv_utils.sh

activate_python_env() {
    case "$PKG_MGR" in
        conda)
            if [ -n "$ENV_NAME" ] && [ -n "$ENV_PATH" ]; then
                activate_conda "$ENV_NAME" "$ENV_PATH" || {
                    echo "Conda activation failed"
                    exit 1
                }
            fi
            ;;
        uv)
            if [ -n "$ENV_PATH" ] && [ -d "$ENV_PATH" ]; then
                activate_uv_env "$ENV_PATH" || {
                    echo "UV activation failed"
                    exit 1
                }
            fi
            ;;
        pip)
            echo "Using system Python with pip"
            ;;
        *)
            echo "Unsupported package manager: $PKG_MGR" >&2
            exit 1
            ;;
    esac
}

install_common_python_deps() {
    python -m pip install coverage pytest-mock diffusers==0.36.0 transformers==4.57.6 --quiet --root-user-action=ignore
}

install_flagscale_cli() {
    python -m pip install . --no-build-isolation --root-user-action=ignore || {
        echo "FlagScale CLI install failed"
        exit 1
    }
    command -v flagscale || {
        echo "FlagScale CLI not found in PATH"
        exit 1
    }
    echo "FlagScale CLI installed successfully: $(flagscale --version 2>/dev/null || echo 'version unknown')"
}

setup_cuda_training_env() {
    local install_dir=""
    if [ "$PKG_MGR" = "conda" ] && [ -n "$ENV_PATH" ]; then
        install_dir=$(dirname "$ENV_PATH")
    fi

    local install_args=(
        --platform cuda
        --task train
        --pkg-mgr "$PKG_MGR"
        --no-system --no-dev --no-base --no-task
        --src-deps megatron-lm
        --pip-deps typer
        --force-build
        --retry-count 3
    )
    [ -n "$ENV_NAME" ] && install_args+=(--env-name "$ENV_NAME")
    [ -n "$install_dir" ] && install_args+=(--install-dir "$install_dir")

    ./tools/install/install.sh "${install_args[@]}"

    # TODO: remove after CI images contain these dependencies.
    python -m pip install \
        qwen_vl_utils==0.0.14 \
        diffusers==0.36.0 \
        websocket-client==1.8.0 \
        websocket==0.2.1 \
        websockets==15.0.1 \
        msgpack==1.1.0 \
        datasets==4.5.0
}

setup_metax_training_env() {
    local megatron_dir="/tmp/Megatron-LM-FL"
    local te_dir="/tmp/TransformerEngine-FL"

    git clone https://github.com/flagos-ai/Megatron-LM-FL.git "$megatron_dir"
    python -m pip install "$megatron_dir" --no-build-isolation --root-user-action=ignore

    git clone --depth 1 https://github.com/flagos-ai/TransformerEngine-FL.git "$te_dir"
    TE_FL_SKIP_CUDA=1 python -m pip install "$te_dir" --no-build-isolation --root-user-action=ignore

    apt-get update
    apt-get install -y curl
}

setup_ascend_training_env() {
    python -m pip install datasets==4.5.0 omegaconf==2.3.0 diffusers==0.36.0 hydra-core==1.3.2
    echo "Ascend CI image is expected to provide platform runtime dependencies"

    apt-get update
    apt-get install -y curl
}

copy_training_data() {
    mkdir -p /opt/data
    cp -r /home/gitlab-runner/data/Megatron-LM/* /opt/data/ 2>/dev/null || true
    cp -r /home/gitlab-runner/tokenizers/Megatron-LM/* /opt/data/ 2>/dev/null || true
}

echo "Preparing training test environment"
echo "Platform: $PLATFORM"
echo "Package Manager: $PKG_MGR"
echo "Environment Name: $ENV_NAME"
echo "Environment Path: $ENV_PATH"
echo "Install common deps: $INSTALL_COMMON_DEPS"
echo "Install FlagScale CLI: $INSTALL_CLI"
echo "Install platform deps: $INSTALL_PLATFORM_DEPS"

activate_python_env

echo "Python location: $(command -v python)"
echo "Python version: $(python --version)"

if [ "$INSTALL_COMMON_DEPS" = true ]; then
    install_common_python_deps
fi

if [ "$INSTALL_CLI" = true ]; then
    install_flagscale_cli
fi

if [ "$INSTALL_PLATFORM_DEPS" = true ]; then
    case "$PLATFORM" in
        cuda) setup_cuda_training_env ;;
        ascend) setup_ascend_training_env ;;
        metax) setup_metax_training_env ;;
        *) echo "No platform-specific training setup for $PLATFORM" ;;
    esac
else
    echo "Skipping platform-specific training dependencies"
fi

if [ "$COPY_DATA" = true ]; then
    copy_training_data
fi

echo "Training test environment ready"
