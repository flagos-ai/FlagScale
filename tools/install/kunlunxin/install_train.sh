#!/bin/bash
# Train task (Kunlunxin): requirements/kunlunxin/train.txt + pinned Megatron-LM-FL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/pkg_utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"
FLAGSCALE_DEPS="${FLAGSCALE_DEPS:-$FLAGSCALE_HOME/deps}"
FLAGSCALE_MEGATRON_PATH="${FLAGSCALE_MEGATRON_PATH:-$FLAGSCALE_DEPS/Megatron-LM-FL}"
MEGATRON_REPO="${FLAGSCALE_MEGATRON_REPO:-https://github.com/flagos-ai/Megatron-LM-FL.git}"
MEGATRON_REF="${FLAGSCALE_MEGATRON_REF:-175ae90ec92a9e6fea2d74ccd24d6a1835d3ae82}"
REQ_FILE="$PROJECT_ROOT/requirements/kunlunxin/train.txt"
SRC_DEPS_LIST="megatron-lm"
MEGATRON_INSTALLED=false

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

install_pip() {
    if is_phase_enabled task; then
        [ ! -f "$REQ_FILE" ] && { log_info "train.txt not found"; return 0; }
        set_step "Installing train requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Train requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing train pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Train pip packages installed"
    fi
}

install_megatron_lm() {
    set_step "Installing pinned Megatron-LM-FL for Kunlunxin"

    local pip_cmd
    pip_cmd=$(get_pip_cmd)
    run_cmd -d "$DEBUG" mkdir -p "$FLAGSCALE_DEPS" || return 1
    retry_git_checkout_ref -d "$DEBUG" "$MEGATRON_REPO" "$MEGATRON_REF" \
        "$FLAGSCALE_MEGATRON_PATH" "$RETRY_COUNT" || return 1

    run_cmd -d "$DEBUG" "$pip_cmd" install --root-user-action=ignore \
        "setuptools==77.0.3" "wheel==0.45.1" "pybind11==3.0.1" \
        "packaging>=24.2" || return 1
    run_cmd -d "$DEBUG" bash -c "cd '$FLAGSCALE_MEGATRON_PATH' && \
        $pip_cmd install --ignore-requires-python --no-deps --force-reinstall \
        --root-user-action=ignore --no-build-isolation . -v" || return 1
    MEGATRON_INSTALLED=true
    log_success "Pinned Megatron-LM-FL installed"
}

install_src() {
    if is_only_pip && ! has_src_deps_for_phase $SRC_DEPS_LIST; then
        log_info "Skipping source deps (only-pip mode)"
        return 0
    fi
    is_phase_enabled task || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src task "megatron-lm" && {
        install_megatron_lm || die "Megatron-LM-FL failed"
    }
}

verify_kunlunxin_training_stack() {
    [ "$DEBUG" = true ] && { log_info "Skipping runtime verification in dry-run mode"; return 0; }
    set_step "Verifying Kunlunxin training stack"

    local actual_ref
    actual_ref=$(git -C "$FLAGSCALE_MEGATRON_PATH" rev-parse HEAD) || return 1
    [ "$actual_ref" = "$MEGATRON_REF" ] || {
        log_error "Megatron-LM-FL ref mismatch: expected $MEGATRON_REF, got $actual_ref"
        return 1
    }

    python - "$FLAGSCALE_MEGATRON_PATH" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys

expected_path = Path(sys.argv[1]).resolve()
assert sys.version_info[:2] == (3, 10), sys.version

import flagcx
import megatron.core
import torch
import transformer_engine
import transformer_engine_torch
from megatron.core.jit import disable_jit_fuser
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs

module_path = Path(megatron.core.__file__).resolve()
assert module_path.is_relative_to(expected_path), (module_path, expected_path)
assert callable(disable_jit_fuser)
assert callable(get_gpt_decoder_layer_specs)
assert metadata.version("megatron-core") == "0.17.1"
assert hasattr(torch, "cuda"), "Kunlunxin torch fork must expose torch.cuda API"
print(f"Kunlunxin training stack import OK: {module_path}")
PY
}

main() {
    install_pip || die "Train pip failed"
    install_src
    if [ "$MEGATRON_INSTALLED" = true ]; then
        verify_kunlunxin_training_stack || die "Kunlunxin training stack verification failed"
    fi
}

main
