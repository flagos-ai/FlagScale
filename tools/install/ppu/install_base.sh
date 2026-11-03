#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/ppu/base.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

if is_phase_enabled base; then
    set_step "Installing PPU base requirements"
    retry_pip_install -d "$DEBUG" "$REQ_FILE" "$RETRY_COUNT" || die "PPU base pip failed"
else
    packages=$(get_pip_deps_for_requirements "$REQ_FILE")
    [ -n "$packages" ] || exit 0
    run_cmd -d "$DEBUG" "$(get_pip_cmd)" install --root-user-action=ignore $packages \
        || die "PPU base pip override failed"
fi
log_success "PPU base requirements installed"
