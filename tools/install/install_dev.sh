#!/bin/bash
# Dev phase: requirements/dev.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/utils.sh"
source "$SCRIPT_DIR/utils/pkg_utils.sh"
source "$SCRIPT_DIR/utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/dev.txt"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

main() {
    if is_phase_enabled dev; then
        # Phase enabled: install full requirements
        [ ! -f "$REQ_FILE" ] && { log_warn "dev.txt not found"; return 0; }
        set_step "Installing dev requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || die "Dev requirements failed"
        log_success "Dev requirements installed"
    else
        # Phase disabled: install only matching pip-deps
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing dev pip packages (override)"
        run_cmd -d $DEBUG pip install --root-user-action=ignore $pkgs || die "Dev pip packages failed"
        log_success "Dev pip packages installed"
    fi
}

main
