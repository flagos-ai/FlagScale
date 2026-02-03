#!/bin/bash
# Dev phase: requirements/dev.txt + dev tools (sccache)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/utils.sh"
source "$SCRIPT_DIR/utils/pkg_utils.sh"
source "$SCRIPT_DIR/utils/retry_utils.sh"

PROJECT_ROOT=$(get_project_root)
DEBUG="${FLAGSCALE_DEBUG:-false}"
RETRY_COUNT="${FLAGSCALE_RETRY_COUNT:-3}"
REQ_FILE="$PROJECT_ROOT/requirements/dev.txt"

# Source deps available for dev phase
SRC_DEPS_LIST="sccache"

# Default versions (override via environment variables)
SCCACHE_VERSION="${SCCACHE_VERSION:-0.8.1}"

while [[ $# -gt 0 ]]; do
    case $1 in --debug) DEBUG=true; shift ;; *) shift ;; esac
done

# =============================================================================
# Pip Installation
# =============================================================================
install_pip() {
    if is_phase_enabled dev; then
        [ ! -f "$REQ_FILE" ] && { log_warn "dev.txt not found"; return 0; }
        set_step "Installing dev requirements"
        retry_pip_install -d $DEBUG "$REQ_FILE" "$RETRY_COUNT" || return 1
        log_success "Dev requirements installed"
    else
        local pkgs=$(get_pip_deps_for_requirements "$REQ_FILE")
        [ -z "$pkgs" ] && return 0
        set_step "Installing dev pip packages (override)"
        run_cmd -d $DEBUG $(get_pip_cmd) install --root-user-action=ignore $pkgs || return 1
        log_success "Dev pip packages installed"
    fi
}

# =============================================================================
# Source Dependencies
# =============================================================================
install_sccache() {
    # Check if already installed
    if command -v sccache &>/dev/null; then
        local ver=$(sccache --version 2>/dev/null | head -n1 | awk '{print $2}')
        [ "$ver" = "$SCCACHE_VERSION" ] && { log_info "sccache $SCCACHE_VERSION already installed"; return 0; }
    fi

    # Check dependencies
    command -v curl &>/dev/null || { log_error "curl not found"; return 1; }
    command -v tar &>/dev/null || { log_error "tar not found"; return 1; }

    # Detect architecture
    local arch
    case "$(uname -m)" in
        x86_64)  arch="x86_64-unknown-linux-musl" ;;
        aarch64) arch="aarch64-unknown-linux-musl" ;;
        *)       log_error "Unsupported architecture: $(uname -m)"; return 1 ;;
    esac

    local url="https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${arch}.tar.gz"
    local tmp_dir="sccache-v${SCCACHE_VERSION}-${arch}"

    set_step "Installing sccache v${SCCACHE_VERSION}"

    if [ "$DEBUG" = true ]; then
        log_info "[DRY-RUN] curl -L $url | tar xz"
        return 0
    fi

    curl --connect-timeout 120 --max-time 600 --retry 5 --retry-delay 60 -L "$url" | tar xz || {
        log_error "Failed to download sccache"
        [ -d "$tmp_dir" ] && rm -rf "$tmp_dir"
        return 1
    }

    [ ! -f "$tmp_dir/sccache" ] && { log_error "sccache binary not found"; rm -rf "$tmp_dir"; return 1; }
    mv "$tmp_dir/sccache" /usr/bin/sccache
    chmod 755 /usr/bin/sccache
    rm -rf "$tmp_dir"

    # Configure for GitHub Actions
    [ -n "${GITHUB_ENV:-}" ] && {
        echo "SCCACHE_DIR=/root/.cache/sccache" >> "$GITHUB_ENV"
        echo "RUSTC_WRAPPER=$(which sccache)" >> "$GITHUB_ENV"
    }

    log_success "sccache v${SCCACHE_VERSION} installed"
}

install_src() {
    # Skip if phase disabled and no matching src-deps
    is_phase_enabled dev || has_src_deps_for_phase $SRC_DEPS_LIST || return 0

    should_install_src dev "sccache" && { install_sccache || die "sccache failed"; }
}

main() {
    install_pip || die "Dev pip failed"
    install_src
}

main
