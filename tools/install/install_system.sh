#!/bin/bash
# =============================================================================
# FlagScale System Dependencies Installation
# =============================================================================
#
# Installs system-level dependencies: apt packages, OpenMPI, Python environment
# Supports multiple package managers: pip, uv (default), conda
#
# Usage:
#   ./install_system.sh --platform PLATFORM [OPTIONS]
#
# Examples:
#   ./install_system.sh --platform cuda                    # Basic installation (uv)
#   ./install_system.sh --platform cuda --pkg-mgr uv       # Use uv package manager
#   ./install_system.sh --platform cuda --pkg-mgr conda    # Use conda package manager
#   ./install_system.sh --platform cuda --pkg-mgr pip      # Use pip (system Python)
#   ./install_system.sh --platform cuda --no-dev           # Skip dev tools
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/utils.sh"

# =============================================================================
# Configuration
# =============================================================================
INSTALL_DEV=true    # Install dev packages by default (use --no-dev to skip)
PLATFORM="${PLATFORM:-}"  # Required: use --platform to specify
PKG_MGR="${PKG_MGR:-uv}"  # pip, uv, conda (default: uv)
DEBUG=false

# Default versions (override via environment variables)
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_VERSION="${UV_VERSION:-0.7.2}"
OPENMPI_VERSION="${OPENMPI_VERSION:-4.1.6}"

# Root installation directory (single source of truth)
FLAGSCALE_HOME="${FLAGSCALE_HOME:-/opt/flagscale}"

# Derived paths from FLAGSCALE_HOME
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$FLAGSCALE_HOME/venv}"
FLAGSCALE_CONDA="${FLAGSCALE_CONDA:-$FLAGSCALE_HOME/miniconda3}"
FLAGSCALE_DOWNLOADS="${FLAGSCALE_DOWNLOADS:-$FLAGSCALE_HOME/downloads}"

# =============================================================================
# Package Lists
# =============================================================================
# Core system packages (common across deepspeed, vllm, sglang, megatron)
# Note: Python is installed via install_python (supports uv, conda, pip)
BASE_PACKAGES="
    software-properties-common ca-certificates curl wget sudo
    git git-lfs unzip tzdata locales gettext
    build-essential cmake ninja-build perl pkg-config file gfortran libopenblas-dev
    openssh-client openssh-server
    rsync lsof kmod netcat-openbsd psmisc uuid-runtime
    net-tools iputils-ping
"

# InfiniBand/RDMA packages (common for distributed training)
RDMA_PACKAGES="
    libibverbs-dev libibverbs1 librdmacm1 rdma-core
    ibverbs-providers infiniband-diags perftest
    libnuma-dev libnuma1 numactl
"

# Libraries for ML frameworks (image, audio, async IO)
# Note: Platform-specific packages (e.g., libcupti-dev) should be in platform install scripts
ML_PACKAGES="
    ffmpeg libsm6 libxext6 libgl1
    libsndfile-dev libjpeg-dev libpng-dev
    libaio-dev libssl-dev libcurl4-openssl-dev
    ccache patchelf
"

DEV_PACKAGES="vim tmux screen htop iftop iotop gdb less tree"

# =============================================================================
# Installation Functions
# =============================================================================

# Configure timezone non-interactively to avoid tzdata prompts
configure_timezone() {
    local tz="${TZ:-Asia/Shanghai}"
    set_step "Configuring timezone ($tz)"

    # Set environment variables for non-interactive installation
    export DEBIAN_FRONTEND=noninteractive
    export TZ="$tz"

    if [ "$DEBUG" = true ]; then
        echo "    [dry-run] ln -sf /usr/share/zoneinfo/$tz /etc/localtime" >&2
        echo "    [dry-run] echo $tz > /etc/timezone" >&2
        return 0
    fi

    # Pre-configure timezone before apt install
    ln -sf "/usr/share/zoneinfo/$tz" /etc/localtime 2>/dev/null || true
    echo "$tz" > /etc/timezone 2>/dev/null || true
    log_success "Timezone configured"
}

install_apt_packages() {
    set_step "Installing apt packages"

    local packages="$BASE_PACKAGES $RDMA_PACKAGES $ML_PACKAGES"
    [ "$INSTALL_DEV" = true ] && packages="$packages $DEV_PACKAGES"
    run_cmd -d $DEBUG -m "Updating package lists..." apt-get update
    # shellcheck disable=SC2086
    run_cmd -d $DEBUG -m "Installing packages..." apt-get install -y --no-install-recommends $packages
    # run_cmd -d $DEBUG -m "Cleaning up..." apt-get clean
    # run_cmd -d $DEBUG rm -rf /var/lib/apt/lists/*
    log_success "Apt packages done"
}

install_python_uv() {
    set_step "Installing Python ${PYTHON_VERSION} (uv ${UV_VERSION})"

    run_cmd -d $DEBUG -m "Installing uv ${UV_VERSION}..." \
        bash -c "curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh"
    run_cmd -d $DEBUG -m "Creating venv at ${UV_PROJECT_ENVIRONMENT}..." \
        "$HOME/.local/bin/uv" venv "${UV_PROJECT_ENVIRONMENT}" --python "${PYTHON_VERSION}"
    run_cmd -d $DEBUG -m "Symlink python3..." ln -sf "${UV_PROJECT_ENVIRONMENT}/bin/python3" /usr/bin/python3
    run_cmd -d $DEBUG ln -sf "${UV_PROJECT_ENVIRONMENT}/bin/python3-config" /usr/bin/python3-config
    run_cmd -d $DEBUG ln -sf "${UV_PROJECT_ENVIRONMENT}/bin/pip" /usr/bin/pip
    run_cmd -d $DEBUG ln -sf /usr/bin/python3 /usr/bin/python
    log_success "Python ready: ${UV_PROJECT_ENVIRONMENT}"
}

install_python_conda() {
    set_step "Installing Python ${PYTHON_VERSION} (conda)"

    local env_name="${FLAGSCALE_ENV_NAME:-}"

    # Skip if conda already installed at FLAGSCALE_CONDA
    if [ -f "${FLAGSCALE_CONDA}/bin/conda" ]; then
        log_info "Conda already installed at ${FLAGSCALE_CONDA}"
    else
        # Create download directory
        mkdir -p "$FLAGSCALE_DOWNLOADS"
        local conda_installer="$FLAGSCALE_DOWNLOADS/miniconda.sh"
        # Download miniconda if not present (cached for future use)
        if [ ! -f "$conda_installer" ]; then
            run_cmd -d $DEBUG -m "Downloading Miniconda to $FLAGSCALE_DOWNLOADS..." \
                wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "$conda_installer"
        else
            log_info "Using cached miniconda installer"
        fi
        # Silent install with TOS acceptance
        run_cmd -d $DEBUG -m "Installing Miniconda to ${FLAGSCALE_CONDA}..." \
            env ANACONDA_ACCEPT_TOS=yes bash "$conda_installer" -b -u -p "${FLAGSCALE_CONDA}"
    fi

    log_info "Configuring conda..."
    run_cmd -d $DEBUG env CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "${FLAGSCALE_CONDA}/bin/conda" init bash
    run_cmd -d $DEBUG env CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "${FLAGSCALE_CONDA}/bin/conda" config --set auto_activate_base false
    run_cmd -d $DEBUG env CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "${FLAGSCALE_CONDA}/bin/conda" config --set channel_priority flexible
    run_cmd -d $DEBUG env CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "${FLAGSCALE_CONDA}/bin/conda" config --set solver classic

    # Create named environment if specified, otherwise install to base
    if [ -n "$env_name" ]; then
        run_cmd -d $DEBUG -m "Creating conda env: $env_name (python=${PYTHON_VERSION})..." \
            env CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "${FLAGSCALE_CONDA}/bin/conda" create -y -n "$env_name" "python=${PYTHON_VERSION}"
        log_info "Setting up symlinks to $env_name env..."
        run_cmd -d $DEBUG ln -sf "${FLAGSCALE_CONDA}/envs/${env_name}/bin/python3" /usr/bin/python3
        run_cmd -d $DEBUG ln -sf "${FLAGSCALE_CONDA}/envs/${env_name}/bin/python3-config" /usr/bin/python3-config
        run_cmd -d $DEBUG ln -sf "${FLAGSCALE_CONDA}/envs/${env_name}/bin/pip" /usr/bin/pip
        log_success "Conda env '$env_name' ready: ${FLAGSCALE_CONDA}/envs/${env_name}"
    else
        run_cmd -d $DEBUG -m "Installing Python ${PYTHON_VERSION} to base..." \
            env CONDA_NO_PLUGINS=true ANACONDA_ACCEPT_TOS=yes "${FLAGSCALE_CONDA}/bin/conda" install -y python="${PYTHON_VERSION}"
        log_info "Setting up symlinks to base..."
        run_cmd -d $DEBUG ln -sf "${FLAGSCALE_CONDA}/bin/python3" /usr/bin/python3
        run_cmd -d $DEBUG ln -sf "${FLAGSCALE_CONDA}/bin/python3-config" /usr/bin/python3-config
        run_cmd -d $DEBUG ln -sf "${FLAGSCALE_CONDA}/bin/pip" /usr/bin/pip
        log_success "Conda base ready: ${FLAGSCALE_CONDA}"
    fi
    run_cmd -d $DEBUG ln -sf /usr/bin/python3 /usr/bin/python
}

install_python_pip() {
    set_step "Installing Python ${PYTHON_VERSION} (system pip)"

    run_cmd -d $DEBUG -m "Adding deadsnakes PPA..." add-apt-repository -y ppa:deadsnakes/ppa
    run_cmd -d $DEBUG -m "Updating package lists..." apt-get update
    run_cmd -d $DEBUG -m "Installing Python ${PYTHON_VERSION}..." apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-venv" python3-pip
    log_info "Configuring alternatives..."
    run_cmd -d $DEBUG update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1
    run_cmd -d $DEBUG update-alternatives --set python3 "/usr/bin/python${PYTHON_VERSION}"
    run_cmd -d $DEBUG ln -sf "/usr/bin/python${PYTHON_VERSION}-config" /usr/bin/python3-config
    run_cmd -d $DEBUG ln -sf /usr/bin/python3 /usr/bin/python
    run_cmd -d $DEBUG -m "Upgrading pip..." python3 -m pip install --root-user-action=ignore --upgrade pip
    log_success "System Python ready"
}

install_python() {
    case "$PKG_MGR" in
        uv)    install_python_uv ;;
        conda) install_python_conda ;;
        pip)   install_python_pip ;;
        *)     log_error "Unknown pkg manager: $PKG_MGR"; exit 1 ;;
    esac
}

install_openmpi() {
    set_step "Installing OpenMPI ${OPENMPI_VERSION}"

    local version="$OPENMPI_VERSION"
    local base_version="${version%.*}"
    local prefix="/usr/local/openmpi-${version}"
    local tarball_url="https://download.open-mpi.org/release/open-mpi/v${base_version}/openmpi-${version}.tar.gz"

    # Download tarball to FLAGSCALE_DOWNLOADS (cached for future use)
    mkdir -p "$FLAGSCALE_DOWNLOADS"
    local tarball="$FLAGSCALE_DOWNLOADS/openmpi-${version}.tar.gz"
    if [ ! -f "$tarball" ]; then
        run_cmd -d $DEBUG -m "Downloading OpenMPI ${version} to $FLAGSCALE_DOWNLOADS..." \
            wget -q "$tarball_url" -O "$tarball"
    else
        log_info "Using cached OpenMPI tarball"
    fi

    # Extract and build
    run_cmd -d $DEBUG -m "Extracting OpenMPI..." \
        bash -c "cd /tmp && tar xzf '$tarball'"
    run_cmd -d $DEBUG -m "Configuring OpenMPI..." \
        bash -c "cd /tmp/openmpi-${version} && ./configure --prefix=${prefix} --quiet"
    run_cmd -d $DEBUG -m "Building OpenMPI (may take a while)..." \
        bash -c "cd /tmp/openmpi-${version} && make -j\$(nproc) install"
    run_cmd -d $DEBUG -m "Creating symlink..." ln -sf "${prefix}" /usr/local/mpi
    run_cmd -d $DEBUG -m "Setting up mpirun wrapper..." \
        bash -c "mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && cat > /usr/local/mpi/bin/mpirun << 'WRAPPER'
#!/bin/bash
exec mpirun.real --allow-run-as-root --prefix /usr/local/mpi \"\\\$@\"
WRAPPER
chmod +x /usr/local/mpi/bin/mpirun"
    run_cmd -d $DEBUG rm -rf /tmp/openmpi-${version}
    log_success "OpenMPI done"
}

install_env_scripts() {
    set_step "Installing environment scripts"

    local profile_dir="/etc/profile.d"
    local env_script="$SCRIPT_DIR/$PLATFORM/env.sh"

    if [ -z "$PLATFORM" ]; then
        log_error "PLATFORM not set. Use --platform to specify (e.g., --platform cuda)"
        exit 1
    fi

    if [ ! -f "$env_script" ]; then
        log_error "Environment script not found: $env_script"
        exit 1
    fi

    run_cmd -d $DEBUG -m "Installing ${PLATFORM} env..." cp "$env_script" "$profile_dir/flagscale-env.sh"

    run_cmd -d $DEBUG -m "Configuring bash.bashrc..." \
        bash -c 'grep -q "flagscale-env.sh" /etc/bash.bashrc 2>/dev/null || cat >> /etc/bash.bashrc << "BASHRC"

# FlagScale environment
[ -f /etc/profile.d/flagscale-env.sh ] && . /etc/profile.d/flagscale-env.sh
BASHRC'
    log_success "Env scripts done"
}

# =============================================================================
# Main
# =============================================================================
usage() {
    cat << EOF
Usage: $0 --platform PLATFORM [OPTIONS]

Options:
    --platform NAME    Platform for env scripts (required, e.g., cuda)
    --no-dev           Skip development tools (vim, tmux, htop, etc.)
    --pkg-mgr MGR      Package manager: pip, uv, conda (default: uv)
    --debug            Debug mode: print commands without executing (dry-run)
    --help             Show this help

Package Managers:
    uv     - Fast, modern package manager with venv (default)
    conda  - Miniconda installation
    pip    - System Python with pip

Versions (override via environment variables):
    PYTHON_VERSION     Python version (default: ${PYTHON_VERSION})
    UV_VERSION         uv version (default: ${UV_VERSION})
    OPENMPI_VERSION    OpenMPI version (default: ${OPENMPI_VERSION})

Environment paths (derived from FLAGSCALE_HOME, override via environment variables):
    FLAGSCALE_HOME          Root installation directory (default: /opt/flagscale)
    UV_PROJECT_ENVIRONMENT  uv venv path (default: \$FLAGSCALE_HOME/venv)
    FLAGSCALE_CONDA         Miniconda path (default: \$FLAGSCALE_HOME/miniconda3)
    FLAGSCALE_DOWNLOADS     Downloads directory (default: \$FLAGSCALE_HOME/downloads)
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-dev)   INSTALL_DEV=false; shift ;;
            --platform) PLATFORM="$2"; shift 2 ;;
            --pkg-mgr)  PKG_MGR="$2"; shift 2 ;;
            --debug)    DEBUG=true; shift ;;
            --help|-h)  usage; exit 0 ;;
            *)          log_error "Unknown option: $1"; exit 1 ;;
        esac
    done
}

main() {
    parse_args "$@"

    # Validate required parameters
    [ -z "$PLATFORM" ] && { log_error "Platform required (use --platform, e.g., --platform cuda)"; usage; exit 1; }

    [ "$DEBUG" = true ] && log_info "Dry-run mode: commands printed, not executed"

    log_info "Python ${PYTHON_VERSION} | ${PKG_MGR} | OpenMPI ${OPENMPI_VERSION}"

    configure_timezone || die "Timezone configuration failed"
    install_apt_packages || die "Apt packages installation failed"
    install_python || die "Python installation failed"
    install_openmpi || die "OpenMPI installation failed"
    install_env_scripts || die "Environment scripts installation failed"

    log_success "System setup complete"
}

main "$@"
