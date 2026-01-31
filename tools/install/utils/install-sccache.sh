#!/bin/bash
# Automate downloading, extracting, and installing sccache

# Configuration Variables - Editable
# sccache version number; check for latest releases at:
# https://github.com/mozilla/sccache/releases
SCCACHE_VERSION="0.8.1"
SCCACHE_ARCH="x86_64-unknown-linux-musl"
SCCACHE_DOWNLOAD_URL="https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}.tar.gz"
SCCACHE_INSTALL_PATH="/usr/bin/sccache"

CURL_OPTS="--connect-timeout 120 --max-time 600 --retry 5 --retry-delay 60 -L"

SCCACHE_TMP_DIR="sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}"

# Source utils for logging functions
_RETRY_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_RETRY_UTILS_DIR/utils.sh"

if ! command -v curl &> /dev/null; then
    log_error "Error: curl command not found. Please install curl first!"
    log_step "Installation command reference: Debian/Ubuntu: apt install curl -y ; CentOS/RHEL: yum install curl -y ; Alpine: apk add curl"
    exit 1
fi

if ! command -v tar &> /dev/null; then
    log_error "Error: tar command not found. Please install tar first!"
    log_step "Installation command reference: Debian/Ubuntu: apt install tar -y ; CentOS/RHEL: yum install tar -y ; Alpine: apk add tar"
    exit 1
fi

log_info "Downloading sccache v${SCCACHE_VERSION} (arch: ${SCCACHE_ARCH})..."
# Download archive via curl and pipe directly to tar for extraction
if ! curl ${CURL_OPTS} "${SCCACHE_DOWNLOAD_URL}" | tar xz; then
    log_error "Error: Failed to download or extract sccache! Please check your network or download URL."
    # Clean up partial extraction directory if it exists
    [ -d "${SCCACHE_TMP_DIR}" ] && rm -rf "${SCCACHE_TMP_DIR}"
    exit 1
fi

# Verify that the extracted binary exists
if [ ! -f "${SCCACHE_TMP_DIR}/sccache" ]; then
    log_error "Error: Extraction succeeded, but executable ${SCCACHE_TMP_DIR}/sccache not found!"
    rm -rf "${SCCACHE_TMP_DIR}"
    exit 1
fi

log_info "Installing sccache to ${SCCACHE_INSTALL_PATH}..."
mv "${SCCACHE_TMP_DIR}/sccache" "${SCCACHE_INSTALL_PATH}"

# Set standard executable permissions: rwxr-xr-x
chmod 755 "${SCCACHE_INSTALL_PATH}"

log_step "Cleaning up temporary extraction directory..."
rm -rf "${SCCACHE_TMP_DIR}"

log_step "Configuring sccache environment variables..."
# Apply environment variables if running in GitHub Actions
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SCCACHE_DIR=/root/.cache/sccache" >> "${GITHUB_ENV}"
    echo "RUSTC_WRAPPER=$(which sccache)" >> "${GITHUB_ENV}"
fi

log_step "Installation complete! sccache version: "
sccache --version | head -n1
