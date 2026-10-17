#!/bin/bash
set -e

# FlagScale RPM package build script
# Usage: ./packaging/rpm/build-flagscale-rpm.sh [base_image_version]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

BASE_IMAGE_VERSION="${1:-43}"

log_info()  { echo "[INFO]  $*"; }
log_step()  { echo "[STEP]  $*"; }
log_error() { echo "[ERROR] $*" >&2; }

DOCKERFILE="${SCRIPT_DIR}/dockerfiles/Dockerfile.rpm"
IMAGE_TAG="flagscale-rpm:f${BASE_IMAGE_VERSION}"
OUTPUT_DIR="${PROJECT_DIR}/rpm-packages"

log_info "Building FlagScale RPM packages"
log_info "Base image: fedora:${BASE_IMAGE_VERSION}"

log_step "Building container image: $IMAGE_TAG"
if ! docker build \
    --network=host \
    -f "$DOCKERFILE" \
    --build-arg BASE_IMAGE_VERSION="$BASE_IMAGE_VERSION" \
    -t "$IMAGE_TAG" \
    "$PROJECT_DIR"; then
    log_error "Docker build failed"
    exit 1
fi

log_step "Extracting packages to $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

CONTAINER_ID=$(docker create "$IMAGE_TAG")
docker cp "$CONTAINER_ID:/output/." "$OUTPUT_DIR/"
docker rm "$CONTAINER_ID" > /dev/null

log_info "Build complete. Packages:"
ls -lh "$OUTPUT_DIR"/*.rpm 2>/dev/null || { log_error "No .rpm files found"; exit 1; }
