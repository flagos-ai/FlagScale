#!/bin/bash
# check_env.sh - Print a snapshot of the hardware/software environment.
# Intended to run at the start of a CI job to aid debugging.
set -eo pipefail

# Resolve the directory containing this script so sibling files can be sourced
# regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$SCRIPT_DIR/utils.sh" ] || { echo "Error: utils.sh not found"; exit 1; }
source "$SCRIPT_DIR/utils.sh"

SEP="=========================================="

# Print a labelled section header via the shared log_info helper.
section() { log_info "\n=== $* ==="; }

echo "$SEP"
echo "       Hardware Environment Check      "
echo "$SEP"

# Detect the available accelerator management tool and dump its full status.
# Priority: NVIDIA GPU → MetaX GPU → Ascend NPU.
section "GPU / NPU Info"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
elif command -v mx-smi &>/dev/null; then
    mx-smi
elif command -v npu-smi &>/dev/null; then
    npu-smi info || true
else
    log_info "No GPU management tool found (nvidia-smi / mx-smi / npu-smi)"
fi

# Show key CPU topology fields; non-fatal if lscpu is unavailable.
section "CPU Info"
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA" || true

# Show total/used/free RAM in human-readable units.
section "Memory Info"
free -h

# /dev/shm is used for inter-process shared memory; low capacity can cause OOM
# errors in multi-process training jobs.
section "Shared Memory (/dev/shm)"
df -h /dev/shm 2>/dev/null || log_info "Not available"

# Show disk usage, excluding virtual filesystems to reduce noise.
section "Disk Info"
df -h --exclude-type=tmpfs --exclude-type=devtmpfs 2>/dev/null || df -h

# List pre-staged datasets used by functional tests.
section "/home/gitlab-runner/data"
ls -lh /home/gitlab-runner/data/ 2>/dev/null || log_info "Directory not found or empty"

# List pre-staged tokenizer files used by functional tests.
section "/home/gitlab-runner/tokenizers"
ls -lh /home/gitlab-runner/tokenizers/ 2>/dev/null || log_info "Directory not found or empty"

echo "$SEP"
log_success "Check Finished"
echo "$SEP"
