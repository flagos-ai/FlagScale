#!/bin/bash
# Shared Utility Functions for FlagScale Test Runners

# Logging
log_info() {
    echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_error() {
    echo -e "\033[0;31m[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $*\033[0m" >&2
}

log_success() {
    echo -e "\033[0;32m[SUCCESS] $(date +'%Y-%m-%d %H:%M:%S') - $*\033[0m" >&2
}

# Validation
validate_platform() {
    local platform="$1"
    local script_dir="$2"
    local config_file="${script_dir}/../config/platforms/${platform}.yaml"

    # Reject template as a platform
    if [ "$platform" = "template" ]; then
        log_error "Cannot use 'template' as a platform"
        log_error "template.yaml is a template file for creating new platform configurations"
        log_error "Available platforms:"
        ls -1 "${script_dir}/../config/platforms/" 2>/dev/null | grep -v '^template\.yaml$' | sed 's/.yaml$//' | sed 's/^/  - /' >&2
        return 1
    fi

    if [ ! -f "$config_file" ]; then
        log_error "Platform configuration not found: $config_file"
        log_info "Available platforms:"
        ls -1 "${script_dir}/../config/platforms/" 2>/dev/null | grep -v '^template\.yaml$' | sed 's/.yaml$//' | sed 's/^/  - /' >&2
        log_info ""
        log_info "To create a new platform configuration:"
        log_info "  1. Copy tests/test_utils/config/platforms/template.yaml to ${platform}.yaml"
        log_info "  2. Customize the configuration for your platform"
        return 1
    fi
    return 0
}

validate_device() {
    local platform="$1"
    local device="$2"
    local script_dir="$3"

    local available_devices=$(python "${script_dir}/parse_config.py" --platform "$platform" --type device_types 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$available_devices" ]; then
        log_error "Failed to query device types for platform '$platform'"
        return 1
    fi

    if ! echo "$available_devices" | grep -q "\"$device\""; then
        log_error "Device '$device' not found in platform '$platform'"
        log_info "Available devices: $available_devices"
        return 1
    fi

    return 0
}

get_device_types() {
    local platform="$1"
    local script_dir="$2"

    python "${script_dir}/parse_config.py" --platform "$platform" --type device_types 2>/dev/null
}

# GPU Management

# Fetch mem_used[] and mem_total[] arrays for nvidia
_gpu_fetch_nvidia() {
    mapfile -t mem_used  < <(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null)
    mapfile -t mem_total < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
}

# Fetch mem_used[] and mem_total[] arrays for metax
_gpu_fetch_metax() {
    local mem_output
    mem_output=$(mx-smi --show-memory 2>/dev/null)
    mapfile -t mem_used  < <(echo "$mem_output" | awk '$1=="vram" && $2=="used"  {print $(NF-1)}')
    mapfile -t mem_total < <(echo "$mem_output" | awk '$1=="vram" && $2=="total" {print $(NF-1)}')
}

# Fetch mem_used[] and mem_total[] arrays for ascend
_gpu_fetch_ascend() {
    local info pci_awk
    info=$(npu-smi info 2>/dev/null)
    # Match Chip rows by PCI bus ID; find last "X / Y" pattern (HBM column)
    pci_awk='/[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+\.[0-9]/ {
        s=$0
        while (match(s, /[0-9]+ *\/ *[0-9]+/)) { last=substr(s,RSTART,RLENGTH); s=substr(s,RSTART+RLENGTH) }
        split(last, a, / *\/ */)
        print a[idx]+0
    }'
    mapfile -t mem_used  < <(echo "$info" | awk -v idx=1 "$pci_awk")
    mapfile -t mem_total < <(echo "$info" | awk -v idx=2 "$pci_awk")
}

# Common polling loop; args: <gpu_count> <fetch_fn>
_gpu_poll_loop() {
    local gpu_count=$1 fetch_fn=$2

    while true; do
        "$fetch_fn"
        local need_wait=false max_pct=0
        for ((i=0; i<gpu_count; i++)); do
            [ -z "${mem_total[i]}" ] || [ "${mem_total[i]}" -eq 0 ] && continue
            local pct=$(( mem_used[i] * 100 / mem_total[i] ))
            [ $pct -gt $max_pct ] && max_pct=$pct
            [ $pct -gt 50 ] && { need_wait=true; break; }
        done
        [ "$need_wait" = false ] && break
        echo "Waiting for GPU memory (current: ${max_pct}%)..."
        sleep 60
    done
    echo "GPU ready (${max_pct}% usage)"
}

wait_for_gpu() {
    local gpu_count fetch_fn
    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        fetch_fn=_gpu_fetch_nvidia
    elif command -v mx-smi &>/dev/null; then
        gpu_count=$(mx-smi --show-hwinfo 2>/dev/null | awk '/Attached GPUs/{print $NF}')
        fetch_fn=_gpu_fetch_metax
    elif command -v npu-smi &>/dev/null; then
        gpu_count=$(npu-smi info -l 2>/dev/null | awk '/Total Count/{print $NF}')
        fetch_fn=_gpu_fetch_ascend
    else
        return 0
    fi
    [ -z "$gpu_count" ] || [ "$gpu_count" -eq 0 ] && return 0
    _gpu_poll_loop "$gpu_count" "$fetch_fn"
}
