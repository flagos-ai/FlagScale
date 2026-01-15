#!/bin/bash

# GPU Detection and Memory Monitoring
# Supports nvidia-smi and mx-smi based GPUs

wait_for_gpu_nvidia() {
    local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    while true; do
        mapfile -t mem_used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        mapfile -t mem_total < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
        local need_wait=false max_pct=0
        for ((i=0; i<gpu_count; i++)); do
            local pct=$(( mem_used[i] * 100 / mem_total[i] ))
            [ $pct -gt $max_pct ] && max_pct=$pct
            [ $pct -gt 50 ] && { need_wait=true; break; }
        done
        [ "$need_wait" = false ] && break
        echo "Waiting for GPU memory (current: ${max_pct}%)..."
        sleep 1m
    done
    echo "GPU ready (${max_pct}% usage)"
}

wait_for_gpu_metax() {
    command -v mx-smi &>/dev/null || { echo "Error: mx-smi not found"; exit 1; }
    while true; do
        mapfile -t mem_used < <(mx-smi --show-memory 2>/dev/null | grep -oP 'vram used\s*:\s*\K\d+')
        mapfile -t mem_total < <(mx-smi --show-memory 2>/dev/null | grep -oP 'vram total\s*:\s*\K\d+')
        [ ${#mem_used[@]} -eq 0 ] && { echo "Warning: Failed to get mx-smi data"; sleep 1m; continue; }
        local need_wait=false max_pct=0
        for ((i=0; i<${#mem_used[@]}; i++)); do
            local pct=$(( mem_used[i] * 100 / mem_total[i] ))
            [ $pct -gt $max_pct ] && max_pct=$pct
            [ $pct -gt 50 ] && { need_wait=true; break; }
        done
        [ "$need_wait" = false ] && break
        echo "Waiting for Metax GPU memory (current: ${max_pct}%)..."
        sleep 1m
    done
    echo "Metax GPU ready (${max_pct}% usage)"
}

wait_for_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        wait_for_gpu_nvidia
    elif command -v mx-smi &>/dev/null; then
        wait_for_gpu_metax
    else
        echo "Error: Neither nvidia-smi nor mx-smi found"; exit 1
    fi
}

# Process Management
stop_all() {
    for pattern in pytest python torchrun; do
        pgrep -f "$pattern" | xargs kill -9 2>/dev/null || true
    done
    echo "Terminated all test processes"
}

# Logging Functions
print_log() {
    echo "------------------ serve log begin -----------------------"
    [ -z "$1" ] || [ ! -f "$1" ] && { echo "No log file at $1"; } || { echo "Log file: $1"; cat "$1"; }
    echo "------------------ env ----------------------"
    env; pip list
    echo "------------------ serve log end   -----------------------"
}
