#!/bin/bash

wait_for_gpu() {
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    memory_usage_max=30000

    while true; do

    IFS=$'\n' read -d '' -r -a memory_usage_array <<< "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    IFS=$'\n' read -d '' -r -a memory_total_array <<< "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"

    need_wait=false

    for ((i=0; i<$gpu_count; i++)); do

        memory_usage_i=$((${memory_usage_array[$i]}))
        memory_total_i=$((${memory_total_array[$i]}))
        memory_remin_i=$(($memory_total_i-$memory_usage_i))

        if [ $memory_remin_i -lt $memory_usage_max ]; then
        need_wait=true
        fi

    done

    if [ "$need_wait" = false ]; then
            break
    fi

    echo "wait for gpu free"
    sleep 1m

    unset memory_usage_array
    unset memory_total_array

    done

    echo "All gpu is free"
}

# Before running, please ensure there are no other processes with a similar name to avoid closing the wrong one.
stop_all() {
    pids=$(ps aux | grep pytest | grep -v grep | awk '{print $2}')

    for pid in $pids
    do
        kill -9 $pid
    done

    pids=$(ps aux | grep python | grep -v grep | awk '{print $2}')

    for pid in $pids
    do
        kill -9 $pid
    done

    pids=$(ps aux | grep torchrun | grep -v grep | awk '{print $2}')

    for pid in $pids
    do
        kill -9 $pid
    done
}

# Function to print serve logs
print_log() {
    local log_file=$1
    echo "------------------ serve log begin -----------------------"
    if [[ -n "$log_file" && -f "$log_file" ]]; then
    echo "Log file found at $log_file. Printing log content:"
    cat "$log_file"
    else
    echo "No log file found at $log_file or path is empty."
    fi
    echo "------------------ env ----------------------"
    env
    pip list
    echo "------------------ serve log end   -----------------------"
}
