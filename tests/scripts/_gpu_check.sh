#!/bin/bash

# Function to wait for GPU availability using nvidia-smi
# This version uses integer arithmetic instead of bc for better compatibility
wait_for_gpu_nvidia() {
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    while true; do
        local memory_usage_array=()
        local memory_total_array=()
        # Query GPU memory usage and total memory, suppress stderr to prevent exit on failure
        mapfile -t memory_usage_array < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        mapfile -t memory_total_array < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)

        local need_wait=false
        local max_usage_percent=0

        # Iterate through each GPU to calculate memory usage percentage
        for ((i=0; i<${#memory_usage_array[@]}; i++)); do
            # Remove whitespace from nvidia-smi output
            local memory_usage_i=${memory_usage_array[$i]// /}
            local memory_total_i=${memory_total_array[$i]// /}

            # Validate that memory values are numeric and total memory is greater than 0
            if [[ $memory_usage_i =~ ^[0-9]+$ ]] && [[ $memory_total_i =~ ^[0-9]+$ ]] && [ "$memory_total_i" -gt 0 ]; then
                # Calculate percentage using integer arithmetic (multiply by 100 first to avoid precision loss)
                local usage_percent=$((memory_usage_i * 100 / memory_total_i))
                # Track the maximum usage percentage across all GPUs
                if [ $usage_percent -gt $max_usage_percent ]; then
                    max_usage_percent=$usage_percent
                fi
            else
                # Log warning for invalid values and continue waiting
                echo "Warning: Invalid memory values - usage: '$memory_usage_i', total: '$memory_total_i'"
                need_wait=true
                break
            fi
        done

        # If max usage percentage does not exceed 50%, we can proceed
        # 50% threshold = 50 (since we're using integer percentages)
        if [ "$need_wait" = false ] && [ $max_usage_percent -le 50 ]; then
            break
        fi

        # Wait and show current status
        echo "Waiting for GPU memory usage to drop below 50% (current max usage: ${max_usage_percent}%)"
        sleep 1m
    done

    echo "All GPUs have sufficient free memory, GPU memory usage ratio is below 50% (current max usage: ${max_usage_percent}%)"
}

# Function to wait for GPU availability using mx-smi
# This function monitors Metax GPU memory usage and waits until it's below 50%
wait_for_gpu_metax() {
    # Check if mx-smi is available
    if ! command -v mx-smi &> /dev/null; then
        echo "Error: mx-smi not found"
        exit 1
    fi

    while true; do
        local memory_usage_array=()
        local memory_total_array=()

        # Query GPU memory usage and total memory
        # mx-smi --show-memory displays memory info in KBytes
        # Extracting vram used and total values for each GPU
        mapfile -t memory_usage_array < <(mx-smi --show-memory 2>/dev/null | grep -oP 'vram used\s*:\s*\K\d+' | tr -d ' ')
        mapfile -t memory_total_array < <(mx-smi --show-memory 2>/dev/null | grep -oP 'vram total\s*:\s*\K\d+' | tr -d ' ')

        # Check if we got any data
        if [ ${#memory_usage_array[@]} -eq 0 ] || [ ${#memory_total_array[@]} -eq 0 ]; then
            echo "Warning: Failed to retrieve memory information from mx-smi"
            sleep 1m
            continue
        fi

        local need_wait=false
        local max_usage_percent=0

        # Iterate through each GPU to calculate memory usage percentage
        for ((i=0; i<${#memory_usage_array[@]}; i++)); do
            local memory_usage_i=${memory_usage_array[$i]}
            local memory_total_i=${memory_total_array[$i]}

            # Validate that memory values are numeric and total memory is greater than 0
            if [[ $memory_usage_i =~ ^[0-9]+$ ]] && [[ $memory_total_i =~ ^[0-9]+$ ]] && [ "$memory_total_i" -gt 0 ]; then
                # Calculate percentage using integer arithmetic
                local usage_percent=$((memory_usage_i * 100 / memory_total_i))
                # Track the maximum usage percentage across all GPUs
                if [ $usage_percent -gt $max_usage_percent ]; then
                    max_usage_percent=$usage_percent
                fi
            else
                # Log warning for invalid values and continue waiting
                echo "Warning: Invalid memory values - usage: '$memory_usage_i', total: '$memory_total_i'"
                need_wait=true
                break
            fi
        done

        # If max usage percentage does not exceed 50%, we can proceed
        if [ "$need_wait" = false ] && [ $max_usage_percent -le 50 ]; then
            break
        fi

        # Wait and show current status
        echo "Waiting for Metax GPU memory usage to drop below 50% (current max usage: ${max_usage_percent}%)"
        sleep 1m
    done

    echo "All Metax GPUs have sufficient free memory, GPU memory usage ratio is below 50% (current max usage: ${max_usage_percent}%)"
}

# Main function to detect GPU tool and call appropriate wait function
# Future: Additional chip types can be added here by extending the detection logic
# and implementing corresponding wait functions (e.g., wait_for_gpu_amd, wait_for_gpu_intel, etc.)
wait_for_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Detected nvidia-smi, using NVIDIA GPU monitoring"
        wait_for_gpu_nvidia
    elif command -v mx-smi &> /dev/null; then
        echo "Detected mx-smi, using Metax GPU monitoring"
        wait_for_gpu_metax
    else
        echo "Error: Neither nvidia-smi nor mx-smi is available"
        echo "Note: If you are using a new chip type, please add GPU idle detection method for your chip"
        exit 1
    fi
}
