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

# Main function to detect GPU tool and call appropriate wait function
# Future: Additional chip types can be added here by extending the detection logic
# and implementing corresponding wait functions (e.g., wait_for_gpu_amd, wait_for_gpu_intel, etc.)
wait_for_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Detected nvidia-smi, using NVIDIA GPU monitoring"
        wait_for_gpu_nvidia
    else
        echo "Error: 'nvidia-smi' command not found or unavailable"
        echo "Note: If you are using a new chip type, please add GPU idle detection method for your chip"
        exit 1
    fi
}
