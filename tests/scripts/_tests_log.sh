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
