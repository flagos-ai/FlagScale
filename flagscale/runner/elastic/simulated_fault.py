import argparse
import random
import time
from datetime import datetime

error_keys_list = [
    # Success indicators
    "completed",
    # Memory errors
    "out of memory",
    "outofmemoryerror",
    "cuda out of memory",
    # Connection and network errors
    "rendezvousconnectionerror",
    "connection refused",
    "connection timeout",
    # Import and code errors
    "importerror",
    "modulenotfounderror",
    "traceback (most recent call last)",
    "fatal error",
    "exception",
    # Process errors
    "process killed",
    "killed by signal",
    "terminated by signal",
    "keyboardinterrupt",
    "sigint",
    "sigterm",
    "segmentation fault",
    "core dumped",
    # CUDA errors
    "cuda error",
    "cudnn error",
    "gpu error",
    # File and storage errors
    "no such file or directory",
    "permission denied",
    "no space left on device",
    # Timeout errors
    "operation timed out",
    "connection timeout",
    "hanging",
]


def simulated_fault_loop(log_file="", error_keys=None, interval=5, iterations=1, mode="a"):
    """
    Simulate faults by writing only the error key (no description) to a log file.
    """
    if error_keys is not None and not isinstance(error_keys, list):
        raise ValueError("error_keys must be a list of strings or None")

    for i in range(iterations):
        with open(log_file, mode) as f:
            f.write(f"--- Simulated log at {datetime.now()} (iteration {i + 1}) ---\n")

            # choose errors to write
            if error_keys:
                current_errors = error_keys
            else:
                current_errors = [random.choice(error_keys_list)]

            for key in current_errors:
                f.write(f"{key}\n")
            f.write("\n")

        time.sleep(interval)


def main():
    """
    Entry point for the fault simulation script.

    This function sets up an argument parser to handle command-line arguments and
    invokes the simulated_fault_loop function to simulate faults by writing errors
    to a specified log file. The arguments control the log file path, error types,
    interval between writes, number of iterations, and file mode (overwrite or append).
    """
    parser = argparse.ArgumentParser(description="Simulate faults by writing errors to log.")
    parser.add_argument("--log_file", type=str, default="output.log", help="Path to log file")
    parser.add_argument(
        "--errors",
        type=str,
        nargs="*",
        default=None,
        help="List of error keys to simulate, e.g. workeroom codeerror",
    )
    parser.add_argument("--interval", type=int, default=5, help="Seconds between writes")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument(
        "--mode", type=str, choices=["w", "a"], default="a", help="File mode: w=overwrite, a=append"
    )

    args = parser.parse_args()

    simulated_fault_loop(
        log_file=args.log_file,
        error_keys=args.errors,
        interval=args.interval,
        iterations=args.iterations,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
