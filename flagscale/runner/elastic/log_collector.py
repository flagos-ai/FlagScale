import glob
import os
import shlex

from flagscale.runner.utils import get_remote_file_size, logger, run_local_command

_log_offsets = {}


def get_file_size(host, filepath, port=22):
    """
    Retrieve the size of a file, either locally or on a remote host (in bytes).

    Parameters:
        host (str): The address of the host (e.g., 'localhost' or 'user@hostname').
        filepath (str): The path to the file, either local or on the remote host.
        port (int): SSH port number for remote hosts (default: 22).

    Returns:
        int: The size of the file in bytes if successful; returns -1 if the file does not exist
             or an error occurs during remote access.

    Notes:
        - For local files ('localhost'), uses os.path to check existence and get size.
    """
    if host == "localhost":
        if os.path.exists(filepath):
            return os.path.getsize(filepath)
        return -1
    else:
        return get_remote_file_size(host, filepath, port)


def find_actual_log_file(log_dir, node_rank, host, no_shared_fs=False):
    """
    Smart file discovery for log files that handles hostname/IP mismatches.

    Args:
        log_dir (str): Directory containing log files
        node_rank (int): Rank of the node
        host (str): Expected hostname or IP
        no_shared_fs (bool): Whether shared filesystem is disabled

    Returns:
        str: Path to the actual log file found, or expected path if not found
    """
    # Construct expected filename based on original logic
    if no_shared_fs:
        expected_file = os.path.join(log_dir, "host.output")
    else:
        expected_file = os.path.join(log_dir, f"host_{node_rank}_{host}.output")

    # Try exact match first
    if os.path.exists(expected_file):
        return expected_file

    # Use glob pattern to find any matching node_rank file
    if not no_shared_fs:
        pattern = os.path.join(log_dir, f"host_{node_rank}_*.output")
        matches = glob.glob(pattern)

        if matches:
            # Return the first match found
            logger.debug(
                f"Smart discovery found log file: {matches[0]} (expected: {expected_file})"
            )
            return matches[0]

    # Return original expected path for error handling
    return expected_file


def collect_logs(config, host, node_rank, destination_dir, dryrun=False):
    """
    Collect logs incrementally from a specified host and node rank, saving to destination_dir.
    Args:
        config (DictConfig): Configuration object containing experiment and logging details.
        host (str): Hostname or IP of the node.
        node_rank (int): Rank of the node.
        destination_dir (str): Directory to store collected logs.
        dryrun (bool): If True, simulate the collection without executing commands.
    Returns:
        str: Path to the collected log file.
    """
    logging_config = config.train.system.logging
    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    log_dir = logging_config.log_dir
    src_log_file = find_actual_log_file(log_dir, node_rank, host, no_shared_fs)
    dest_log_file = os.path.join(destination_dir, f"host_{node_rank}_{host}_current.log")

    os.makedirs(destination_dir, exist_ok=True)

    log_key = f"{host}_{node_rank}"
    offset = _log_offsets.get(log_key, 0)

    ssh_port = config.experiment.runner.get("ssh_port", 22)

    try:
        if host != "localhost":
            command = f"ssh -p {ssh_port} {host} 'tail -c +{offset + 1} {shlex.quote(src_log_file)}' >> {shlex.quote(dest_log_file)}"
            run_local_command(command, dryrun)
            logger.debug(
                f"Collected incremental log from {host} (node {node_rank}) to {dest_log_file}"
            )
        else:
            command = (
                f"tail -c +{offset + 1} {shlex.quote(src_log_file)} >> {shlex.quote(dest_log_file)}"
            )
            run_local_command(command, dryrun)
            logger.debug(f"Collected incremental local log to {dest_log_file}")

        # Check if the source file exists and update the offset
        if os.path.exists(src_log_file):
            current_src_size = get_file_size(host, src_log_file, ssh_port)
            if current_src_size > 0:
                if current_src_size > offset:  # There is new content in the source file
                    _log_offsets[log_key] = current_src_size

                    if os.path.exists(dest_log_file) and os.path.getsize(dest_log_file) > 0:
                        logger.debug(
                            f"Collected {current_src_size - offset} bytes from {src_log_file}"
                        )
                        return dest_log_file
                    else:
                        logger.debug(f"No new content extracted from {src_log_file}")
                        if os.path.exists(dest_log_file):
                            os.remove(dest_log_file)
                        return None
            else:
                logger.debug(f"No new content in source file {src_log_file}")
                if os.path.exists(dest_log_file):
                    os.remove(dest_log_file)
                return None
        else:
            logger.debug(f"Source log file {src_log_file} not found")
            if os.path.exists(dest_log_file):
                os.remove(dest_log_file)
            return None

    except Exception as e:
        logger.error(f"Failed to collect logs from {host} (node {node_rank}): {e}")
        if os.path.exists(dest_log_file):
            os.remove(dest_log_file)
        return None
