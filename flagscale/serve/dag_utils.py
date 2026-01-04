import socket

from flagscale.logger import logger


def check_and_get_port(target_port=None, host="0.0.0.0"):
    """
    Check if a specific port is free; if not, allocate a free port.
    :param target_port: The port number to check, default is None.
    :param host: The host address to check, default is "0.0.0.0".
    :return: the allocated port (target_port if free, or a new free port).
    """
    if target_port is None:
        # The same as Ray
        port = 6379
    else:
        port = target_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Try binding the target port
            s.bind((host, port))
            logger.info(f"Port {port} is free and can be used.")
            return port
        except OSError:
            # Target port is occupied, get a free port
            s.bind((host, 0))
            free_port = s.getsockname()[1]
            if target_port is None:
                logger.info(f"Port {port} is occupied. Allocated free port: {free_port}")
            else:
                logger.warning(f"Port {port} is occupied. Allocated free port: {free_port}")
            return free_port
