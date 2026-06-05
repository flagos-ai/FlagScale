import logging
import sys


class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s] %(message)s"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message, stacklevel=2)

    def warning(self, message):
        self.logger.warning(message, stacklevel=2)

    def error(self, message):
        self.logger.error(message, stacklevel=2)

    def critical(self, message):
        self.logger.critical(message, stacklevel=2)

    def debug(self, message):
        self.logger.debug(message, stacklevel=2)


GLOBAL_LOGGER = None


def get_logger():
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER is None:
        GLOBAL_LOGGER = Logger("FlagScale")
    return GLOBAL_LOGGER


logger = get_logger()
