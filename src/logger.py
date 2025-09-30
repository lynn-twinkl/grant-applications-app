
import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a handler to log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger only if it has no handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
