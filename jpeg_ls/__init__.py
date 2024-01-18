import logging

from .CharLS import encode, decode, write, read  # noqa: F401
from _CharLS import encode_to_buffer, decode_from_buffer  # noqa: F401


__version__ = "1.1.0"


# Setup default logging
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def debug_logger() -> None:
    """Setup the logging for debugging."""
    logger = logging.getLogger(__name__)
    logger.handlers = []
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname).1s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
