"""
Output formatting helpers for nnBenchmark logging.
Provides functions for dual-output (file+console) and formatting utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru._logger import Logger


def log_and_print(logger: Logger, message: str, level: str = "INFO") -> None:
    """
    Write a message to both the log file and console output.

    This helper eliminates the common pattern of calling both logger.info()
    and print() for the same message. The log file gets the formatted message
    with timestamp/level, while console gets clean output.

    Args:
        logger: Loguru logger instance
        message: Message to log and print
        level: Log level (INFO, WARNING, ERROR, DEBUG). Default: INFO
    """
    # Write to log file with appropriate level
    level = level.upper()
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)
    else:
        logger.info(message)  # Default to info for unknown levels

    # Print to console (clean, no formatting)
    print(message)


def log_only(logger: Logger, message: str, level: str = "INFO") -> None:
    """
    Write a message to the log file only (no console output).

    Args:
        logger: Loguru logger instance
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG). Default: INFO
    """
    # Write to log file with appropriate level
    level = level.upper()
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)
    else:
        logger.info(message)  # Default to info for unknown levels


def log_separator(
    logger: Logger, char: str = "=", width: int = 70, print_too: bool = True
) -> None:
    """
    Write a separator line to log file and optionally to console.

    Args:
        logger: Loguru logger instance
        char: Character to use for separator (default: "=")
        width: Width of separator line (default: 70)
        print_too: If True, also print to console (default: True)
    """
    separator = char * width
    logger.info(separator)
    if print_too:
        print(separator)


def log_header(
    logger: Logger,
    message: str,
    char: str = "=",
    width: int = 70,
    print_too: bool = True,
) -> None:
    """
    Write a section header with separator lines to log file and optionally to console.

    Formats as:
        ======================================================================
        Message
        ======================================================================

    Args:
        logger: Loguru logger instance
        message: Header message
        char: Character to use for separators (default: "=")
        width: Width of separator lines (default: 70)
        print_too: If True, also print to console (default: True)
    """
    separator = char * width
    logger.info(separator)
    logger.info(message)
    logger.info(separator)

    if print_too:
        print(separator)
        print(message)
        print(separator)
