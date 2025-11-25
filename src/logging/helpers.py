"""
Output formatting helpers for nnBenchmark logging.
Provides functions for dual-output (file+console) and formatting utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru._logger import Logger


_VALID_LEVELS = {"INFO", "WARNING", "ERROR", "DEBUG"}


def log_and_print(logger: Logger, message: str, level: str = "INFO") -> None:
    """
    Write a message to both the log file and console output.

    Args:
        logger: Loguru logger instance
        message: Message to log and print
        level: Log level (INFO, WARNING, ERROR, DEBUG). Default: INFO
    """
    log_level = level.upper() if level.upper() in _VALID_LEVELS else "INFO"
    logger.log(log_level, message)
    print(message)


def log_only(logger: Logger, message: str, level: str = "INFO") -> None:
    """
    Write a message to the log file only (no console output).

    Args:
        logger: Loguru logger instance
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG). Default: INFO
    """
    log_level = level.upper() if level.upper() in _VALID_LEVELS else "INFO"
    logger.log(log_level, message)


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
