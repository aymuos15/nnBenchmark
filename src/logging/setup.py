"""
Logger setup utilities for nnBenchmark.
Provides functions to configure loguru loggers for different use cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logger(results_dir: str, log_name: str = "train") -> Any:
    """
    Setup loguru logger that writes to a log file only.
    Always appends to existing log file if it exists.

    Args:
        results_dir: Directory where log file will be saved
        log_name: Name of the log file (without .log extension)

    Returns:
        Configured logger instance
    """
    # Remove default handler (console output)
    logger.remove()

    # Add file handler for logs (always append)
    log_path = str(Path(results_dir) / f"{log_name}.log")

    _ = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation=None,  # No rotation
        retention=None,  # Keep all logs
        enqueue=True,  # Thread-safe logging
    )

    return logger


def setup_train_logger(results_dir: str) -> Any:
    """
    Setup loguru logger for training that writes to train.log file only.
    Always appends to existing log file.

    Args:
        results_dir: Directory where train.log will be saved

    Returns:
        Configured logger instance
    """
    return setup_logger(results_dir, "train")


def setup_test_logger(results_dir: str) -> Any:
    """
    Setup loguru logger for testing that writes to test.log file only.

    Args:
        results_dir: Directory where test.log will be saved

    Returns:
        Configured logger instance
    """
    return setup_logger(results_dir, "test")


def setup_val_logger(results_dir: str) -> Any:
    """
    Setup loguru logger for validation that writes to val.log file only.

    Args:
        results_dir: Directory where val.log will be saved

    Returns:
        Configured logger instance
    """
    return setup_logger(results_dir, "val")


def setup_verbose_logger(
    level: str = "DEBUG", format_string: str | None = None
) -> None:
    """
    Configure loguru logger for verbose console output.

    Useful for scripts that want detailed console logging (e.g., plan.py with --verbose).
    Removes default handler and adds a new one with specified verbosity level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Default: DEBUG
        format_string: Optional custom format string. If None, uses default loguru format.
    """
    logger.remove()  # Remove default handler
    if format_string is not None:
        _ = logger.add(sys.stderr, level=level.upper(), format=format_string)
    else:
        _ = logger.add(sys.stderr, level=level.upper())


def setup_dual_logging(log_file: str) -> str:
    """
    Setup dual logging: verbose to file, minimal to console.

    Removes default handler and configures:
    - File handler: DEBUG level with detailed format
    - Console handler: INFO level with minimal format (only important messages)

    All debug/trace messages go to file only. Console shows only INFO/WARNING/ERROR.

    Args:
        log_file: Path to the log file to write verbose logs to

    Returns:
        Path to the log file created
    """
    # Remove default handler
    logger.remove()

    # Ensure log file directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old log file if it exists (start fresh)
    if log_path.exists():
        log_path.unlink()

    # Add file handler with DEBUG level (verbose, detailed format)
    _ = logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation=None,
        retention=None,
        enqueue=True,
    )

    # Add console handler with INFO level (minimal output)
    _ = logger.add(
        sys.stderr,
        format="{message}",
        level="INFO",
        colorize=True,
    )

    return str(log_path)
