"""
Logger setup utilities for nnBenchmark.
Provides functions to configure loguru loggers for different use cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logger(
    results_dir: str, log_name: str = "train", resume: bool = False
) -> Any:
    """
    Setup loguru logger that writes to a log file only.

    Args:
        results_dir: Directory where log file will be saved
        log_name: Name of the log file (without .log extension)
        resume: If True, append to existing log file; if False, start fresh (default: False)

    Returns:
        Configured logger instance
    """
    # Remove default handler (console output)
    logger.remove()

    # Add file handler for logs
    log_path = str(Path(results_dir) / f"{log_name}.log")

    # Remove old log file if starting fresh (not resuming)
    if not resume and Path(log_path).exists():
        Path(log_path).unlink()

    _ = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation=None,  # No rotation
        retention=None,  # Keep all logs
        enqueue=True,  # Thread-safe logging
    )

    return logger


def setup_train_logger(results_dir: str, resume: bool = False) -> Any:
    """
    Setup loguru logger for training that writes to train.log file only.

    Args:
        results_dir: Directory where train.log will be saved
        resume: If True, append to existing log; if False, start fresh (default: False)

    Returns:
        Configured logger instance
    """
    return setup_logger(results_dir, "train", resume=resume)


def setup_test_logger(results_dir: str) -> Any:
    """
    Setup loguru logger for testing that writes to test.log file only.

    Args:
        results_dir: Directory where test.log will be saved

    Returns:
        Configured logger instance
    """
    return setup_logger(results_dir, "test")


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
