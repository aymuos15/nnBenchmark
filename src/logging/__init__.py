"""Logging utilities for nnBenchmark."""

from src.logging.helpers import log_and_print, log_header, log_only, log_separator
from src.logging.setup import (
    setup_dual_logging,
    setup_logger,
    setup_test_logger,
    setup_train_logger,
    setup_verbose_logger,
)
from src.logging.system import (
    get_gpu_memory_string,
    log_gpu_memory,
    log_system_info,
)

__all__ = [
    # Setup functions
    "setup_logger",
    "setup_train_logger",
    "setup_test_logger",
    "setup_verbose_logger",
    "setup_dual_logging",
    # Helper functions
    "log_and_print",
    "log_only",
    "log_separator",
    "log_header",
    # System/GPU logging
    "log_gpu_memory",
    "get_gpu_memory_string",
    "log_system_info",
]
