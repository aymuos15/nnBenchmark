"""
System and hardware logging utilities for nnBenchmark.
Provides functions for logging GPU memory, system info, and training configuration.
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from loguru._logger import Logger


def format_bytes(bytes_val: float) -> str:
    """
    Convert bytes to human-readable format.

    Args:
        bytes_val: Size in bytes

    Returns:
        Formatted string (e.g., "1.50 GB" or "256.00 MB")
    """
    gb = bytes_val / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    else:
        mb = bytes_val / (1024**2)
        return f"{mb:.2f} MB"


def log_gpu_memory(
    logger: Logger, context: str, device: torch.device, reset_peak: bool = False
) -> None:
    """
    Log GPU memory statistics to the provided logger.

    Logs current and peak memory for both allocated and reserved (cached) memory.
    Only logs to file, does not print to terminal.

    Args:
        logger: Loguru logger instance to write to
        context: Context string describing when this is called (e.g., "Training Step 5/100")
        device: torch device (logs only if CUDA)
        reset_peak: If True, resets peak memory statistics after logging
    """
    if device.type != "cuda":
        return

    # Get current memory stats
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    # Log to file only
    logger.info(
        f"GPU Memory [{context}] - Allocated: {format_bytes(allocated)} (Peak: {format_bytes(max_allocated)}), Reserved: {format_bytes(reserved)} (Peak: {format_bytes(max_reserved)})"
    )

    # Reset peak stats if requested
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_memory_string(device: torch.device) -> str:
    """
    Get formatted GPU memory statistics string.

    Returns a formatted string with current and peak memory for both
    allocated and reserved (cached) memory. Returns empty string if not CUDA.

    Args:
        device: torch device

    Returns:
        Formatted GPU memory string, e.g., " | GPU: 1.23 GB (Peak: 2.34 GB), ..."
    """
    if device.type != "cuda":
        return ""

    allocated = torch.cuda.memory_allocated(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    return f" | GPU: {format_bytes(allocated)} (Peak: {format_bytes(max_allocated)}), Reserved: {format_bytes(reserved)} (Peak: {format_bytes(max_reserved)})"


def log_system_info(logger: Logger, device: torch.device) -> None:
    """
    Log system and hardware information to the provided logger.

    Logs Python version, PyTorch version, platform, device information,
    and GPU details if CUDA is available.

    Args:
        logger: Loguru logger instance to write to
        device: torch device
    """
    logger.info("SYSTEM INFO:")
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("=" * 70)
