"""
System and hardware logging utilities for nnBenchmark.
Provides functions for logging GPU memory, system info, and training configuration.
"""


import platform
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from loguru._logger import Logger


def log_system_info(logger: "Logger", device: torch.device) -> None:
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
