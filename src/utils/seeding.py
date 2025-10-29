"""
Centralised random seeding for reproducibility.

This module provides utilities for setting random seeds across all libraries
(random, numpy, torch) and managing CUDA determinism settings.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for all libraries for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators

    Args:
        seed: The random seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_cuda_determinism(deterministic: bool = False) -> None:
    """
    Enable or disable CUDA determinism for fully reproducible results.

    When deterministic=True:
    - Sets torch.backends.cudnn.deterministic = True
    - Sets torch.backends.cudnn.benchmark = False
    Note: This may result in slower execution

    When deterministic=False (default):
    - Sets torch.backends.cudnn.deterministic = False
    - Sets torch.backends.cudnn.benchmark = True
    Note: This allows for faster execution but may not be fully reproducible

    Args:
        deterministic: Whether to enable deterministic behavior (default: False for better performance)
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_seed_from_config(cfg: dict[str, Any]) -> int:
    """
    Extract seed from config with consistent priority order.

    Checks for seed in the following priority:
    1. Top-level 'seed' key
    2. 'training.seed' key
    3. 'testing.seed' key
    4. Default value of 12345

    Args:
        cfg: Configuration dictionary

    Returns:
        The seed value (int)
    """
    return (
        cfg.get("seed")
        or cfg.get("training", {}).get("seed")
        or cfg.get("testing", {}).get("seed")
        or 12345
    )
