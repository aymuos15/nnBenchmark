"""
Engines module for training and inference execution.

This module consolidates all training, inference, and shared Ignite/MONAI utilities.
"""

# Suppress MONAI deprecation warnings for get_mask_edges (used internally by SurfaceDiceMetric)
# Centralized here to avoid duplication in run.py files
# Note: Must be after imports to avoid E402
import warnings  # noqa: E402

from src.engines.inference import run_inference
from src.engines.train import run_training

warnings.filterwarnings("ignore", category=FutureWarning, module="monai")

__all__ = [
    "run_training",
    "run_inference",
]
