"""
Engines module for training and inference execution.

This module consolidates all training, inference, and shared Ignite/MONAI utilities.
"""

from src.engines.inference import run_inference
from src.engines.train import run_training

__all__ = [
    "run_training",
    "run_inference",
]
