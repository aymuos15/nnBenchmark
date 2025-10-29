"""
Training module using PyTorch Lightning.
"""

from src.train.cli import main
from src.train.run import run_training

__all__ = [
    "run_training",
    "main",
]
