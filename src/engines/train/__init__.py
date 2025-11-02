"""
Training module using MONAI SupervisedTrainer.
"""

from src.engines.train.cli import main
from src.engines.train.run import run_training

__all__ = [
    "run_training",
    "main",
]
