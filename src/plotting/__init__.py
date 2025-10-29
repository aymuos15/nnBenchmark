"""
Plotting and visualization module for nnBenchmark.
Provides functions for generating training, validation, and test plots.
"""

from src.plotting.cli import generate_plots, main
from src.plotting.testing import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import (
    plot_validation_metric,
    save_validation_visualizations,
)

__all__ = [
    "plot_training_loss",
    "plot_validation_metric",
    "plot_classwise_scores",
    "save_validation_visualizations",
    "generate_plots",
    "main",
]
