"""Factory pattern implementation for all nnBenchmark components.

This module provides a formal factory pattern for creating models, losses,
optimizers, metrics, and transforms from configuration. Each component type
has its own registry that manages available implementations and handles instantiation.
"""

from src.factory.losses.registry import loss_registry
from src.factory.metrics.registry import metric_registry
from src.factory.models.registry import model_registry
from src.factory.optimizers.registry import optimizer_registry
from src.factory.transforms.registry import transform_registry

__all__ = [
    "model_registry",
    "loss_registry",
    "optimizer_registry",
    "metric_registry",
    "transform_registry",
]
