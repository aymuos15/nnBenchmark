"""Loss factory for creating loss functions from configuration."""

from src.factory.losses.registry import LossRegistry, loss_registry

__all__ = ["LossRegistry", "loss_registry"]
