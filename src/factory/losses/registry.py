"""Loss registry for creating loss functions from configuration.

This module provides a registry-based factory for creating MONAI loss functions.
Loss functions are registered with their native MONAI parameter names, and the
registry handles instantiation with proper parameter extraction.
"""

from typing import Any

import torch.nn as nn
from monai import losses

from src.factory.base_registry import BaseRegistry
from src.factory.losses.blob import BlobLoss
from src.factory.losses.cc import CCLoss


class LossRegistry(BaseRegistry):
    """Registry for creating loss functions from configuration.

    The registry maintains a mapping of loss names to their MONAI classes
    and provides methods for building loss functions with native parameters.
    """

    def __init__(self) -> None:
        """Initialize the loss registry with default MONAI losses."""
        super().__init__()
        self._register_default_losses()

    def _register_default_losses(self) -> None:
        """Register default MONAI loss functions and custom losses."""
        # Register common MONAI losses for segmentation
        self.register("DiceCELoss", getattr(losses, "DiceCELoss"))
        self.register("DiceLoss", getattr(losses, "DiceLoss"))
        self.register("DiceFocalLoss", getattr(losses, "DiceFocalLoss"))
        self.register("FocalLoss", getattr(losses, "FocalLoss"))
        self.register("GeneralizedDiceLoss", getattr(losses, "GeneralizedDiceLoss"))
        self.register("TverskyLoss", getattr(losses, "TverskyLoss"))
        self.register(
            "GeneralizedWassersteinDiceLoss",
            getattr(losses, "GeneralizedWassersteinDiceLoss"),
        )
        self.register("MaskedDiceLoss", getattr(losses, "MaskedDiceLoss"))

        # Register custom losses
        self.register("BlobLoss", BlobLoss)
        self.register("CCLoss", CCLoss)

    def build(self, config: dict[str, Any]) -> nn.Module:
        """Build a loss function from configuration.

        This method:
        1. Looks up the loss class by type name
        2. Extracts native MONAI parameters from config
        3. Instantiates the loss function

        Args:
            config: Loss configuration dictionary with 'type' field and
                   native MONAI parameters for that loss type

        Returns:
            Initialized loss function

        Raises:
            KeyError: If loss type is not registered
            TypeError: If config parameters don't match loss signature
        """
        loss_type = config["type"]
        loss_class = self._validate_type(loss_type)

        # Extract parameters (everything except 'type')
        loss_params = self._extract_params(config)

        # Instantiate loss function
        return loss_class(**loss_params)


# Create a global registry instance
loss_registry = LossRegistry()

__all__ = ["LossRegistry", "loss_registry"]
