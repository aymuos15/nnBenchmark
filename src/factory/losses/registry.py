"""Loss registry for creating loss functions from configuration.

This module provides a registry-based factory for creating MONAI loss functions.
Loss functions are registered with their native MONAI parameter names, and the
registry handles instantiation with proper parameter extraction.
"""

from typing import Any

import torch.nn as nn
from monai import losses


class LossRegistry:
    """Registry for creating loss functions from configuration.

    The registry maintains a mapping of loss names to their MONAI classes
    and provides methods for building loss functions with native parameters.

    Example:
        >>> registry = LossRegistry()
        >>> registry.register("CustomLoss", CustomLossClass)
        >>> loss_fn = registry.build(config)
    """

    def __init__(self) -> None:
        """Initialize the loss registry with default MONAI losses."""
        self._registry: dict[str, type[nn.Module]] = {}
        self._register_default_losses()

    def _register_default_losses(self) -> None:
        """Register default MONAI loss functions."""
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

    def register(self, name: str, loss_class: type[nn.Module]) -> None:
        """Register a loss class with the given name.

        Args:
            name: Name to register the loss under
            loss_class: The loss class to register

        Raises:
            ValueError: If the name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"Loss '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = loss_class

    def unregister(
        self, name: str
    ) -> None:  # Part of public API for registry management  # noqa: D401  # noqa: D401  # noqa: D401
        """Remove a loss from the registry.

        Part of the public API for registry management.

        Args:
            name: Name of the loss to unregister

        Raises:
            KeyError: If the loss name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Loss '{name}' is not registered")
        del self._registry[name]

    def list_available(self) -> list[str]:
        """Get a list of all registered loss names.

        Returns:
            Sorted list of registered loss names
        """
        return sorted(self._registry.keys())

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

        Example:
            >>> config = {
            ...     "type": "DiceCELoss",
            ...     "to_onehot_y": True,
            ...     "softmax": True,
            ...     "batch": True
            ... }
            >>> loss_fn = registry.build(config)
        """
        loss_type = config["type"]

        if loss_type not in self._registry:
            available = ", ".join(self.list_available())
            raise KeyError(
                f"Loss type '{loss_type}' is not registered. "
                f"Available losses: {available}"
            )

        loss_class = self._registry[loss_type]

        # Extract parameters (everything except 'type')
        loss_params = {k: v for k, v in config.items() if k != "type"}

        # Instantiate loss function
        return loss_class(**loss_params)


# Create a global registry instance
loss_registry = LossRegistry()

__all__ = ["LossRegistry", "loss_registry"]
