"""Optimizer registry for creating optimizers from configuration.

This module provides a registry-based factory for creating PyTorch optimizers.
Optimizers are registered with their native PyTorch parameter names, and the
registry handles instantiation with proper parameter extraction.
"""

from typing import Any, Iterator

import torch
import torch.nn as nn


class OptimizerRegistry:
    """Registry for creating optimizers from configuration.

    The registry maintains a mapping of optimizer names to their PyTorch classes
    and provides methods for building optimizers with native parameters.
    """

    def __init__(self) -> None:
        """Initialize the optimizer registry with default PyTorch optimizers."""
        self._registry: dict[str, type[Any]] = {}
        self._register_default_optimizers()

    def _register_default_optimizers(self) -> None:
        """Register default PyTorch optimizers."""
        # Register common PyTorch optimizers
        self.register("SGD", torch.optim.SGD)
        self.register("Adam", torch.optim.Adam)
        self.register("AdamW", torch.optim.AdamW)
        self.register("RMSprop", torch.optim.RMSprop)
        self.register("Adagrad", torch.optim.Adagrad)
        self.register("Adadelta", torch.optim.Adadelta)
        self.register("Adamax", torch.optim.Adamax)
        self.register("NAdam", torch.optim.NAdam)
        self.register("RAdam", torch.optim.RAdam)

    def register(self, name: str, optimizer_class: type[Any]) -> None:
        """Register an optimizer class with the given name.

        Args:
            name: Name to register the optimizer under
            optimizer_class: The optimizer class to register

        Raises:
            ValueError: If the name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"Optimizer '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = optimizer_class

    def unregister(
        self, name: str
    ) -> (
        None
    ):  # Part of public API for registry management  # noqa: D401  # pragma: no cover
        """Remove an optimizer from the registry.

        Part of the public API for registry management.

        Args:
            name: Name of the optimizer to unregister

        Raises:
            KeyError: If the optimizer name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Optimizer '{name}' is not registered")
        del self._registry[name]

    def list_available(self) -> list[str]:
        """Get a list of all registered optimizer names.

        Returns:
            Sorted list of registered optimizer names
        """
        return sorted(self._registry.keys())

    def build(
        self,
        config: dict[str, Any],
        params: Iterator[nn.Parameter],
        learning_rate: float,
    ) -> Any:
        """Build an optimizer from configuration.

        This method:
        1. Looks up the optimizer class by type name
        2. Extracts native PyTorch parameters from config
        3. Instantiates the optimizer with model parameters and learning rate

        Args:
            config: Optimizer configuration dictionary with 'type' field and
                   native PyTorch parameters for that optimizer type
            params: Iterator of model parameters to optimize
            learning_rate: Learning rate for the optimizer

        Returns:
            Initialized optimizer

        Raises:
            KeyError: If optimizer type is not registered
            TypeError: If config parameters don't match optimizer signature
        """
        optimizer_type = config["type"]

        if optimizer_type not in self._registry:
            available = ", ".join(self.list_available())
            raise KeyError(
                f"Optimizer type '{optimizer_type}' is not registered. "
                f"Available optimizers: {available}"
            )

        optimizer_class = self._registry[optimizer_type]

        # Extract parameters (everything except 'type')
        optimizer_params = {k: v for k, v in config.items() if k != "type"}

        # Instantiate optimizer with parameters and learning rate
        return optimizer_class(params, lr=learning_rate, **optimizer_params)


# Create a global registry instance
optimizer_registry = OptimizerRegistry()
