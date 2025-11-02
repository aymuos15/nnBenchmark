"""Optimizer registry for creating optimizers from configuration.

This module provides a registry-based factory for creating PyTorch optimizers.
Optimizers are registered with their native PyTorch parameter names, and the
registry handles instantiation with proper parameter extraction.
"""

from typing import Any, Iterator

import torch
import torch.nn as nn

from src.factory.base_registry import BaseRegistry


class OptimizerRegistry(BaseRegistry):
    """Registry for creating optimizers from configuration.

    The registry maintains a mapping of optimizer names to their PyTorch classes
    and provides methods for building optimizers with native parameters.
    """

    def __init__(self) -> None:
        """Initialize the optimizer registry with default PyTorch optimizers."""
        super().__init__()
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
        optimizer_class = self._validate_type(optimizer_type)

        # Extract parameters (everything except 'type')
        optimizer_params = self._extract_params(config)

        # Instantiate optimizer with parameters and learning rate
        return optimizer_class(params, lr=learning_rate, **optimizer_params)


# Create a global registry instance
optimizer_registry = OptimizerRegistry()
