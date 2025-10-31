"""Model registry for creating segmentation models from configuration.

This module provides a registry-based factory for creating MONAI models.
Models are registered with their native MONAI parameter names, and the
registry handles instantiation, weight initialization, and device placement.
"""

from typing import Any

import torch
import torch.nn as nn
from monai.networks import nets as monai_nets


def _initialize_weights(module: nn.Module) -> None:
    """
    Initialize weights using Kaiming (He) Normal distribution.

    Matches nnU-Net v2.4.1 initialization strategy for deep networks.
    Applied to Conv/ConvTranspose layers with a=0.01 (LeakyReLU slope).

    Args:
        module: PyTorch module to initialize (applied via .apply())
    """
    if isinstance(
        module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    ):
        nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ModelRegistry:
    """Registry for creating models from configuration.

    The registry maintains a mapping of model names to their MONAI classes
    and provides methods for building models with proper initialization.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("CustomUNet", CustomUNetClass)
        >>> model = registry.build(config, device)
    """

    def __init__(self) -> None:
        """Initialize the model registry with default MONAI models."""
        self._registry: dict[str, type[nn.Module]] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default MONAI segmentation models."""
        # DynUNet exactly replicates nnU-Net PlainConvUNet architecture
        # Other models are not currently supported as the planning pipeline
        # is tightly coupled to DynUNet parameter generation
        self.register("DynUNet", getattr(monai_nets, "DynUNet"))

    def register(self, name: str, model_class: type[nn.Module]) -> None:
        """Register a model class with the given name.

        Args:
            name: Name to register the model under
            model_class: The model class to register

        Raises:
            ValueError: If the name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = model_class

    def unregister(
        self, name: str
    ) -> (
        None
    ):  # Part of public API for registry management  # noqa: D401  # pragma: no cover
        """Remove a model from the registry.

        Part of the public API for registry management.

        Args:
            name: Name of the model to unregister

        Raises:
            KeyError: If the model name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' is not registered")
        del self._registry[name]

    def list_available(self) -> list[str]:
        """Get a list of all registered model names.

        Returns:
            Sorted list of registered model names
        """
        return sorted(self._registry.keys())

    def build(self, config: dict[str, Any], device: torch.device) -> nn.Module:
        """Build a model from configuration with weight initialization.

        This method:
        1. Looks up the model class by type name
        2. Extracts native MONAI parameters from config
        3. Instantiates the model
        4. Applies Kaiming Normal weight initialization
        5. Moves model to specified device

        Args:
            config: Model configuration dictionary with 'type' field and
                   native MONAI parameters for that model type
            device: torch device to place model on

        Returns:
            Initialized model on specified device

        Raises:
            KeyError: If model type is not registered
            TypeError: If config parameters don't match model signature

        Example:
            >>> config = {
            ...     "type": "UNet",
            ...     "spatial_dims": 3,
            ...     "in_channels": 1,
            ...     "out_channels": 3,
            ...     "channels": [16, 32, 64],
            ...     "strides": [2, 2],
            ...     "num_res_units": 0
            ... }
            >>> model = registry.build(config, torch.device("cuda"))
        """
        model_type = config["type"]

        if model_type not in self._registry:
            available = ", ".join(self.list_available())
            raise KeyError(
                f"Model type '{model_type}' is not registered. "
                f"Available models: {available}"
            )

        model_class = self._registry[model_type]

        # Extract parameters (everything except 'type' and non-model fields)
        # Non-model fields are used by other parts of the system (e.g., ds_weights)
        exclude_keys = {"type", "ds_weights"}
        model_params = {k: v for k, v in config.items() if k not in exclude_keys}

        # Special handling for DynUNet parameters that need tuple conversion
        if model_type == "DynUNet":
            model_params = self._prepare_dynunet_params(model_params)

        # Instantiate and move to device
        model = model_class(**model_params).to(device)

        # Apply Kaiming (He) Normal initialization (nnU-Net v2.4.1 style)
        model.apply(_initialize_weights)

        return model

    def _prepare_dynunet_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Prepare DynUNet-specific parameters.

        DynUNet requires some parameters as tuples (e.g., norm_name, act_name).
        This method converts list configurations to tuples and adds default values
        to maintain nnU-Net compatibility.

        Args:
            params: Raw parameter dictionary from config

        Returns:
            Processed parameters ready for DynUNet instantiation
        """
        processed = params.copy()

        # Convert norm_name and act_name to tuples if present
        if "norm_name" in processed:
            processed["norm_name"] = tuple(processed["norm_name"])
        if "act_name" in processed:
            processed["act_name"] = tuple(processed["act_name"])

        # Add nnU-Net compatibility defaults if not present
        if "trans_bias" not in processed:
            processed["trans_bias"] = True  # nnU-Net has bias in transpose convs

        return processed


# Create a global registry instance
model_registry = ModelRegistry()
