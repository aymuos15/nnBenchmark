"""Model registry for creating segmentation models from configuration.

This module provides a registry-based factory for creating MONAI models.
Models are registered with their native MONAI parameter names, and the
registry handles instantiation, weight initialization, and device placement.
"""

from typing import Any

import torch
import torch.nn as nn
from monai.networks import nets as monai_nets

from src.factory.base_registry import BaseRegistry
from src.factory.models.kiunet import KiUNet2D, KiUNet3D


def _initialize_weights(module: nn.Module) -> None:
    """
    Initialize weights using Kaiming (He) Normal distribution.

    Matches nnU-Net v2.4.1 initialization strategy for deep models.
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


class ModelRegistry(BaseRegistry):
    """Registry for creating models from configuration.

    The registry maintains a mapping of model names to their MONAI classes
    and provides methods for building models with proper initialization.
    """

    def __init__(self) -> None:
        """Initialize the model registry with default MONAI models."""
        super().__init__()
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register default MONAI segmentation models."""
        # DynUNet exactly replicates nnU-Net PlainConvUNet architecture
        self.register("DynUNet", getattr(monai_nets, "DynUNet"))
        # UNet provides a faster alternative with simpler architecture
        self.register("UNet", getattr(monai_nets, "UNet"))
        # KiU-Net uses dual-branch architecture with cross-resolution fusion
        self.register("KiUNet2D", KiUNet2D)
        self.register("KiUNet3D", KiUNet3D)

    def build(self, config: dict[str, Any], device: torch.device) -> nn.Module:
        """Build a model from configuration with weight initialization.

        This method:
        1. Looks up the model class by type name
        2. Extracts and merges parameters (shared + model-specific if nested)
        3. Instantiates the model
        4. Applies Kaiming Normal weight initialization
        5. Moves model to specified device

        Supports both flat and nested config formats:
        - Flat: All params at top level (backward compatibility)
        - Nested: Shared params at top level, model-specific in nested sections

        Args:
            config: Model configuration dictionary with 'type' field and
                   native MONAI parameters for that model type
            device: torch device to place model on

        Returns:
            Initialized model on specified device

        Raises:
            KeyError: If model type is not registered
            TypeError: If config parameters don't match model signature
        """
        model_type = config["type"]
        model_class = self._validate_type(model_type)

        # Determine if config is nested (has model-specific sections)
        nested_keys = {"DynUNet", "UNet", "KiUNet2D", "KiUNet3D"}
        is_nested = any(key in config for key in nested_keys)

        # Non-model fields that should never be passed to model constructors
        non_model_fields = {"type", "ds_weights", "deep_supr_num"}

        # Add deep_supervision to non-model fields for UNet (doesn't support it)
        if model_type == "UNet":
            non_model_fields.add("deep_supervision")

        if is_nested:
            # Extract shared parameters (exclude 'type', nested sections, and non-model fields)
            exclude_keys = non_model_fields | nested_keys
            shared_params = {k: v for k, v in config.items() if k not in exclude_keys}

            # Get model-specific parameters from nested section
            model_specific = config.get(model_type, {})

            # Merge: shared + model-specific (model-specific takes precedence)
            model_params = {**shared_params, **model_specific}
        else:
            # Flat config (backward compatibility)
            # Extract parameters (everything except 'type' and non-model fields)
            model_params = {
                k: v for k, v in config.items() if k not in non_model_fields
            }

        # Model-specific parameter preparation
        if model_type == "DynUNet":
            model_params = self._prepare_dynunet_params(model_params)
        elif model_type == "UNet":
            model_params = self._prepare_unet_params(model_params)
        elif model_type in ("KiUNet2D", "KiUNet3D"):
            model_params = self._prepare_kiunet_params(model_params)

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

    def _prepare_unet_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Prepare UNet-specific parameters.

        UNet has simpler parameter requirements than DynUNet:
        - No tuple conversions needed (norm/act names are strings)
        - No special kernel size or stride processing required
        - Parameters are used directly as provided in config

        Args:
            params: Raw parameter dictionary from config

        Returns:
            Processed parameters ready for UNet instantiation
        """
        # UNet parameters are straightforward - just return a copy
        # No special processing needed unlike DynUNet
        return params.copy()

    def _prepare_kiunet_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Prepare KiU-Net-specific parameters.

        KiU-Net uses a configurable dual-branch architecture with:
        - features: list of channel counts at each encoder level
        - norm_name: normalization type ('batch', 'instance', 'group')
        - act_name: activation type ('relu', 'leakyrelu', 'prelu')
        - deep_supervision/deep_supr_num: deep supervision configuration

        Note: spatial_dims is removed since KiUNet2D/3D set it internally.

        Args:
            params: Raw parameter dictionary from config

        Returns:
            Processed parameters ready for KiUNet instantiation
        """
        processed = params.copy()

        # Remove spatial_dims - KiUNet2D/3D set this internally
        processed.pop("spatial_dims", None)

        # Convert features list if provided as tuple
        if "features" in processed and isinstance(processed["features"], tuple):
            processed["features"] = list(processed["features"])

        # Ensure deep_supervision defaults match nnBenchmark conventions
        if "deep_supervision" not in processed:
            processed["deep_supervision"] = False
        if "deep_supr_num" not in processed:
            processed["deep_supr_num"] = 1

        return processed


# Create a global registry instance
model_registry = ModelRegistry()
