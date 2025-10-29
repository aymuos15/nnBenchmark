"""
Factory functions for building MONAI components from configuration.
Provides builders for models, losses, optimizers, metrics, and transforms.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from monai import losses, metrics, networks, transforms


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


def _extract_component_params(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract parameters from config, excluding 'type' field.

    Helper function to reduce boilerplate in builder functions.

    Args:
        config: Configuration dictionary with 'type' and other parameters

    Returns:
        Dictionary with all parameters except 'type'
    """
    return {k: v for k, v in config.items() if k != "type"}


def build_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    """
    Build model from configuration with Kaiming (He) Normal initialization.

    Initializes weights using Kaiming Normal distribution to match nnU-Net v2.4.1.
    This ensures better gradient flow through deep networks during training.

    Args:
        cfg: Configuration dictionary with 'model' section
        device: torch device to place model on

    Returns:
        Initialized model on specified device
    """
    model_cls = getattr(networks.nets, cfg["model"]["type"])  # type: ignore[attr-defined]
    model_params = _extract_component_params(cfg["model"])

    # Remove deep supervision parameters (handled separately in training, not in model)
    model_params.pop("deep_supervision", None)
    model_params.pop("ds_weights", None)

    model = model_cls(**model_params).to(device)

    # Apply Kaiming (He) Normal initialization (nnU-Net v2.4.1 style)
    model.apply(_initialize_weights)

    return model


def build_loss(cfg: dict[str, Any]) -> nn.Module:
    """
    Build loss function from configuration.

    Args:
        cfg: Configuration dictionary with 'loss' section

    Returns:
        Initialized loss function
    """
    loss_cls = getattr(losses, cfg["loss"]["type"])
    loss_params = _extract_component_params(cfg["loss"])
    return loss_cls(**loss_params)


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> optim.Optimizer:
    """
    Build optimizer from configuration.

    Args:
        model: PyTorch model to optimize
        cfg: Configuration dictionary with 'optimizer' and 'training' sections

    Returns:
        Initialized optimizer
    """
    optim_cls = getattr(torch.optim, cfg["optimizer"]["type"])
    optimizer_params = _extract_component_params(cfg["optimizer"])
    return optim_cls(
        model.parameters(), lr=cfg["training"]["learning_rate"], **optimizer_params
    )


def build_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Build all metrics from configuration.

    Args:
        cfg: Configuration dictionary with 'metrics' section

    Returns:
        Dictionary mapping metric names to metric instances.
        Example: {"Dice": DiceMetric(...), "HausdorffDistance": HausdorffDistanceMetric(...)}
    """
    metric_dict: dict[str, Any] = {}

    for metric_cfg in cfg["metrics"]:
        metric_type = metric_cfg["type"]
        # Extract metric name (e.g., "DiceMetric" -> "Dice")
        metric_name = metric_type.replace("Metric", "")

        metric_cls = getattr(metrics, metric_type)
        metric_params = _extract_component_params(metric_cfg)
        metric_dict[metric_name] = metric_cls(**metric_params)

    return metric_dict


def build_transforms(cfg: dict[str, Any], mode: str = "train") -> transforms.Compose:  # type: ignore[name-defined]
    """
    Build transform pipeline from configuration.

    Requires 'common', 'train', 'val', and 'test' sections in transforms config.
    Common transforms are applied first, then mode-specific transforms are inserted
    before the ToTensord transform.

    Args:
        cfg: Configuration dictionary with 'transforms' section
        mode: Transform mode ('train', 'val', or 'test')

    Returns:
        Composed MONAI transform pipeline

    Raises:
        KeyError: If required transform sections are missing
    """
    # Verify required sections exist
    if "common" not in cfg["transforms"]:
        raise KeyError(
            "Missing 'common' section in transforms config. "
            "Transforms must use the format with 'common', 'train', 'val', and 'test' sections."
        )

    if mode not in cfg["transforms"]:
        raise KeyError(
            f"Missing '{mode}' section in transforms config. "
            f"Please define transforms for mode '{mode}'."
        )

    transform_list: list[Any] = []

    # Build common transforms first
    for t_cfg in cfg["transforms"]["common"]:
        # Insert mode-specific transforms before ToTensord
        if t_cfg["type"] == "ToTensord":
            # Add mode-specific transforms before ToTensord
            for mode_t_cfg in cfg["transforms"][mode]:
                transform_cls = getattr(transforms, mode_t_cfg["type"])
                transform_params = _extract_component_params(mode_t_cfg)
                transform_list.append(transform_cls(**transform_params))

        # Add the common transform
        transform_cls = getattr(transforms, t_cfg["type"])
        transform_params = _extract_component_params(t_cfg)
        transform_list.append(transform_cls(**transform_params))

    return transforms.Compose(transform_list)  # type: ignore[attr-defined]
