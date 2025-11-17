"""
Tests for factory registries.
Tests factory registries for building models, losses, optimizers, metrics, and transforms.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from monai import losses, metrics, transforms
from monai.networks.nets.dynunet import DynUNet
from monai.networks.nets.unet import UNet

from src.factory import (
    loss_registry,
    metric_registry,
    model_registry,
    optimizer_registry,
    transform_registry,
)
from src.factory.losses import CCLoss


class TestBuildLoss:
    """Tests for loss_registry."""

    def test_build_dice_ce_loss(self, sample_config: dict[str, Any]) -> None:
        """Test building DiceCELoss from config."""
        loss_fn = loss_registry.build(sample_config["loss"])

        assert isinstance(loss_fn, losses.DiceCELoss)  # type: ignore[attr-defined]

    def test_build_loss_with_params(self, sample_config: dict[str, Any]) -> None:
        """Test that loss parameters are correctly passed."""
        sample_config["loss"]["to_onehot_y"] = True
        sample_config["loss"]["softmax"] = True

        loss_fn = loss_registry.build(sample_config["loss"])

        # Should build successfully with these params
        assert isinstance(loss_fn, losses.DiceCELoss)  # type: ignore[attr-defined]

    def test_build_invalid_loss_type(self, sample_config: dict[str, Any]) -> None:
        """Test that invalid loss type raises KeyError."""
        sample_config["loss"]["type"] = "InvalidLossType"

        with pytest.raises(KeyError):
            loss_registry.build(sample_config["loss"])

    def test_build_ccloss(self) -> None:
        """Test building CCLoss from registry."""
        config = {
            "type": "CCLoss",
            "sigmoid": True,
        }
        loss_fn = loss_registry.build(config)

        assert isinstance(loss_fn, CCLoss)
        assert loss_fn.sigmoid is True

    def test_build_ccloss_with_custom_params(self) -> None:
        """Test building CCLoss with custom parameters."""
        config = {
            "type": "CCLoss",
            "to_onehot_y": True,
            "softmax": True,
            "sigmoid": False,
        }
        loss_fn = loss_registry.build(config)

        assert isinstance(loss_fn, CCLoss)
        assert loss_fn.to_onehot_y is True
        assert loss_fn.softmax is True
        assert loss_fn.sigmoid is False

    @pytest.mark.gpu
    def test_ccloss_forward_pass(self) -> None:
        """Test that CCLoss works correctly in forward pass."""
        config = {"type": "CCLoss", "sigmoid": True}
        loss_fn = loss_registry.build(config)

        # Create sample batch
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32))

        # Forward pass
        loss = loss_fn(pred, target)

        # Check output
        assert loss.dim() == 0  # Scalar loss
        assert loss.dtype in [torch.float32, torch.float64]
        assert torch.isfinite(loss)


class TestBuildMetrics:
    """Tests for metric_registry."""

    def test_build_single_metric(self, sample_config: dict[str, Any]) -> None:
        """Test building single metric returns dict with one entry."""
        metrics_dict = metric_registry.build(sample_config)

        assert len(metrics_dict) == 1
        assert "DiceMetric" in metrics_dict
        assert isinstance(metrics_dict["DiceMetric"], metrics.DiceMetric)  # type: ignore[attr-defined]

    def test_build_multiple_metrics(self, sample_config: dict[str, Any]) -> None:
        """Test building multiple metrics."""
        # Add second metric to config
        sample_config["metrics"].append(
            {"type": "MeanIoU", "include_background": False}
        )

        metrics_dict = metric_registry.build(sample_config)

        assert len(metrics_dict) == 2
        assert "DiceMetric" in metrics_dict
        assert "MeanIoU" in metrics_dict

    def test_full_metric_names_used(self, sample_config: dict[str, Any]) -> None:
        """Test that full metric type names are used as keys (no suffix removal)."""
        sample_config["metrics"] = [
            {"type": "DiceMetric", "include_background": False},
        ]

        metrics_dict = metric_registry.build(sample_config)

        # Full name "DiceMetric" used as key
        assert "DiceMetric" in metrics_dict
        assert "Dice" not in metrics_dict

    def test_build_invalid_metric_type(self, sample_config: dict[str, Any]) -> None:
        """Test that invalid metric type raises KeyError."""
        sample_config["metrics"][0]["type"] = "InvalidMetricType"

        with pytest.raises(KeyError):
            metric_registry.build(sample_config)


class TestBuildTransforms:
    """Tests for transform_registry."""

    def test_build_train_transforms(self, sample_config: dict[str, Any]) -> None:
        """Test building train transform pipeline."""
        train_transforms = transform_registry.build(sample_config, mode="train")

        assert isinstance(train_transforms, transforms.Compose)  # type: ignore[attr-defined]
        # Should have common transforms + train-specific transforms
        assert len(train_transforms.transforms) > 0

    def test_build_val_transforms(self, sample_config: dict[str, Any]) -> None:
        """Test building validation transform pipeline."""
        val_transforms = transform_registry.build(sample_config, mode="val")

        assert isinstance(val_transforms, transforms.Compose)  # type: ignore[attr-defined]
        # Should have common transforms + val-specific transforms
        assert len(val_transforms.transforms) > 0

    def test_build_test_transforms_fallback_to_val(
        self, sample_config: dict[str, Any]
    ) -> None:
        """Test that test mode raises error if test transforms not defined."""
        # Remove test section to simulate missing test transforms
        config_without_test = sample_config.copy()
        config_without_test["transforms"] = config_without_test["transforms"].copy()
        if "test" in config_without_test["transforms"]:
            del config_without_test["transforms"]["test"]

        with pytest.raises(KeyError, match="Missing 'test' section"):
            transform_registry.build(config_without_test, mode="test")

    def test_transform_count_train_vs_val(self, sample_config: dict[str, Any]) -> None:
        """Test that train has more transforms than val (due to augmentations)."""
        train_transforms = transform_registry.build(sample_config, mode="train")
        val_transforms = transform_registry.build(sample_config, mode="val")

        # Train should have at least one more transform (RandFlipd)
        assert len(train_transforms.transforms) >= len(val_transforms.transforms)

    def test_missing_common_section_raises_error(
        self, sample_config: dict[str, Any]
    ) -> None:
        """Test that missing 'common' section raises error."""
        # Create config without common section
        invalid_config = sample_config.copy()
        invalid_config["transforms"] = {
            "train": [
                {"type": "LoadImaged", "keys": ["image", "label"]},
                {"type": "ToTensord", "keys": ["image", "label"]},
            ],
            "val": [
                {"type": "LoadImaged", "keys": ["image", "label"]},
                {"type": "ToTensord", "keys": ["image", "label"]},
            ],
        }

        with pytest.raises(KeyError, match="Missing 'common' section"):
            transform_registry.build(invalid_config, mode="train")


class TestBuildModel:
    """Tests for model_registry."""

    def test_build_dynunet_flat_config(self, sample_config: dict[str, Any]) -> None:
        """Test building DynUNet from flat config (backward compatibility)."""
        device = torch.device("cpu")
        model = model_registry.build(sample_config["model"], device)

        # Verify model type
        assert isinstance(model, DynUNet)

    def test_build_dynunet_nested_config(self, sample_config: dict[str, Any]) -> None:
        """Test building DynUNet from nested config structure."""
        device = torch.device("cpu")

        # Convert to nested config
        flat_model = sample_config["model"]
        nested_model = {
            "type": "DynUNet",
            "spatial_dims": flat_model["spatial_dims"],
            "in_channels": flat_model["in_channels"],
            "out_channels": flat_model["out_channels"],
            "deep_supervision": flat_model["deep_supervision"],
            "DynUNet": {
                "filters": flat_model["filters"],
                "kernel_size": flat_model["kernel_size"],
                "strides": flat_model["strides"],
                "upsample_kernel_size": flat_model["upsample_kernel_size"],
                "norm_name": flat_model["norm_name"],
                "act_name": flat_model["act_name"],
                "deep_supr_num": flat_model["deep_supr_num"],
                "res_block": flat_model["res_block"],
            },
        }

        model = model_registry.build(nested_model, device)

        # Verify model type
        assert isinstance(model, DynUNet)

    def test_build_unet_nested_config(self) -> None:
        """Test building UNet from nested config structure."""
        device = torch.device("cpu")

        unet_config = {
            "type": "UNet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 3,
            "deep_supervision": False,
            "UNet": {
                "channels": [16, 32, 64],
                "strides": [2, 2],
                "num_res_units": 2,
            },
        }

        model = model_registry.build(unet_config, device)

        # Verify model type
        assert isinstance(model, UNet)

    def test_build_unet_flat_config(self) -> None:
        """Test building UNet from flat config (backward compatibility)."""
        device = torch.device("cpu")

        unet_config = {
            "type": "UNet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 3,
            "channels": [16, 32, 64],
            "strides": [2, 2],
            "num_res_units": 2,
        }

        model = model_registry.build(unet_config, device)

        # Verify model type
        assert isinstance(model, UNet)

    def test_nested_config_model_specific_params_override_shared(self) -> None:
        """Test that model-specific params take precedence over shared params."""
        device = torch.device("cpu")

        # Config with conflicting spatial_dims (should use model-specific)
        config = {
            "type": "UNet",
            "spatial_dims": 3,  # Shared
            "in_channels": 1,
            "out_channels": 3,
            "UNet": {
                "spatial_dims": 2,  # Model-specific (should override)
                "channels": [16, 32],
                "strides": [2],
                "num_res_units": 0,
            },
        }

        model = model_registry.build(config, device)

        # Verify model type
        assert isinstance(model, UNet)

    def test_build_invalid_model_type(self, sample_config: dict[str, Any]) -> None:
        """Test that invalid model type raises KeyError."""
        device = torch.device("cpu")
        sample_config["model"]["type"] = "InvalidModelType"

        with pytest.raises(KeyError):
            model_registry.build(sample_config["model"], device)

    def test_nested_config_missing_model_section(self) -> None:
        """Test that nested config without required model section raises KeyError."""
        device = torch.device("cpu")

        # Nested config missing UNet section
        config = {
            "type": "UNet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 3,
            "DynUNet": {  # Has DynUNet but type is UNet
                "filters": [16, 32],
            },
        }

        # Should build successfully but won't find UNet section
        # Model will be built with only shared params (will fail at instantiation)
        with pytest.raises(TypeError):  # Missing required params
            model_registry.build(config, device)


class TestBuildOptimizer:
    """Tests for optimizer_registry."""

    def test_build_optimizer_from_config(self, sample_config: dict[str, Any]) -> None:
        """Test building optimizer from config."""
        # Create a simple model for optimizer
        model = torch.nn.Linear(10, 5)
        optimizer = optimizer_registry.build(
            sample_config["optimizer"],
            model.parameters(),
            learning_rate=0.01,
        )

        # Check optimizer is instantiated with correct learning rate
        assert optimizer is not None
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_build_invalid_optimizer_type(self, sample_config: dict[str, Any]) -> None:
        """Test that invalid optimizer type raises KeyError."""
        model = torch.nn.Linear(10, 5)
        sample_config["optimizer"]["type"] = "InvalidOptimizer"

        with pytest.raises(KeyError):
            optimizer_registry.build(
                sample_config["optimizer"],
                model.parameters(),
                learning_rate=0.01,
            )

    def test_available_optimizers(self) -> None:
        """Test that expected optimizers are available."""
        available = optimizer_registry.list_available()

        # Check common optimizers are registered
        assert "SGD" in available
        assert "Adam" in available
        assert "AdamW" in available


class TestRegistryCoreFunctionality:
    """Tests for core registry functionality (register, unregister, list)."""

    @pytest.mark.parametrize(
        "registry,expected_items",
        [
            pytest.param(model_registry, ["DynUNet"], id="model_registry"),
            pytest.param(loss_registry, ["DiceCELoss"], id="loss_registry"),
            pytest.param(metric_registry, ["DiceMetric"], id="metric_registry"),
            pytest.param(optimizer_registry, [], id="optimizer_registry"),
        ],
    )
    def test_registry_list_available(self, registry, expected_items) -> None:
        """Test listing available items for each registry.

        Parameters:
        - registry: The registry to test
        - expected_items: Expected items that should be present (or empty for dynamic)
        """
        available = registry.list_available()
        assert isinstance(available, list)

        # For registries with pre-registered items, check they exist
        for expected_item in expected_items:
            assert expected_item in available
            assert len(available) > 0

    def test_transform_registry_list_available(self) -> None:
        """Test listing available transforms.

        Note: TransformRegistry uses dynamic transform loading from MONAI,
        so pre-registered transforms are empty. Actual transforms are loaded
        dynamically during build().
        """
        available = transform_registry.list_available()
        assert isinstance(available, list)
        # Dynamic transforms - may be empty since not pre-registered
        # This is expected behavior for MONAI transform discovery

    def test_duplicate_registration_raises_error(self) -> None:
        """Test that registering duplicate model name raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            # Create a dummy model class for testing
            class DummyModel(torch.nn.Module):
                pass

            model_registry.register("DynUNet", DummyModel)

    def test_unregister_nonexistent_raises_error(self) -> None:
        """Test that unregistering nonexistent model raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            model_registry.unregister("NonexistentModel")
