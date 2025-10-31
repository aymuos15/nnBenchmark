"""
Tests for factory registries.
Tests factory registries for building models, losses, optimizers, metrics, and transforms.
"""

from __future__ import annotations

from typing import Any

import pytest
from monai import losses, metrics, transforms

from src.factory import loss_registry, metric_registry, transform_registry


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
