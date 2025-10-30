"""
Tests for deep supervision configuration and loss computation.
"""

from __future__ import annotations

import pytest
import torch

from src.config.validation import validate_deep_supervision_config
from src.planning.planner.heuristics import calculate_deep_supervision_weights


class TestDeepSupervisionWeights:
    """Test automatic deep supervision weight calculation."""

    def test_weights_3_stages(self) -> None:
        """Test weight calculation for 3 decoder stages."""
        weights = calculate_deep_supervision_weights(3)
        assert len(weights) == 3
        assert weights == [1.0, 0.5, 0.25]

    def test_weights_4_stages(self) -> None:
        """Test weight calculation for 4 decoder stages."""
        weights = calculate_deep_supervision_weights(4)
        assert len(weights) == 4
        assert weights == [1.0, 0.5, 0.25, 0.125]

    def test_weights_1_stage(self) -> None:
        """Test weight calculation for single stage (no deep supervision)."""
        weights = calculate_deep_supervision_weights(1)
        assert len(weights) == 1
        assert weights == [1.0]

    def test_weights_decreasing(self) -> None:
        """Test that weights follow exponential decay pattern."""
        for num_stages in range(1, 6):
            weights = calculate_deep_supervision_weights(num_stages)
            # Each weight should be half of previous
            for i in range(len(weights) - 1):
                assert abs(weights[i + 1] - weights[i] / 2.0) < 1e-10

    def test_weights_positive(self) -> None:
        """Test that all weights are positive."""
        for num_stages in range(1, 10):
            weights = calculate_deep_supervision_weights(num_stages)
            assert all(w > 0 for w in weights)


class TestDeepSupervisionValidation:
    """Test deep supervision configuration validation."""

    def test_validation_disabled(self) -> None:
        """Test validation passes when deep supervision is disabled."""
        cfg = {"model": {"deep_supervision": False}}
        # Should not raise
        validate_deep_supervision_config(cfg)

    def test_validation_enabled_with_weights(self) -> None:
        """Test validation passes when enabled with valid weights."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [1.0, 0.5, 0.25],
            }
        }
        # Should not raise
        validate_deep_supervision_config(cfg)

    def test_validation_enabled_missing_weights(self) -> None:
        """Test validation fails when enabled without ds_weights."""
        cfg = {"model": {"deep_supervision": True}}
        with pytest.raises(ValueError, match="ds_weights"):
            validate_deep_supervision_config(cfg)

    def test_validation_weights_not_list(self) -> None:
        """Test validation fails when ds_weights is not a list."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": "1.0, 0.5, 0.25",  # String instead of list
            }
        }
        with pytest.raises(ValueError, match="must be a list"):
            validate_deep_supervision_config(cfg)

    def test_validation_weights_empty_list(self) -> None:
        """Test validation fails when ds_weights is empty."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [],
            }
        }
        with pytest.raises(ValueError, match="non-empty"):
            validate_deep_supervision_config(cfg)

    def test_validation_weights_non_numeric(self) -> None:
        """Test validation fails when weights contain non-numeric values."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [1.0, "0.5", 0.25],  # String in list
            }
        }
        with pytest.raises(ValueError, match="must be a number"):
            validate_deep_supervision_config(cfg)

    def test_validation_weights_negative(self) -> None:
        """Test validation fails when weights are negative."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [1.0, -0.5, 0.25],
            }
        }
        with pytest.raises(ValueError, match="must be positive"):
            validate_deep_supervision_config(cfg)

    def test_validation_weights_zero(self) -> None:
        """Test validation fails when weights are zero."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [1.0, 0.0, 0.25],
            }
        }
        with pytest.raises(ValueError, match="must be positive"):
            validate_deep_supervision_config(cfg)

    def test_validation_no_model_section(self) -> None:
        """Test validation passes when no model section exists."""
        cfg = {}
        # Should not raise - defaults to disabled
        validate_deep_supervision_config(cfg)

    def test_validation_equal_weights(self) -> None:
        """Test validation passes with equal weight scheme."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [1.0, 1.0, 1.0],
            }
        }
        validate_deep_supervision_config(cfg)

    def test_validation_normalized_weights(self) -> None:
        """Test validation passes with normalized weights."""
        cfg = {
            "model": {
                "deep_supervision": True,
                "ds_weights": [0.57, 0.29, 0.14],
            }
        }
        validate_deep_supervision_config(cfg)


class TestDeepSupervisionLossComputation:
    """Test deep supervision loss computation in LightningModule."""

    def test_deep_supervision_enabled(self) -> None:
        """Test that deep supervision configuration is properly loaded."""
        from unittest.mock import MagicMock, patch

        from src.lightning.module import SegmentationModule

        cfg = {
            "dataset": {"num_classes": 3},
            "model": {
                "type": "UNet",
                "deep_supervision": True,
                "ds_weights": [1.0, 0.5, 0.25],
            },
            "optimizer": {"type": "Adam"},
            "loss": {"type": "DiceCELoss", "to_onehot_y": True},
            "training": {"checkpoint_metric": "Dice", "plot_metrics": ["Dice"]},
            "metrics": [{"type": "DiceMetric"}],
        }

        device = torch.device("cpu")

        with (
            patch("src.lightning.module.build_model"),
            patch("src.lightning.module.build_loss"),
            patch(
                "src.lightning.module.build_metrics", return_value={"Dice": MagicMock()}
            ),
        ):
            module = SegmentationModule(cfg, device)

        assert module.deep_supervision is True
        assert module.ds_weights == [1.0, 0.5, 0.25]

    def test_deep_supervision_disabled(self) -> None:
        """Test that deep supervision can be disabled."""
        from unittest.mock import MagicMock, patch

        from src.lightning.module import SegmentationModule

        cfg = {
            "dataset": {"num_classes": 3},
            "model": {
                "type": "UNet",
                "deep_supervision": False,
                "ds_weights": [],
            },
            "optimizer": {"type": "Adam"},
            "loss": {"type": "DiceCELoss", "to_onehot_y": True},
            "training": {"checkpoint_metric": "Dice", "plot_metrics": ["Dice"]},
            "metrics": [{"type": "DiceMetric"}],
        }

        device = torch.device("cpu")

        with (
            patch("src.lightning.module.build_model"),
            patch("src.lightning.module.build_loss"),
            patch(
                "src.lightning.module.build_metrics", return_value={"Dice": MagicMock()}
            ),
        ):
            module = SegmentationModule(cfg, device)

        assert module.deep_supervision is False
        assert module.ds_weights == []

    def test_deep_supervision_missing_weights_raises(self) -> None:
        """Test that error is raised if deep supervision enabled without weights."""
        from unittest.mock import MagicMock, patch

        from src.lightning.module import SegmentationModule

        cfg = {
            "dataset": {"num_classes": 3},
            "model": {
                "type": "UNet",
                "deep_supervision": True,
                # Missing ds_weights
            },
            "optimizer": {"type": "Adam"},
            "loss": {"type": "DiceCELoss", "to_onehot_y": True},
            "training": {"checkpoint_metric": "Dice", "plot_metrics": ["Dice"]},
            "metrics": [{"type": "DiceMetric"}],
        }

        device = torch.device("cpu")

        with (
            patch("src.lightning.module.build_model"),
            patch("src.lightning.module.build_loss"),
            patch(
                "src.lightning.module.build_metrics", return_value={"Dice": MagicMock()}
            ),
        ):
            with pytest.raises(ValueError, match="ds_weights"):
                SegmentationModule(cfg, device)

    def test_compute_loss_mismatch_raises(self) -> None:
        """Test that error is raised if output count doesn't match weight count."""
        from unittest.mock import MagicMock, patch

        from src.lightning.module import SegmentationModule

        cfg = {
            "dataset": {"num_classes": 3},
            "model": {
                "type": "UNet",
                "deep_supervision": True,
                "ds_weights": [1.0, 0.5],  # 2 weights
            },
            "optimizer": {"type": "Adam"},
            "loss": {"type": "DiceCELoss", "to_onehot_y": True},
            "training": {"checkpoint_metric": "Dice", "plot_metrics": ["Dice"]},
            "metrics": [{"type": "DiceMetric"}],
        }

        device = torch.device("cpu")

        with (
            patch("src.lightning.module.build_model"),
            patch("src.lightning.module.build_loss"),
            patch(
                "src.lightning.module.build_metrics", return_value={"Dice": MagicMock()}
            ),
        ):
            module = SegmentationModule(cfg, device)

        # Create dummy tensors
        outputs = [
            torch.randn(2, 3, 32, 32, 32),  # 3 outputs
            torch.randn(2, 3, 16, 16, 16),
            torch.randn(2, 3, 8, 8, 8),
        ]
        labels = torch.randint(0, 3, (2, 1, 32, 32, 32))

        with pytest.raises(ValueError, match="doesn't match"):
            module._compute_deep_supervision_loss(outputs, labels)


class TestDeepSupervisionIntegration:
    """Integration tests for deep supervision with training pipeline."""

    def test_yaml_generation_includes_deep_supervision(self, tmp_path) -> None:
        """Test that YAML generator includes deep supervision config."""
        from src.planning.planner.create import ExperimentPlan
        from src.planning.yaml_generator import generate_config_yaml

        plan = ExperimentPlan(
            dataset_name="TestDataset",
            num_classes=3,
            is_2d=False,
            patch_size=(32, 48, 32),
            batch_size=2,
            filters=[32, 64, 128],
            kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(2, 2, 2), (2, 2, 2)],
            upsample_kernel_size=[(2, 2, 2), (2, 2, 2)],
            deep_supervision=True,
            ds_weights=[1.0, 0.5, 0.25],
            normalization_scheme="ZScoreNormalization",
            intensity_clip_min=0.0,
            intensity_clip_max=100.0,
            target_spacing=(1.0, 1.0, 1.0),
        )

        output_path = tmp_path / "test_config.yaml"
        generate_config_yaml(plan, "datasets/TestDataset", str(output_path))

        # Verify file was created
        assert output_path.exists()

        # Verify deep supervision is in config
        content = output_path.read_text()
        assert "deep_supervision: true" in content
        assert "ds_weights:" in content
        assert "[1.0, 0.5, 0.25]" in content
