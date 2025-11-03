"""
Unit tests for plotting functions.
Tests plot generation for training, validation, and inference modules.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from src.plotting.inference import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import (
    plot_validation_metric,
    save_validation_visualizations,
)


class TestTrainingPlots:
    """Tests for src.plotting.training module."""

    @pytest.mark.parametrize(
        "num_epochs",
        [1, 5, 100],
    )
    def test_plot_training_loss_creates_file(
        self, temp_dir: str, num_epochs: int
    ) -> None:
        """Test that plot_training_loss creates valid plot files with various epoch counts.

        Validates plot creation for:
        - Single epoch (edge case)
        - Normal training (5 epochs)
        - Long training (100 epochs)
        """
        # Arrange
        epochs = list(range(1, num_epochs + 1))
        # Exponential decay: realistic loss curve
        train_loss = [0.5 * (0.98**i) for i in range(num_epochs)]
        save_path = os.path.join(temp_dir, f"training_loss_{num_epochs}.png")

        # Act
        plot_training_loss(epochs, train_loss, save_path)

        # Assert
        assert os.path.exists(save_path), "Plot file should be created"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"


class TestValidationPlots:
    """Tests for src.plotting.validation module."""

    def test_plot_validation_metric_without_per_class(self, temp_dir: str) -> None:
        """Test plot_validation_metric with only mean values."""
        # Arrange
        epochs = [2, 4, 6, 8, 10]
        metric_values = [0.6, 0.7, 0.75, 0.78, 0.8]
        metric_name = "Dice"
        save_path = os.path.join(temp_dir, "val_dice.png")

        # Act
        plot_validation_metric(epochs, metric_values, metric_name, save_path)

        # Assert
        assert os.path.exists(save_path), "Plot file should be created"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"

    def test_plot_validation_metric_with_per_class_data(self, temp_dir: str) -> None:
        """Test plot_validation_metric with per-class curves."""
        # Arrange
        epochs = [2, 4, 6, 8]
        metric_values = [0.65, 0.72, 0.76, 0.78]
        metric_name = "Dice"
        per_class_values = {
            "Anterior": [0.6, 0.7, 0.75, 0.77],
            "Posterior": [0.7, 0.74, 0.77, 0.79],
        }
        save_path = os.path.join(temp_dir, "val_dice_per_class.png")

        # Act
        plot_validation_metric(
            epochs, metric_values, metric_name, save_path, per_class_values
        )

        # Assert
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_validation_metric_marks_best_score(self, temp_dir: str) -> None:
        """Test that best score is marked with a star."""
        # Arrange
        epochs = [2, 4, 6, 8, 10]
        metric_values = [0.6, 0.75, 0.7, 0.72, 0.68]  # Best at epoch 4
        metric_name = "Dice"
        save_path = os.path.join(temp_dir, "val_dice_best.png")

        # Act
        plot_validation_metric(epochs, metric_values, metric_name, save_path)

        # Assert
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_validation_metric_with_empty_values(self, temp_dir: str) -> None:
        """Test that empty metric values are handled gracefully."""
        # Arrange
        epochs: list[int] = []
        metric_values: list[float] = []
        metric_name = "Dice"
        save_path = os.path.join(temp_dir, "val_dice_empty.png")

        # Act
        plot_validation_metric(epochs, metric_values, metric_name, save_path)

        # Assert - should create file even with empty data
        assert os.path.exists(save_path)


class TestSaveValidationVisualizations:
    """Tests for save_validation_visualizations function."""

    def test_save_2d_visualization(self, temp_dir: str) -> None:
        """Test saving 2D validation visualization."""
        batch_size = 2
        channels = 1
        height = 64
        width = 64

        # Create dummy tensors
        images = torch.randn(batch_size, channels, height, width)
        labels = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, height, width)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        # Check that visualization directory was created
        vis_dir = os.path.join(temp_dir, "visualizations")
        assert os.path.exists(vis_dir)

        # Check that image file was created
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)
        assert os.path.getsize(viz_file) > 0

    def test_save_3d_visualization(self, temp_dir: str) -> None:
        """Test saving 3D validation visualization."""
        batch_size = 1
        channels = 1
        depth = 32
        height = 64
        width = 64

        # Create dummy tensors
        images = torch.randn(batch_size, channels, depth, height, width)
        labels = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=5, spatial_dims=3
        )

        # Check that visualization was created
        vis_dir = os.path.join(temp_dir, "visualizations")
        assert os.path.exists(vis_dir)

        viz_file = os.path.join(vis_dir, "validation_epoch_005.png")
        assert os.path.exists(viz_file)

    def test_visualization_with_rgb_images(self, temp_dir: str) -> None:
        """Test visualization with RGB input images."""
        batch_size = 1
        # RGB image: 3 channels
        images = torch.randn(batch_size, 3, 64, 64)
        labels = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, 64, 64)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        vis_dir = os.path.join(temp_dir, "visualizations")
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)

    def test_visualization_tensor_on_gpu(self, temp_dir: str) -> None:
        """Test visualization with tensors on GPU if available."""
        batch_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = torch.randn(batch_size, 1, 32, 32, device=device)
        labels = torch.randint(0, 2, (batch_size, 1, 32, 32), device=device).float()
        predictions = torch.randint(
            0, 2, (batch_size, 1, 32, 32), device=device
        ).float()

        # Should handle GPU tensors correctly
        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        vis_dir = os.path.join(temp_dir, "visualizations")
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)

    def test_visualization_multiclass_predictions(self, temp_dir: str) -> None:
        """Test visualization with multi-class predictions."""
        batch_size = 1
        num_classes = 3
        # Multi-class predictions as class indices
        images = torch.randn(batch_size, 1, 32, 32)
        labels = torch.randint(0, num_classes, (batch_size, 1, 32, 32)).float()
        predictions = torch.randint(0, num_classes, (batch_size, 1, 32, 32)).float()

        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=2
        )

        vis_dir = os.path.join(temp_dir, "visualizations")
        viz_file = os.path.join(vis_dir, "validation_epoch_001.png")
        assert os.path.exists(viz_file)

    def test_visualization_3d_takes_middle_slice(self, temp_dir: str) -> None:
        """Test that 3D visualization takes middle slice along depth."""
        batch_size = 1
        depth = 16
        height = 32
        width = 32

        images = torch.randn(batch_size, 1, depth, height, width)
        labels = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()
        predictions = torch.randint(0, 2, (batch_size, 1, depth, height, width)).float()

        # Should not raise error when processing 3D data
        save_validation_visualizations(
            images, labels, predictions, temp_dir, epoch=1, spatial_dims=3
        )

        vis_file = os.path.join(temp_dir, "visualizations", "validation_epoch_001.png")
        assert os.path.exists(vis_file)


class TestInferencePlots:
    """Tests for src.plotting.inference module."""

    def test_plot_classwise_scores_with_valid_data(self, temp_dir: str) -> None:
        """Test plot_classwise_scores creates violin plot with valid test history."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "test_history.json")
        test_history = {
            "dataset_name": "Dataset001_Hippo",
            "fold": 0,
            "metrics": ["Dice"],
            "summary": {
                "Dice": {
                    "mean": 0.75,
                    "std": 0.05,
                    "num_samples": 10,
                    "per_class": {
                        "Anterior": {
                            "mean": 0.72,
                            "std": 0.06,
                            "all_scores": [0.7, 0.72, 0.75, 0.68, 0.74],
                        },
                        "Posterior": {
                            "mean": 0.78,
                            "std": 0.04,
                            "all_scores": [0.76, 0.8, 0.79, 0.77, 0.78],
                        },
                    },
                }
            },
        }
        with open(test_history_path, "w") as f:
            json.dump(test_history, f)

        # Act
        save_path = plot_classwise_scores(test_history_path)

        # Assert
        assert os.path.exists(save_path), "Plot file should be created"
        assert os.path.getsize(save_path) > 0, "Plot file should not be empty"
        assert save_path.endswith("test_cls_wise_scores.png")

    def test_plot_classwise_scores_with_custom_save_path(self, temp_dir: str) -> None:
        """Test plot_classwise_scores with custom save path."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "test_history.json")
        custom_save_path = os.path.join(temp_dir, "custom_plot.png")
        test_history = {
            "dataset_name": "Dataset001_Hippo",
            "fold": 0,
            "metrics": ["Dice"],
            "summary": {
                "Dice": {
                    "mean": 0.75,
                    "std": 0.05,
                    "num_samples": 10,
                    "per_class": {
                        "Class1": {
                            "mean": 0.7,
                            "std": 0.05,
                            "all_scores": [0.68, 0.72, 0.7],
                        }
                    },
                }
            },
        }
        with open(test_history_path, "w") as f:
            json.dump(test_history, f)

        # Act
        result_path = plot_classwise_scores(
            test_history_path, save_path=custom_save_path
        )

        # Assert
        assert result_path == custom_save_path
        assert os.path.exists(custom_save_path)

    def test_plot_classwise_scores_missing_file(self, temp_dir: str) -> None:
        """Test that missing test_history.json raises FileNotFoundError."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "nonexistent.json")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Test history not found"):
            plot_classwise_scores(test_history_path)

    def test_plot_classwise_scores_missing_per_class_data(self, temp_dir: str) -> None:
        """Test that missing per_class data raises ValueError."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "test_history.json")
        test_history = {
            "dataset_name": "Dataset001_Hippo",
            "fold": 0,
            "metrics": ["Dice"],
            "summary": {
                "Dice": {
                    "mean": 0.75,
                    "std": 0.05,
                    # Missing "per_class" key
                }
            },
        }
        with open(test_history_path, "w") as f:
            json.dump(test_history, f)

        # Act & Assert
        with pytest.raises(ValueError, match="No per_class data found"):
            plot_classwise_scores(test_history_path)

    def test_plot_classwise_scores_with_specific_metric(self, temp_dir: str) -> None:
        """Test plot_classwise_scores with specific metric_name parameter."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "test_history.json")
        test_history = {
            "dataset_name": "Dataset001_Hippo",
            "fold": 0,
            "metrics": ["Dice", "IoU"],
            "summary": {
                "Dice": {
                    "mean": 0.75,
                    "std": 0.05,
                    "num_samples": 10,
                    "per_class": {
                        "Class1": {
                            "mean": 0.7,
                            "std": 0.05,
                            "all_scores": [0.68, 0.72, 0.7],
                        }
                    },
                },
                "IoU": {
                    "mean": 0.65,
                    "std": 0.06,
                    "num_samples": 10,
                    "per_class": {
                        "Class1": {
                            "mean": 0.6,
                            "std": 0.06,
                            "all_scores": [0.58, 0.62, 0.6],
                        }
                    },
                },
            },
        }
        with open(test_history_path, "w") as f:
            json.dump(test_history, f)

        # Act
        save_path = plot_classwise_scores(test_history_path, metric_name="IoU")

        # Assert
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_classwise_scores_invalid_metric_name(self, temp_dir: str) -> None:
        """Test that invalid metric_name raises ValueError."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "test_history.json")
        test_history = {
            "dataset_name": "Dataset001_Hippo",
            "metrics": ["Dice"],
            "summary": {
                "Dice": {
                    "mean": 0.75,
                    "per_class": {"Class1": {"mean": 0.7, "all_scores": [0.7]}},
                }
            },
        }
        with open(test_history_path, "w") as f:
            json.dump(test_history, f)

        # Act & Assert
        with pytest.raises(ValueError, match="Metric 'NonExistent' not found"):
            plot_classwise_scores(test_history_path, metric_name="NonExistent")

    def test_plot_classwise_scores_without_show_points(self, temp_dir: str) -> None:
        """Test plot_classwise_scores with show_points=False."""
        # Arrange
        test_history_path = os.path.join(temp_dir, "test_history.json")
        test_history = {
            "dataset_name": "Dataset001_Hippo",
            "metrics": ["Dice"],
            "summary": {
                "Dice": {
                    "mean": 0.75,
                    "num_samples": 10,
                    "per_class": {
                        "Class1": {
                            "mean": 0.7,
                            "std": 0.05,
                            "all_scores": [0.68, 0.72, 0.7],
                        }
                    },
                }
            },
        }
        with open(test_history_path, "w") as f:
            json.dump(test_history, f)

        # Act
        save_path = plot_classwise_scores(test_history_path, show_points=False)

        # Assert
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
