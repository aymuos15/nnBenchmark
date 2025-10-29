"""
Unit tests for plotting functions.
Tests plot generation for training, validation, and testing modules.
"""

from __future__ import annotations

import json
import os

import pytest

from src.plotting.testing import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import plot_validation_metric


class TestTrainingPlots:
    """Tests for src.plotting.training module."""

    @pytest.mark.parametrize(
        "num_epochs",
        [1, 5, 100],
    )
    def test_plot_training_loss_creates_file(self, temp_dir: str, num_epochs: int) -> None:
        """Test that plot_training_loss creates valid plot files with various epoch counts.

        Validates plot creation for:
        - Single epoch (edge case)
        - Normal training (5 epochs)
        - Long training (100 epochs)
        """
        # Arrange
        epochs = list(range(1, num_epochs + 1))
        # Exponential decay: realistic loss curve
        train_loss = [0.5 * (0.98 ** i) for i in range(num_epochs)]
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


class TestTestingPlots:
    """Tests for src.plotting.testing module."""

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
        result_path = plot_classwise_scores(test_history_path, save_path=custom_save_path)

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
