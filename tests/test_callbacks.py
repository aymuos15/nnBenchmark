"""
Unit tests for Lightning callbacks.
Tests TrainingHistoryCallback, ValidationVisualizationCallback, and GPUMemoryCallback.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.lightning.callbacks import (
    GPUMemoryCallback,
    TrainingHistoryCallback,
    ValidationVisualizationCallback,
)


class TestTrainingHistoryCallback:
    """Tests for TrainingHistoryCallback."""

    def test_callback_initializes_with_empty_history(self, temp_dir: str) -> None:
        """Test that callback initializes with empty history structure."""
        # Arrange
        results_dir = temp_dir

        # Act
        callback = TrainingHistoryCallback(results_dir)

        # Assert
        assert callback.training_history == {
            "epochs": [],
            "train_loss": [],
            "val_epochs": [],
        }
        assert callback.history_path == os.path.join(
            results_dir, "training_history.json"
        )

    def test_callback_loads_existing_history_on_resume(self, temp_dir: str) -> None:
        """Test that callback loads existing history when resuming training."""
        # Arrange
        results_dir = temp_dir
        history_path = os.path.join(results_dir, "training_history.json")
        existing_history = {
            "epochs": [1, 2],
            "train_loss": [0.5, 0.4],
            "val_epochs": [2],
            "val_Dice": [0.7],
        }
        with open(history_path, "w") as f:
            json.dump(existing_history, f)

        # Act
        callback = TrainingHistoryCallback(results_dir)

        # Assert
        assert callback.training_history == existing_history

    def test_on_train_epoch_end_records_loss(self, temp_dir: str) -> None:
        """Test that on_train_epoch_end records training loss."""
        # Arrange
        callback = TrainingHistoryCallback(temp_dir)
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0  # 0-indexed
        mock_trainer.callback_metrics = {"train_loss_epoch": torch.tensor(0.5)}
        mock_pl_module = Mock()

        # Act
        callback.on_train_epoch_end(mock_trainer, mock_pl_module)

        # Assert
        assert callback.training_history["epochs"] == [1]  # 1-indexed
        assert callback.training_history["train_loss"] == [0.5]

        # Check that file was saved
        history_path = os.path.join(temp_dir, "training_history.json")
        assert os.path.exists(history_path)
        with open(history_path) as f:
            saved_history = json.load(f)
        assert saved_history["epochs"] == [1]
        assert saved_history["train_loss"] == [0.5]

    def test_on_train_epoch_end_avoids_duplicates(self, temp_dir: str) -> None:
        """Test that on_train_epoch_end avoids recording duplicate epochs."""
        # Arrange
        callback = TrainingHistoryCallback(temp_dir)
        callback.training_history = {
            "epochs": [1],
            "train_loss": [0.5],
            "val_epochs": [],
        }
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0  # Same epoch
        mock_trainer.callback_metrics = {"train_loss_epoch": torch.tensor(0.4)}
        mock_pl_module = Mock()

        # Act
        callback.on_train_epoch_end(mock_trainer, mock_pl_module)

        # Assert - should not add duplicate
        assert len(callback.training_history["epochs"]) == 1
        assert len(callback.training_history["train_loss"]) == 1

    def test_on_validation_end_records_metrics(self, temp_dir: str) -> None:
        """Test that on_validation_end records validation metrics."""
        # Arrange
        callback = TrainingHistoryCallback(temp_dir)
        callback.training_history = {
            "epochs": [1, 2],
            "train_loss": [0.5, 0.4],
            "val_epochs": [],
        }

        mock_trainer = Mock()
        mock_trainer.current_epoch = 1  # Epoch 2 (0-indexed)
        mock_trainer.state.stage = MagicMock()
        # Mock the stage to not be SANITY_CHECKING
        from pytorch_lightning.trainer.states import RunningStage

        mock_trainer.state.stage = RunningStage.TRAINING

        mock_trainer.callback_metrics = {
            "val_Dice": torch.tensor(0.75),
            "val_IoU": torch.tensor(0.65),
        }
        mock_pl_module = Mock()

        # Act
        callback.on_validation_end(mock_trainer, mock_pl_module)

        # Assert
        assert callback.training_history["val_epochs"] == [2]
        assert callback.training_history["val_Dice"] == pytest.approx([0.75])
        assert callback.training_history["val_IoU"] == pytest.approx([0.65])

    def test_on_validation_end_skips_sanity_check(self, temp_dir: str) -> None:
        """Test that on_validation_end skips sanity checking phase."""
        # Arrange
        callback = TrainingHistoryCallback(temp_dir)
        mock_trainer = Mock()
        from pytorch_lightning.trainer.states import RunningStage

        mock_trainer.state.stage = RunningStage.SANITY_CHECKING
        mock_trainer.callback_metrics = {"val_Dice": torch.tensor(0.75)}
        mock_pl_module = Mock()

        # Act
        callback.on_validation_end(mock_trainer, mock_pl_module)

        # Assert - should not record anything
        assert callback.training_history["val_epochs"] == []

    def test_on_validation_end_avoids_duplicate_epochs(self, temp_dir: str) -> None:
        """Test that on_validation_end avoids recording duplicate validation epochs."""
        # Arrange
        callback = TrainingHistoryCallback(temp_dir)
        callback.training_history = {
            "epochs": [1, 2],
            "train_loss": [0.5, 0.4],
            "val_epochs": [2],
            "val_Dice": [0.75],
        }

        mock_trainer = Mock()
        mock_trainer.current_epoch = 1  # Same epoch as already recorded
        from pytorch_lightning.trainer.states import RunningStage

        mock_trainer.state.stage = RunningStage.TRAINING
        mock_trainer.callback_metrics = {"val_Dice": torch.tensor(0.8)}
        mock_pl_module = Mock()

        # Act
        callback.on_validation_end(mock_trainer, mock_pl_module)

        # Assert - should not add duplicate
        assert len(callback.training_history["val_epochs"]) == 1
        assert len(callback.training_history["val_Dice"]) == 1


class TestValidationVisualizationCallback:
    """Tests for ValidationVisualizationCallback."""

    def test_callback_initializes_correctly(self, temp_dir: str) -> None:
        """Test that callback initializes with correct settings."""
        # Arrange
        cfg = {"model": {"spatial_dims": 3}}

        # Act
        callback = ValidationVisualizationCallback(temp_dir, cfg)

        # Assert
        assert callback.results_dir == temp_dir
        assert callback.spatial_dims == 3
        assert callback.save_viz is True

    def test_callback_disables_viz_for_unsupported_dims(self, temp_dir: str) -> None:
        """Test that callback disables visualization for unsupported spatial dims."""
        # Arrange
        cfg = {"model": {"spatial_dims": 1}}  # 1D not supported

        # Act
        callback = ValidationVisualizationCallback(temp_dir, cfg)

        # Assert
        assert callback.save_viz is False

    @patch("src.lightning.callbacks.save_validation_visualizations")
    def test_on_validation_batch_end_saves_first_batch(
        self, mock_save_viz: Mock, temp_dir: str
    ) -> None:
        """Test that on_validation_batch_end saves visualization for first batch."""
        # Arrange
        cfg = {"model": {"spatial_dims": 3}}
        callback = ValidationVisualizationCallback(temp_dir, cfg)

        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_pl_module = Mock()

        outputs = {
            "images": torch.randn(2, 1, 32, 32, 32),
            "labels": torch.randint(0, 3, (2, 1, 32, 32, 32)),
            "predictions": torch.randint(0, 3, (2, 1, 32, 32, 32)),
        }
        batch = None
        batch_idx = 0

        # Act
        callback.on_validation_batch_end(
            mock_trainer, mock_pl_module, outputs, batch, batch_idx
        )

        # Assert
        mock_save_viz.assert_called_once()
        call_args = mock_save_viz.call_args
        assert torch.equal(call_args.kwargs["images"], outputs["images"])
        assert torch.equal(call_args.kwargs["labels"], outputs["labels"])
        assert torch.equal(call_args.kwargs["predictions"], outputs["predictions"])
        assert call_args.kwargs["save_dir"] == temp_dir
        assert call_args.kwargs["epoch"] == 1  # 1-indexed
        assert call_args.kwargs["spatial_dims"] == 3

    @patch("src.lightning.callbacks.save_validation_visualizations")
    def test_on_validation_batch_end_skips_non_first_batch(
        self, mock_save_viz: Mock, temp_dir: str
    ) -> None:
        """Test that on_validation_batch_end skips non-first batches."""
        # Arrange
        cfg = {"model": {"spatial_dims": 3}}
        callback = ValidationVisualizationCallback(temp_dir, cfg)

        mock_trainer = Mock()
        mock_pl_module = Mock()
        outputs = {"images": None, "labels": None, "predictions": None}
        batch = None
        batch_idx = 1  # Not first batch

        # Act
        callback.on_validation_batch_end(
            mock_trainer, mock_pl_module, outputs, batch, batch_idx
        )

        # Assert
        mock_save_viz.assert_not_called()

    @patch("src.lightning.callbacks.save_validation_visualizations")
    def test_on_validation_batch_end_skips_when_viz_disabled(
        self, mock_save_viz: Mock, temp_dir: str
    ) -> None:
        """Test that on_validation_batch_end skips when visualization is disabled."""
        # Arrange
        cfg = {"model": {"spatial_dims": 1}}  # Unsupported, viz disabled
        callback = ValidationVisualizationCallback(temp_dir, cfg)

        mock_trainer = Mock()
        mock_pl_module = Mock()
        outputs = {"images": None, "labels": None, "predictions": None}
        batch = None
        batch_idx = 0

        # Act
        callback.on_validation_batch_end(
            mock_trainer, mock_pl_module, outputs, batch, batch_idx
        )

        # Assert
        mock_save_viz.assert_not_called()


class TestGPUMemoryCallback:
    """Tests for GPUMemoryCallback."""

    @patch("src.lightning.callbacks.log_gpu_memory")
    def test_on_train_epoch_start_logs_memory(self, mock_log_gpu: Mock) -> None:
        """Test that on_train_epoch_start logs GPU memory."""
        # Arrange
        mock_logger = Mock()
        callback = GPUMemoryCallback(mock_logger)

        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_pl_module = Mock()
        mock_pl_module.device = torch.device("cuda:0")

        # Act
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)

        # Assert
        mock_log_gpu.assert_called_once_with(
            mock_logger,
            "Epoch 1 Start",
            torch.device("cuda:0"),
            reset_peak=True,
        )

    @patch("src.lightning.callbacks.log_gpu_memory")
    def test_on_validation_start_logs_memory(self, mock_log_gpu: Mock) -> None:
        """Test that on_validation_start logs GPU memory."""
        # Arrange
        mock_logger = Mock()
        callback = GPUMemoryCallback(mock_logger)

        mock_trainer = Mock()
        mock_trainer.current_epoch = 2  # Epoch 3 (0-indexed)
        mock_pl_module = Mock()
        mock_pl_module.device = torch.device("cuda:0")

        # Act
        callback.on_validation_start(mock_trainer, mock_pl_module)

        # Assert
        mock_log_gpu.assert_called_once_with(
            mock_logger,
            "Before Validation (Epoch 3)",
            torch.device("cuda:0"),
        )

    @patch("src.lightning.callbacks.log_gpu_memory")
    def test_gpu_memory_callback_with_cpu_device(self, mock_log_gpu: Mock) -> None:
        """Test that GPU memory callback works with CPU device."""
        # Arrange
        mock_logger = Mock()
        callback = GPUMemoryCallback(mock_logger)

        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_pl_module = Mock()
        mock_pl_module.device = torch.device("cpu")

        # Act
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)

        # Assert - should still call log_gpu_memory (it handles CPU internally)
        mock_log_gpu.assert_called_once()
