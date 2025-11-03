"""
Custom MONAI handlers for nnBenchmark.
Maintains exact output formats (training_history.json, visualizations, logs).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from ignite.engine import Engine, Events

from src.logging import log_only
from src.plotting.validation import save_validation_visualizations
from src.utils.files import save_json

if TYPE_CHECKING:
    from loguru._logger import Logger


class TrainingHistoryHandler:
    """
    Records training/validation metrics to training_history.json.
    Maintains exact same format as Lightning implementation for plot.py compatibility.
    """

    def __init__(
        self, results_dir: str, training_all_data: bool = False, resume: bool = False
    ):
        """
        Args:
            results_dir: Directory to save training_history.json
            training_all_data: If True, skip validation metrics (training on all data)
            resume: If True, load existing history file
        """
        self.results_dir = results_dir
        self.history_path = str(Path(results_dir) / "training_history.json")
        self.training_all_data = training_all_data
        self.best_val_dice = 0.0

        # Load existing history if resuming, otherwise start fresh
        if resume and Path(self.history_path).exists():
            import json

            with open(self.history_path) as f:
                self.training_history = json.load(f)
                # Restore best_val_dice from history if available
                if "best_val_DiceMetric" in self.training_history:
                    best_list = self.training_history["best_val_DiceMetric"]
                    self.best_val_dice = best_list[-1] if best_list else 0.0
        else:
            # Initialize history structure (matches original format)
            self.training_history: dict[str, Any] = {
                "epochs": [],
                "train_loss": [],
                "val_epochs": [],
                "best_val_DiceMetric": [],
            }

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        # Record training loss after each epoch
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._record_training_epoch)

    def _record_training_epoch(self, engine: Engine) -> None:
        """Record training loss after each epoch."""
        current_epoch = engine.state.epoch

        # Get train_loss from engine state
        train_loss = engine.state.output.get("loss", None)  # type: ignore[union-attr]

        if train_loss is not None:
            # Avoid duplicates when resuming
            if current_epoch not in self.training_history["epochs"]:
                self.training_history["epochs"].append(current_epoch)
                # Convert tensor to float if needed
                loss_val = (
                    train_loss.item()
                    if isinstance(train_loss, torch.Tensor)
                    else train_loss
                )
                self.training_history["train_loss"].append(loss_val)
                # Save immediately to persist on-the-fly
                save_json(self.training_history, self.history_path)

    def record_validation_metrics(
        self, epoch: int, metrics: dict[str, torch.Tensor | float]
    ) -> None:
        """
        Record validation metrics manually (called from validation workflow).

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names and values
        """
        if self.training_all_data:
            return

        # Only record if this is a new validation epoch (avoid duplicates)
        if epoch in self.training_history["val_epochs"]:
            return

        # Record validation epoch
        self.training_history["val_epochs"].append(epoch)

        # Record all validation metrics (including per-class)
        for key, value in metrics.items():
            if key.startswith("val_"):
                # Initialize metric list if needed
                if key not in self.training_history:
                    self.training_history[key] = []

                # Append metric value
                metric_val = value.item() if isinstance(value, torch.Tensor) else value
                self.training_history[key].append(metric_val)

        # Track best validation dice metric
        val_dice = metrics.get("val_DiceMetric", None)
        if val_dice is not None:
            dice_val = (
                val_dice.item() if isinstance(val_dice, torch.Tensor) else val_dice
            )
            if dice_val > self.best_val_dice:
                self.best_val_dice = dice_val

            # Store best dice in history (one entry per validation epoch)
            if "best_val_DiceMetric" not in self.training_history:
                self.training_history["best_val_DiceMetric"] = []
            self.training_history["best_val_DiceMetric"].append(self.best_val_dice)

        # Save after each validation
        save_json(self.training_history, self.history_path)


class ValidationVisualizationHandler:
    """
    Saves validation visualizations using existing viz utilities.
    Creates PNG files in visualizations/ directory.
    """

    def __init__(
        self, results_dir: str, spatial_dims: int, skip_if_no_validation: bool = False
    ):
        """
        Args:
            results_dir: Directory to save visualizations
            spatial_dims: Number of spatial dimensions (2 or 3)
            skip_if_no_validation: If True, skip visualization when there's no validation
        """
        self.results_dir = results_dir
        self.spatial_dims = spatial_dims
        self.save_viz = spatial_dims in [2, 3] and not skip_if_no_validation

    def save_visualization(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        epoch: int,
    ) -> None:
        """
        Save visualization for first batch of validation.

        Args:
            images: Input images [B, C, ...]
            labels: Reference labels [B, 1, ...]
            predictions: Model predictions [B, 1, ...]
            epoch: Current epoch number
        """
        if not self.save_viz:
            return

        # Save using existing visualization utility
        save_validation_visualizations(
            images=images,
            labels=labels,
            predictions=predictions,
            save_dir=self.results_dir,
            epoch=epoch,
            spatial_dims=self.spatial_dims,
        )


class TrainingLogger:
    """Logs training progress with loss, learning rate, and validation metrics."""

    def __init__(self, logger: Logger):
        """
        Args:
            logger: Loguru logger instance
        """
        self.logger = logger

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)

    def _log_epoch(self, engine: Engine) -> None:
        """Log single line per epoch with loss to file only."""
        current_epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs

        # Get epoch loss from output
        epoch_loss = engine.state.output.get("loss", None)  # type: ignore[union-attr]
        loss_str = f"loss={epoch_loss:.4f}" if epoch_loss is not None else "loss=?"

        # Get learning rate if available
        lr_str = ""
        if hasattr(engine.state, "optimizer"):
            current_lr = engine.state.optimizer.param_groups[0]["lr"]  # type: ignore[attr-defined]
            lr_str = f", lr={current_lr:.6f}"

        # Build message with loss and learning rate
        msg = f"Epoch {current_epoch}/{max_epochs}: {loss_str}{lr_str}"

        # Log to file only (no console output)
        log_only(self.logger, msg)

    def log_validation_metrics(
        self, epoch: int, max_epochs: int, metrics: dict[str, float]
    ) -> None:
        """
        Log validation metrics to file.

        Args:
            epoch: Current epoch number
            max_epochs: Total number of epochs
            metrics: Dictionary of metric names and values
        """
        # Separate mean metrics from per-class metrics
        mean_metrics = []
        class_metrics: dict[str, dict[int, float]] = {}

        for key, value in metrics.items():
            if key.startswith("val_"):
                if "_class" in key:
                    # Extract metric name and class index
                    parts = key.replace("val_", "").split("_class")
                    metric_name = parts[0]
                    class_idx = int(parts[1])

                    if metric_name not in class_metrics:
                        class_metrics[metric_name] = {}
                    class_metrics[metric_name][class_idx] = value
                else:
                    metric_name = key.replace("val_", "")
                    mean_metrics.append(f"{metric_name}={value:.4f}")

        # Build message
        msg = f"Epoch {epoch}/{max_epochs} validation:"
        if mean_metrics:
            msg += " " + ", ".join(mean_metrics)

        # Add per-class metrics
        for metric_name in sorted(class_metrics.keys()):
            classes = class_metrics[metric_name]
            class_strs = [f"c{idx}={val:.4f}" for idx, val in sorted(classes.items())]
            msg += f", {metric_name}_per_class=[" + ",".join(class_strs) + "]"

        # Log to file only
        log_only(self.logger, msg)
