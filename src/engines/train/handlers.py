"""
Custom MONAI handlers for nnBenchmark.
Maintains exact output formats (training_history.json, visualizations, logs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from ignite.engine import Engine, Events

from src.logging import log_only
from src.utils.files import save_json

if TYPE_CHECKING:
    from loguru._logger import Logger


class TrainingHistoryHandler:
    """
    Records training metrics to training_history.json.
    Only tracks training loss (no validation metrics).
    """

    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: Directory to save training_history.json
        """
        self.results_dir = results_dir

        # Create history subdirectory
        history_dir = Path(results_dir) / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = str(history_dir / "training.json")

        # Always try to load existing history if file exists
        if Path(self.history_path).exists():
            with open(self.history_path) as f:
                self.training_history = json.load(f)
        else:
            # Initialize history structure
            self.training_history: dict[str, Any] = {
                "epochs": [],
                "train_loss": [],
            }

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        # Record training loss after each epoch
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._record_training_epoch)

    def _record_training_epoch(self, engine: Engine) -> None:
        """Record training loss after each epoch."""
        current_epoch = engine.state.epoch

        # Avoid duplicates when resuming
        if current_epoch in self.training_history["epochs"]:
            return

        # Get train_loss from engine state
        train_loss = engine.state.output.get("loss", None)  # type: ignore[union-attr]

        if train_loss is not None:
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


class TrainingLogger:
    """Logs training progress with loss and learning rate."""

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
        """Log epoch with loss and learning rate to file only."""
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


class ComprehensiveCheckpointHandler:
    """
    Saves comprehensive checkpoints with all training state.
    Includes model, optimizer, lr_scheduler, scaler, epoch, and config metadata.
    Tracks best model based on training loss (lower is better).
    """

    def __init__(
        self,
        save_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        scaler: torch.amp.GradScaler,
        cfg: dict[str, Any],
        save_interval: int = 1,
        checkpoint_metric: str | None = None,
        evaluator: Engine | None = None,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            lr_scheduler: Learning rate scheduler to checkpoint
            scaler: GradScaler to checkpoint
            cfg: Configuration dictionary (for metadata)
            save_interval: Save checkpoint every N epochs (default: 1)
            checkpoint_metric: 'loss' to track best based on training loss (ignored otherwise)
            evaluator: Ignored (kept for signature compatibility)
        """
        # Create checkpoints subdirectory
        self.save_dir = Path(save_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.cfg = cfg
        self.save_interval = save_interval
        self.checkpoint_metric = checkpoint_metric
        self.best_loss = float("inf")  # Lower loss is better
        self.trainer_engine = None  # Will be set in attach()

        # Create config metadata once
        self.config_metadata = {
            "dataset_name": cfg.get("dataset", {}).get("name", "unknown"),
            "fold": cfg.get("dataset", {}).get("fold", -1),
            "model_type": cfg.get("model", {}).get("type", "unknown"),
            "epochs": cfg.get("training", {}).get("epochs", 0),
        }

    def attach(self, trainer_engine: Engine) -> None:
        """Attach handler to trainer engine events."""
        # Store trainer engine reference for accessing epoch number
        self.trainer_engine = trainer_engine

        # Save checkpoint at specified intervals
        trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self._save_checkpoint)

    def _save_checkpoint(self, engine: Engine) -> None:
        """Save numbered checkpoint for each epoch and best model based on training loss."""
        current_epoch = engine.state.epoch

        # Only save at specified intervals
        if current_epoch % self.save_interval != 0:
            return

        # Save numbered checkpoint for this epoch
        checkpoint_path = self.save_dir / f"epoch_{current_epoch:03d}.pt"
        self._save(checkpoint_path, current_epoch, is_best=False)

        # Also save as final checkpoint (for easy access to latest)
        final_checkpoint_path = self.save_dir / "final.pt"
        self._save(final_checkpoint_path, current_epoch, is_best=False)

        # Check if we should save best model based on training loss
        if self.checkpoint_metric == "loss":
            # Get current loss from engine state
            current_loss = engine.state.output.get("loss", None)  # type: ignore[union-attr]

            if current_loss is not None:
                # Convert tensor to float if needed
                if isinstance(current_loss, torch.Tensor):
                    current_loss = current_loss.item()

                # Check if this is the best (lowest) loss so far
                if current_loss < self.best_loss:
                    self.best_loss = current_loss

                    # Save best checkpoint with loss value in filename
                    best_checkpoint_path = (
                        self.save_dir / f"best_loss={current_loss:.4f}.pt"
                    )

                    self._save(best_checkpoint_path, current_epoch, is_best=True)

                    # Remove old best checkpoints
                    self._cleanup_old_best_checkpoints(best_checkpoint_path)

    def _save(self, path: Path, epoch: int, is_best: bool = False) -> None:
        """
        Save comprehensive checkpoint to disk.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best checkpoint
        """
        # Create checkpoint dictionary with all training state
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "config_metadata": self.config_metadata,
        }

        # If this is a best checkpoint, also save the best loss value
        if is_best and self.checkpoint_metric == "loss":
            checkpoint["best_loss"] = self.best_loss

        # Save checkpoint
        torch.save(checkpoint, path)

    def _cleanup_old_best_checkpoints(self, current_best_path: Path) -> None:
        """Remove old best checkpoint files (keep only current best)."""
        # Clean up old loss-based checkpoints
        pattern = "best_loss*.pt"
        for old_checkpoint in self.save_dir.glob(pattern):
            if old_checkpoint != current_best_path:
                old_checkpoint.unlink()
