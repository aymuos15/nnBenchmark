"""
Custom Lightning callbacks for nnBenchmark.
Maintains exact output formats (training_history.json, visualizations, logs).
"""

from pathlib import Path
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from src.logging import log_gpu_memory
from src.plotting.validation import save_validation_visualizations
from src.utils.files import save_json


class TrainingHistoryCallback(Callback):
    """
    Records training/validation metrics to training_history.json.
    Maintains exact same format as manual implementation for plot.py compatibility.
    Supports training with or without validation (e.g., when fold=-1).
    """

    def __init__(self, results_dir: str, training_all_data: bool = False):
        """
        Args:
            results_dir: Directory to save training_history.json
            training_all_data: If True, skip validation metrics (training on all data)
        """
        super().__init__()
        self.results_dir = results_dir
        self.history_path = str(Path(results_dir) / "training_history.json")
        self.training_all_data = training_all_data

        # Load existing history if resuming, otherwise start fresh
        if Path(self.history_path).exists():
            import json

            with open(self.history_path, "r") as f:
                self.training_history = json.load(f)
        else:
            # Initialize history structure (matches original format)
            self.training_history: dict[str, Any] = {
                "epochs": [],
                "train_loss": [],
                "val_epochs": [],
            }

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record training loss after each epoch."""
        current_epoch = trainer.current_epoch + 1

        # Get train_loss from logged metrics
        train_loss = trainer.callback_metrics.get("train_loss_epoch", None)

        if train_loss is not None:
            # Avoid duplicates when resuming
            if current_epoch not in self.training_history["epochs"]:
                self.training_history["epochs"].append(current_epoch)
                self.training_history["train_loss"].append(train_loss.item())
                # Save immediately to persist on-the-fly
                save_json(self.training_history, self.history_path)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Record validation metrics after validation completes.

        IMPORTANT: Must use on_validation_end (not on_validation_epoch_end) because
        validation metrics are logged in LightningModule.on_validation_epoch_end(),
        which runs AFTER callbacks' on_validation_epoch_end(). Using on_validation_end
        ensures metrics are available in trainer.callback_metrics.

        Skips recording when training_all_data is True (no validation phase).
        """
        # Skip if training on all data (no validation)
        if self.training_all_data:
            return

        # Skip sanity checking phase
        from pytorch_lightning.trainer.states import RunningStage

        if trainer.state.stage == RunningStage.SANITY_CHECKING:
            return

        current_epoch = trainer.current_epoch + 1

        # Only record if this is a new validation epoch (avoid duplicates)
        if current_epoch in self.training_history["val_epochs"]:
            return  # Already recorded, skip

        # Record validation epoch
        self.training_history["val_epochs"].append(current_epoch)

        # Record all validation metrics (including per-class)
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val_"):
                # Initialize metric list if needed
                if key not in self.training_history:
                    self.training_history[key] = []

                # Append metric value
                self.training_history[key].append(value.item())

        # Save after each validation (for resumption and monitoring)
        save_json(self.training_history, self.history_path)


class ValidationVisualizationCallback(Callback):
    """
    Saves validation visualizations using existing viz utilities.
    Creates PNG files in visualizations/ directory.
    Skips visualization when training without validation (fold=-1).
    """

    def __init__(
        self, results_dir: str, cfg: dict, skip_if_no_validation: bool = False
    ):
        """
        Args:
            results_dir: Directory to save visualizations
            cfg: Configuration dictionary
            skip_if_no_validation: If True, skip visualization when there's no validation
        """
        super().__init__()
        self.results_dir = results_dir
        self.skip_if_no_validation = skip_if_no_validation
        self.spatial_dims = cfg["model"]["spatial_dims"]
        self.save_viz = self.spatial_dims in [2, 3] and not skip_if_no_validation

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Save visualization for first batch of first validation.

        Args:
            outputs: Dictionary from validation_step with images/labels/predictions
        """
        # Only save first batch
        if batch_idx != 0:
            return

        # Only save if 2D or 3D data
        if not self.save_viz:
            return

        # Extract tensors from validation_step output
        images = outputs["images"]
        labels = outputs["labels"]
        predictions = outputs["predictions"]

        # Save using existing visualization utility
        current_epoch = trainer.current_epoch + 1
        save_validation_visualizations(
            images=images,
            labels=labels,
            predictions=predictions,
            save_dir=self.results_dir,
            epoch=current_epoch,
            spatial_dims=self.spatial_dims,
        )


class TrainingStepLogger(Callback):
    """Logs training steps with loss and optional GPU memory."""

    def __init__(self, logger: Any, log_gpu_mem: bool = True):
        """
        Args:
            logger: Loguru logger instance
            log_gpu_mem: Whether to log GPU memory per step
        """
        super().__init__()
        self.logger = logger
        self.log_gpu_mem = log_gpu_mem

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log single line per epoch with loss and validation metrics to file only."""
        from src.logging import log_only

        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs

        # Get epoch loss from logged metrics
        epoch_loss = trainer.callback_metrics.get("train_loss", None)
        loss_str = f"loss={epoch_loss:.4f}" if epoch_loss is not None else "loss=?"

        # Get learning rate from optimizer
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        lr_str = f"lr={current_lr:.6f}"

        # Build message with loss and learning rate
        msg = f"Epoch {current_epoch}/{max_epochs}: {loss_str}, {lr_str}"

        # Separate mean metrics from per-class metrics
        mean_metrics = []
        class_metrics = {}

        for key, value in trainer.callback_metrics.items():
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

        # Add mean metrics to message
        if mean_metrics:
            msg += ", " + ", ".join(mean_metrics)

        # Add per-class metrics
        for metric_name in sorted(class_metrics.keys()):
            classes = class_metrics[metric_name]
            class_strs = [f"c{idx}={val:.4f}" for idx, val in sorted(classes.items())]
            msg += f", {metric_name}_per_class=[" + ",".join(class_strs) + "]"

        # Log to file only (no console output)
        log_only(self.logger, msg)


class GPUMemoryCallback(Callback):
    """
    Logs GPU memory usage at key training points.
    Uses existing log_gpu_memory() utility from logging module.
    """

    def __init__(self, logger: Any):
        """
        Args:
            logger: Loguru logger instance
        """
        super().__init__()
        self.logger = logger

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log GPU memory at start of training epoch."""
        current_epoch = trainer.current_epoch + 1
        device = pl_module.device

        log_gpu_memory(
            self.logger,
            f"Epoch {current_epoch} Start",
            device,
            reset_peak=True,
        )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log GPU memory before validation."""
        current_epoch = trainer.current_epoch + 1
        device = pl_module.device

        log_gpu_memory(
            self.logger,
            f"Before Validation (Epoch {current_epoch})",
            device,
        )
