"""
MONAI SupervisedTrainer wrapper for nnBenchmark.
Handles training loop, validation, checkpointing, and metric computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from monai.data import DataLoader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers.lr_schedule_handler import LrScheduleHandler
from monai.networks.utils import one_hot

from src.engines.ignite_utils.progress import (
    ConsoleProgressHandler,
    ValidationProgressHandler,
)
from src.engines.train.handlers import (
    ComprehensiveCheckpointHandler,
    TrainingHistoryHandler,
    TrainingLogger,
    ValidationVisualizationHandler,
)
from src.factory import (
    loss_registry,
    metric_registry,
    model_registry,
    optimizer_registry,
)
from src.utils.lr_scheduler import PolyLRScheduler

if TYPE_CHECKING:
    from loguru._logger import Logger


def _prepare_batch(
    batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare batch for training/validation.

    Args:
        batch: Dictionary with 'image' and 'label' keys
        device: Device to move tensors to
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Tuple of (images, labels)
    """
    images = batch["image"].to(device, non_blocking=non_blocking)
    labels = batch["label"].to(device, non_blocking=non_blocking)
    return images, labels


class DeepSupervisionLossWrapper(nn.Module):
    """
    Wrapper for computing deep supervision loss.
    Handles multiple decoder outputs with weighted loss computation.
    """

    def __init__(self, loss_fn: nn.Module, ds_weights: list[float], spatial_dims: int):
        """
        Args:
            loss_fn: Base loss function
            ds_weights: Weights for each decoder level
            spatial_dims: Number of spatial dimensions (2 or 3)
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.ds_weights = ds_weights  # type: ignore[misc]
        self.spatial_dims = spatial_dims  # type: ignore[misc]

    def forward(
        self, outputs: torch.Tensor | list[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss across multiple decoder outputs.

        Args:
            outputs: Model outputs (either list or tensor with extra dimension)
            labels: Target labels

        Returns:
            Weighted sum of losses
        """
        # Handle deep supervision output format
        # DynUNet with deep_supervision=True outputs shape:
        #   2D: [B, num_outputs, C, H, W] (5D) vs [B, C, H, W] (4D) without
        #   3D: [B, num_outputs, C, D, H, W] (6D) vs [B, C, D, H, W] (5D) without
        expected_ds_ndim = 3 + self.spatial_dims

        if isinstance(outputs, torch.Tensor) and outputs.ndim == expected_ds_ndim:
            # DynUNet deep supervision format - split into list
            outputs_list = [outputs[:, i, ...] for i in range(outputs.shape[1])]
        elif isinstance(outputs, list):
            # Already in list format
            outputs_list = outputs
        else:
            # No deep supervision - single output
            return self.loss_fn(outputs, labels)

        # Validate number of outputs matches weights
        if len(outputs_list) != len(self.ds_weights):
            raise ValueError(
                f"Number of outputs ({len(outputs_list)}) doesn't match "
                f"number of weights ({len(self.ds_weights)})"
            )

        target_size = labels.shape[2:]  # Get spatial dimensions
        total_loss = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)

        for output, weight in zip(outputs_list, self.ds_weights):
            # Upsample output to match target size if needed
            if output.shape[2:] != target_size:
                import torch.nn.functional as F

                output_up = F.interpolate(
                    output,
                    size=target_size,
                    mode="trilinear" if len(target_size) == 3 else "bilinear",
                    align_corners=False,
                )
            else:
                output_up = output

            # Compute loss for this level and add weighted to total
            level_loss = self.loss_fn(output_up, labels)
            total_loss = total_loss + weight * level_loss

        return total_loss


def create_trainer(
    cfg: dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    results_dir: str,
    logger: Logger,
    resume: bool = False,
) -> tuple[
    SupervisedTrainer,
    SupervisedEvaluator | None,
    torch.optim.Optimizer,
    Any,  # LR scheduler
    torch.amp.GradScaler,
]:  # type: ignore[return]
    """
    Create MONAI SupervisedTrainer and Evaluator.

    Args:
        cfg: Configuration dictionary
        device: Device to use
        train_loader: Training data loader
        val_loader: Validation data loader (None if training on all data)
        results_dir: Directory to save results
        logger: Loguru logger instance
        resume: Whether to resume from checkpoint

    Returns:
        Tuple of (trainer, evaluator). Evaluator is None if val_loader is None.
    """
    # Build model
    model = model_registry.build(cfg["model"], device)

    # Build optimizer
    learning_rate = cfg["training"]["learning_rate"]
    optimizer = optimizer_registry.build(
        cfg["optimizer"], model.parameters(), learning_rate
    )

    # Build loss function
    loss_fn_base = loss_registry.build(cfg["loss"])

    # Wrap loss function for deep supervision if enabled
    model_cfg = cfg.get("model", {})
    deep_supervision = model_cfg.get("deep_supervision", False)
    if deep_supervision:
        ds_weights = model_cfg.get("ds_weights", [])
        spatial_dims = model_cfg.get("spatial_dims", 3)
        loss_fn = DeepSupervisionLossWrapper(loss_fn_base, ds_weights, spatial_dims)
    else:
        loss_fn = loss_fn_base

    # Get training parameters
    max_epochs = cfg["training"]["epochs"]
    use_amp = cfg.get("training", {}).get("mixed_precision", False)
    training_all_data = val_loader is None

    # Create learning rate scheduler
    lr_config = cfg.get("lr_scheduler", {})
    mode = lr_config.get("mode", "linear")
    decay_rate = lr_config.get("decay_rate", 0.00001)
    exponent = lr_config.get("exponent", 0.9)

    lr_scheduler = PolyLRScheduler(
        optimizer,
        initial_lr=learning_rate,
        max_epochs=max_epochs,
        exponent=exponent,
        mode=mode,
        decay_rate=decay_rate,
    )

    # Create GradScaler for AMP (automatic mixed precision)
    # GradScaler is essential for stable FP16 training - it scales losses to prevent
    # gradient underflow and ensures proper gradient clipping
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Custom iteration update function with AMP and gradient clipping
    def iteration_update(engine: Engine, batch: Any) -> dict[str, Any]:
        """Custom iteration update with AMP GradScaler and gradient clipping."""
        model.train()

        # Prepare batch
        images, labels = _prepare_batch(batch, device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        # Backward pass with gradient scaling for AMP
        # scaler.scale() multiplies loss by scale factor before backward()
        # This prevents FP16 gradient underflow
        scaler.scale(loss).backward()

        # Unscale gradients before clipping (required for accurate clipping)
        # This divides gradients by the scale factor
        scaler.unscale_(optimizer)

        # Gradient clipping (matches nnU-Net: max_norm=12)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)

        # Optimizer step with scaling
        # scaler.step() first checks for inf/NaN gradients
        # If gradients are finite, unscales and calls optimizer.step()
        # If gradients are inf/NaN, skips the update
        scaler.step(optimizer)

        # Update the scale factor for next iteration
        # Increases scale if no inf/NaN, decreases if inf/NaN found
        scaler.update()

        return {"loss": loss.item()}

    # Create trainer
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        iteration_update=iteration_update,
        amp=use_amp,
    )

    # Add learning rate scheduler handler
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        LrScheduleHandler(lr_scheduler=lr_scheduler),  # type: ignore[arg-type]
    )

    # Add GPU cache clearing after each epoch to prevent fragmentation
    # This is especially important for small GPUs (e.g., 4GB RTX A1000)
    def clear_gpu_cache(engine: Engine) -> None:
        """Clear CUDA cache to prevent memory fragmentation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, clear_gpu_cache)

    # Add training history handler
    history_handler = TrainingHistoryHandler(
        results_dir, training_all_data=training_all_data, resume=resume
    )
    history_handler.attach(trainer)

    # Add console progress handler
    progress_handler = ConsoleProgressHandler(logger, max_epochs)
    progress_handler.attach(trainer)

    # Add training logger
    training_logger = TrainingLogger(logger)
    training_logger.attach(trainer)

    # Add checkpoint saver
    checkpoint_dir = Path(results_dir)

    # Create evaluator for validation if validation data exists
    evaluator = None
    if val_loader is not None:
        # Build metrics
        metric_fns = metric_registry.build(cfg)
        num_classes = cfg["dataset"]["num_classes"]
        spatial_dims = model_cfg.get("spatial_dims", 3)

        # Validation post-processing function
        def validation_iteration(engine: Engine, batch: Any) -> dict[str, Any]:
            """Validation iteration with metric computation."""
            model.eval()

            with torch.no_grad():
                # Prepare batch
                images, labels = _prepare_batch(batch, device, non_blocking=True)

                # Ensure labels are integers and within valid range
                labels = torch.round(labels).long()
                labels = torch.clamp(labels, min=0, max=num_classes - 1)

                # Forward pass
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(images)

                # Handle deep supervision output format (use only final output)
                expected_ds_ndim = 3 + spatial_dims
                if (
                    deep_supervision
                    and isinstance(outputs, torch.Tensor)
                    and outputs.ndim == expected_ds_ndim
                ):
                    final_output = outputs[:, 0, ...]  # First output is final
                elif deep_supervision and isinstance(outputs, list):
                    final_output = outputs[-1]  # Last output
                else:
                    final_output = outputs

                # Get predictions (argmax for multi-class)
                preds = torch.argmax(final_output, dim=1, keepdim=True)

                # Convert to one-hot format for metrics
                preds_one_hot = one_hot(preds, num_classes=num_classes)
                labels_one_hot = one_hot(labels, num_classes=num_classes)

                # Accumulate metrics
                for metric_fn in metric_fns.values():
                    metric_fn(preds_one_hot, labels_one_hot)

                return {
                    "images": images,
                    "labels": labels,
                    "predictions": preds,
                    "outputs": final_output,
                }

        # Create evaluator
        evaluator = Engine(validation_iteration)

        # Initialize metrics dict in evaluator state (needed for CheckpointSaver)
        evaluator.state.metrics = {}

        # Add validation progress handler
        val_progress_handler = ValidationProgressHandler()
        val_progress_handler.attach(evaluator)

        # Add validation visualization handler
        viz_handler = ValidationVisualizationHandler(
            results_dir, spatial_dims, skip_if_no_validation=False
        )

        # Save visualization for first batch
        @evaluator.on(Events.ITERATION_COMPLETED(once=1))
        def save_viz(engine: Engine) -> None:
            """Save visualization for first batch."""
            output = engine.state.output
            current_epoch = trainer.state.epoch
            viz_handler.save_visualization(
                output["images"],  # type: ignore[index, call-overload]
                output["labels"],  # type: ignore[index, call-overload]
                output["predictions"],  # type: ignore[index, call-overload]
                current_epoch,
            )

        # Compute metrics at end of validation (before checkpoint saver)
        @evaluator.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine: Engine) -> None:
            """Compute and log metrics after validation."""
            current_epoch = trainer.state.epoch
            metrics_dict = {}

            # Compute final metrics from accumulated values
            for name, metric_fn in metric_fns.items():
                result = metric_fn.aggregate()

                # Extract mean value and per-class values
                if hasattr(result, "mean") and hasattr(result, "shape") and len(result.shape) > 0:
                    # Multi-dimensional result (per-class metrics)
                    mean_val = result.mean().item()
                    per_class_vals = result
                else:
                    # Scalar result (single metric value)
                    mean_val = result.item()
                    per_class_vals = None

                # Store mean metric
                metrics_dict[f"val_{name}"] = mean_val

                # Store per-class metrics if available
                if per_class_vals is not None:
                    for class_idx in range(per_class_vals.shape[0]):
                        class_val = per_class_vals[class_idx].item()
                        metrics_dict[f"val_{name}_class{class_idx}"] = class_val

                # Reset metric for next epoch
                metric_fn.reset()

            # Log metrics to history
            history_handler.record_validation_metrics(current_epoch, metrics_dict)

            # Log metrics to file
            training_logger.log_validation_metrics(
                current_epoch, max_epochs, metrics_dict
            )

            # Update console with validation metrics
            progress_handler.update_with_validation_metrics(metrics_dict)

            # Store metrics in evaluator state for checkpoint saver
            engine.state.metrics = metrics_dict

        # Run validation at specified intervals
        val_interval = cfg["training"].get("val_interval", 1)

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine: Engine) -> None:
            """Run validation at specified intervals."""
            current_epoch = engine.state.epoch
            if current_epoch % val_interval == 0:
                evaluator.run(val_loader)

        # Add comprehensive checkpoint saver based on validation metric
        from src.config.validation import validate_metrics_config

        checkpoint_metric, _ = validate_metrics_config(cfg, metric_fns)

        checkpoint_handler = ComprehensiveCheckpointHandler(
            save_dir=str(checkpoint_dir),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            cfg=cfg,
            save_interval=1,
            checkpoint_metric=f"val_{checkpoint_metric}",
            evaluator=evaluator,
        )
        checkpoint_handler.attach(trainer)

    else:
        # No validation - save comprehensive checkpoint every epoch
        checkpoint_handler = ComprehensiveCheckpointHandler(
            save_dir=str(checkpoint_dir),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            cfg=cfg,
            save_interval=1,
            checkpoint_metric=None,
            evaluator=None,
        )
        checkpoint_handler.attach(trainer)

    return trainer, evaluator, optimizer, lr_scheduler, scaler  # type: ignore[return-value]
