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
from monai import losses as monai_losses
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers.lr_schedule_handler import LrScheduleHandler
from monai.networks import nets as monai_nets

from src.models.dynunet import NativeDSDynUNet

monai_nets.NativeDSDynUNet = NativeDSDynUNet  # type: ignore[attr-defined]

from src.engines.ignite_utils.progress import ConsoleProgressHandler
from src.engines.shared import safe_getattr
from src.engines.train.handlers import (
    ComprehensiveCheckpointHandler,
    TrainingHistoryHandler,
    TrainingLogger,
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

        import torch.nn.functional as F

        total_loss = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)

        for output, weight in zip(outputs_list, self.ds_weights):
            # Downsample labels to match output resolution (nnU-Net style)
            # This preserves gradient flow at native decoder resolution
            if output.shape[2:] != labels.shape[2:]:
                labels_down = F.interpolate(
                    labels.float(),
                    size=output.shape[2:],
                    mode="nearest",
                ).to(labels.dtype)
            else:
                labels_down = labels

            level_loss = self.loss_fn(output, labels_down)
            total_loss = total_loss + weight * level_loss

        return total_loss


def create_trainer(
    cfg: dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    results_dir: str,
    logger: Logger,
) -> tuple[
    SupervisedTrainer,
    torch.optim.Optimizer,
    Any,  # LR scheduler
    torch.amp.GradScaler,
]:  # type: ignore[return]
    """
    Create MONAI SupervisedTrainer.
    Logs always append to existing files.
    Validation is now performed post-training via nnBench.validate.

    Args:
        cfg: Configuration dictionary
        device: Device to use
        train_loader: Training data loader
        results_dir: Directory to save results
        logger: Loguru logger instance

    Returns:
        Tuple of (trainer, optimizer, lr_scheduler, scaler).
    """
    # Build model via getattr (supports any MONAI model)
    model_cfg = cfg["model"].copy()
    model_type = model_cfg.pop("type")
    # Remove training-only parameters that shouldn't be passed to model constructor
    model_cfg.pop("ds_weights", None)  # Used by DeepSupervisionLossWrapper, not model
    # deep_supervision only supported by DynUNet and BasicUNetPlusPlus
    if model_type not in ("DynUNet", "NativeDSDynUNet", "BasicUNetPlusPlus"):
        model_cfg.pop("deep_supervision", None)
        model_cfg.pop("deep_supr_num", None)
    # Merge model-specific parameters
    # NativeDSDynUNet uses DynUNet params from the config
    params_key = "DynUNet" if model_type == "NativeDSDynUNet" else model_type
    dynunet_params = model_cfg.pop("DynUNet", None)
    model_cfg.pop("UNet", None)
    if params_key == "DynUNet" and dynunet_params:
        model_cfg.update(dynunet_params)
    elif params_key in cfg["model"] and isinstance(cfg["model"][params_key], dict):
        model_cfg.update(cfg["model"][params_key])
    model_class = safe_getattr(monai_nets, model_type, "monai.networks.nets")
    model = model_class(**model_cfg).to(device)

    # Build optimizer via getattr (supports any PyTorch optimizer)
    learning_rate = cfg["training"]["learning_rate"]
    opt_cfg = cfg["optimizer"].copy()
    opt_type = opt_cfg.pop("type")
    opt_class = safe_getattr(torch.optim, opt_type, "torch.optim")
    optimizer = opt_class(model.parameters(), lr=learning_rate, **opt_cfg)

    # Build loss function via getattr (supports any MONAI loss)
    loss_cfg = cfg["loss"].copy()
    loss_type = loss_cfg.pop("type")
    loss_class = safe_getattr(monai_losses, loss_type, "monai.losses")
    loss_fn_base = loss_class(**loss_cfg)

    # Wrap loss function for deep supervision if enabled
    model_cfg = cfg.get("model", {})
    deep_supervision = model_cfg.get("deep_supervision", False)
    if deep_supervision:
        ds_weights = model_cfg.get("ds_weights", [])
        if not ds_weights:
            raise ValueError(
                "Deep supervision is enabled but 'ds_weights' is not configured or is empty. "
                "Please provide 'ds_weights' list in the model configuration (e.g., ds_weights: [1.0, 0.5, 0.25, 0.125])"
            )
        spatial_dims = model_cfg.get("spatial_dims", 3)
        loss_fn = DeepSupervisionLossWrapper(loss_fn_base, ds_weights, spatial_dims)
    else:
        loss_fn = loss_fn_base

    # Get training parameters
    max_epochs = cfg["training"]["epochs"]
    use_amp = cfg.get("training", {}).get("mixed_precision", False)

    # Create learning rate scheduler
    lr_config = cfg.get("lr_scheduler", {})
    mode = lr_config.get("mode", "polynomial")
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

    # Add training history handler (always appends to existing history)
    history_handler = TrainingHistoryHandler(results_dir)
    history_handler.attach(trainer)

    # Add console progress handler
    progress_handler = ConsoleProgressHandler(logger, max_epochs)
    progress_handler.attach(trainer)

    # Add training logger
    training_logger = TrainingLogger(logger)
    training_logger.attach(trainer)

    # Add checkpoint saver based on training loss
    checkpoint_dir = Path(results_dir)
    checkpoint_handler = ComprehensiveCheckpointHandler(
        save_dir=str(checkpoint_dir),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        cfg=cfg,
        save_interval=1,
        checkpoint_metric="loss",  # Use training loss for best model
        evaluator=None,  # No validation during training
    )
    checkpoint_handler.attach(trainer)

    return trainer, optimizer, lr_scheduler, scaler  # type: ignore[return-value]
