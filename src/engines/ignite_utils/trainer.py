"""
MONAI SupervisedTrainer wrapper for nnBenchmark.
Handles training loop, validation, checkpointing, and metric computation.
"""


from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from monai.bundle import ConfigParser
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers.lr_schedule_handler import LrScheduleHandler

from src.engines.ignite_utils.progress import ConsoleProgressHandler
from src.engines.setup import _instantiate_component, build_model
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
        expected_ds_ndim = 3 + self.spatial_dims

        if isinstance(outputs, torch.Tensor) and outputs.ndim == expected_ds_ndim:
            outputs_list = [outputs[:, i, ...] for i in range(outputs.shape[1])]
        elif isinstance(outputs, list):
            outputs_list = outputs
        else:
            return self.loss_fn(outputs, labels)

        if len(outputs_list) != len(self.ds_weights):
            raise ValueError(
                f"Number of outputs ({len(outputs_list)}) doesn't match "
                f"number of weights ({len(self.ds_weights)})"
            )

        import torch.nn.functional as F

        total_loss = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)

        for output, weight in zip(outputs_list, self.ds_weights):
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
    cfg: ConfigParser,
    device: torch.device,
    train_loader: DataLoader,
    results_dir: str,
    logger: "Logger",
) -> tuple[
    SupervisedTrainer,
    torch.optim.Optimizer,
    Any,  # LR scheduler
    torch.amp.GradScaler,
]:  # type: ignore[return]
    """
    Create MONAI SupervisedTrainer.

    Args:
        cfg: ConfigParser instance
        device: Device to use
        train_loader: Training data loader
        results_dir: Directory to save results
        logger: Loguru logger instance

    Returns:
        Tuple of (trainer, optimizer, lr_scheduler, scaler).
    """
    # Build model via _target_
    model = build_model(cfg, device)

    # Build optimizer via _target_
    learning_rate = cfg["training"]["learning_rate"]
    opt_cfg = dict(cfg["optimizer"])
    opt_cfg["params"] = model.parameters()
    opt_cfg["lr"] = learning_rate
    optimizer = _instantiate_component(opt_cfg)

    # Build loss via _target_
    loss_fn_base = _instantiate_component(dict(cfg["loss"]))

    # Wrap loss for deep supervision if enabled
    model_cfg = dict(cfg["model"])
    training_cfg = dict(cfg["training"])
    deep_supervision = model_cfg.get("deep_supervision", False)
    if deep_supervision:
        ds_weights = training_cfg.get("ds_weights") or model_cfg.get("ds_weights", [])
        if not ds_weights:
            raise ValueError(
                "Deep supervision is enabled but 'ds_weights' is not configured or is empty. "
                "Please provide 'ds_weights' in the training configuration."
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
    mode = lr_config.get("mode", "polynomial") if lr_config else "polynomial"
    decay_rate = lr_config.get("decay_rate", 0.00001) if lr_config else 0.00001
    exponent = lr_config.get("exponent", 0.9) if lr_config else 0.9

    lr_scheduler = PolyLRScheduler(
        optimizer,
        initial_lr=learning_rate,
        max_epochs=max_epochs,
        exponent=exponent,
        mode=mode,
        decay_rate=decay_rate,
    )

    # Create GradScaler for AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Custom iteration update function with AMP and gradient clipping
    def iteration_update(engine: Engine, batch: Any) -> dict[str, Any]:
        """Custom iteration update with AMP GradScaler and gradient clipping."""
        model.train()

        images, labels = _prepare_batch(batch, device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
        scaler.step(optimizer)
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

    # Add learning rate scheduler handler — step at epoch START to match nnU-Net
    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        LrScheduleHandler(lr_scheduler=lr_scheduler),  # type: ignore[arg-type]
    )

    # Add GPU cache clearing after each epoch
    def clear_gpu_cache(engine: Engine) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, clear_gpu_cache)

    # Add training history handler
    history_handler = TrainingHistoryHandler(results_dir)
    history_handler.attach(trainer)

    # Add console progress handler
    progress_handler = ConsoleProgressHandler(logger, max_epochs)
    progress_handler.attach(trainer)

    # Add training logger
    training_logger = TrainingLogger(logger)
    training_logger.attach(trainer)

    # Add checkpoint saver
    checkpoint_dir = Path(results_dir)
    checkpoint_handler = ComprehensiveCheckpointHandler(
        save_dir=str(checkpoint_dir),
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        cfg=cfg,
        save_interval=1,
        checkpoint_metric="loss",
        evaluator=None,
    )
    checkpoint_handler.attach(trainer)

    return trainer, optimizer, lr_scheduler, scaler  # type: ignore[return-value]
