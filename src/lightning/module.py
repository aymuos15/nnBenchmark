"""
LightningModule wrapper for MONAI segmentation models.
Handles training/validation steps, optimizer configuration, and metric computation.
Supports deep supervision for improved gradient flow during training.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.utils import one_hot
from pytorch_lightning import LightningModule

from src.config.validation import validate_metrics_config
from src.lightning.lr_scheduler import PolyLRScheduler
from src.utils.builders import build_loss, build_metrics, build_model, build_optimizer


class SegmentationModule(LightningModule):
    cfg: dict[str, Any]
    checkpoint_metric: str
    plot_metrics: list[str]
    """
    Lightning wrapper for segmentation models.

    Integrates existing builders (model, loss, optimizer, metrics) with Lightning's
    training infrastructure while maintaining compatibility with MONAI components.
    """

    def __init__(self, cfg: dict, device: torch.device):
        """
        Initialize the Lightning module.

        Args:
            cfg: Configuration dictionary (from YAML)
            device: Device to use (cuda/cpu)
        """
        super().__init__()
        object.__setattr__(self, "cfg", cfg)

        # Save hyperparameters (enables checkpoint resumption)
        self.save_hyperparameters(cfg)

        # Build components using existing builders
        self.model: nn.Module = build_model(cfg, device)
        self.loss_fn: nn.Module = build_loss(cfg)
        self.metric_fns: dict[str, Any] = build_metrics(cfg)

        # Validate metrics and get checkpoint metric
        checkpoint_metric, plot_metrics = validate_metrics_config(cfg, self.metric_fns)
        object.__setattr__(self, "checkpoint_metric", checkpoint_metric)
        object.__setattr__(self, "plot_metrics", plot_metrics)

        # Store number of classes for one-hot encoding in validation
        self.num_classes: int = cfg["dataset"]["num_classes"]

        # Deep supervision configuration
        model_cfg = cfg.get("model", {})
        self.deep_supervision: bool = model_cfg.get("deep_supervision", False)
        self.ds_weights: list[float] = model_cfg.get("ds_weights", [])

        # Validate deep supervision config if enabled
        if self.deep_supervision and not self.ds_weights:
            raise ValueError(
                "deep_supervision is enabled but ds_weights is not specified. "
                "Please provide ds_weights in model config."
            )

        # Track best validation dice for console logging
        self.best_val_dice: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _compute_deep_supervision_loss(
        self, outputs: list[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss across multiple decoder outputs (deep supervision).

        Args:
            outputs: List of model outputs from different decoder levels
            labels: Target labels

        Returns:
            Weighted sum of losses
        """
        if len(outputs) != len(self.ds_weights):
            raise ValueError(
                f"Number of outputs ({len(outputs)}) doesn't match "
                f"number of weights ({len(self.ds_weights)})"
            )

        target_size = labels.shape[2:]  # Get spatial dimensions (H, W) or (H, W, D)
        total_loss = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)

        for output, weight in zip(outputs, self.ds_weights):
            # Upsample output to match target size if needed
            if output.shape[2:] != target_size:
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

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step - compute loss and log metrics.

        Supports deep supervision: computes loss at multiple decoder levels
        when enabled, otherwise uses single final output.

        Args:
            batch: Dictionary with 'image' and 'label' keys
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        inputs = batch["image"]
        labels = batch["label"]

        # Forward pass
        outputs = self(inputs)

        # Handle deep supervision output format
        # DynUNet with deep_supervision=True outputs shape:
        #   2D: [B, num_outputs, C, H, W] (5D)
        #   3D: [B, num_outputs, C, H, W, D] (6D)
        # We need to extract each output and compute weighted loss
        if self.deep_supervision and len(outputs.shape) in (5, 6):
            # DynUNet deep supervision format
            # Split into list of outputs for loss computation
            outputs_list = [outputs[:, i, ...] for i in range(outputs.shape[1])]
            loss = self._compute_deep_supervision_loss(outputs_list, labels)
        elif self.deep_supervision and isinstance(outputs, list):
            # List format (alternative deep supervision format)
            loss = self._compute_deep_supervision_loss(outputs, labels)
        else:
            # No deep supervision or single output
            loss = self.loss_fn(outputs, labels)

        # Log training loss (to file and console progress bar)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log learning rate to console progress bar
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Validation step - compute metrics and return outputs for callbacks.

        Uses only final output for metric computation, ignoring auxiliary
        deep supervision outputs (those are for training only).

        Args:
            batch: Dictionary with 'image' and 'label' keys
            batch_idx: Batch index

        Returns:
            Dictionary with images, labels, predictions for visualization
        """
        inputs = batch["image"]
        labels = batch["label"]

        # Forward pass
        outputs = self(inputs)

        # Handle deep supervision output format
        # DynUNet with deep_supervision=True outputs shape:
        #   2D: [B, num_outputs, C, H, W] (5D)
        #   3D: [B, num_outputs, C, H, W, D] (6D)
        if self.deep_supervision and len(outputs.shape) in (5, 6):
            # DynUNet format: use first output (final prediction)
            final_output = outputs[:, 0, ...]  # First output is final prediction
        elif self.deep_supervision and isinstance(outputs, list):
            # List format: use last output
            final_output = outputs[-1]
        else:
            # No deep supervision
            final_output = outputs

        # Get predictions (argmax for multi-class)
        preds = torch.argmax(final_output, dim=1, keepdim=True)

        # Convert predictions and labels to one-hot format for metric calculation
        # Metrics expect: [B, C, H, W(, D)] where C is number of classes
        preds_one_hot = one_hot(preds, num_classes=self.num_classes)
        labels_one_hot = one_hot(labels, num_classes=self.num_classes)

        # Accumulate metrics (they handle batching internally)
        for metric_fn in self.metric_fns.values():
            metric_fn(preds_one_hot, labels_one_hot)

        # Return data for visualization callback
        return {
            "images": inputs,
            "labels": labels,
            "predictions": preds,
            "outputs": final_output,
        }

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log accumulated metrics at end of validation epoch.
        Called automatically by Lightning after all validation_step calls.
        Logs both mean and per-class metrics to file. Tracks and logs best val_dice to console.
        """
        # Track if we found dice metric
        dice_found = False

        # Compute final metrics from accumulated values
        for name, metric_fn in self.metric_fns.items():
            # Aggregate accumulated metric
            result = metric_fn.aggregate()

            # Extract mean value (handle both scalar and per-class results)
            if hasattr(result, "mean"):
                mean_val = result.mean().item()
                per_class_vals = result
            else:
                mean_val = result.item()
                per_class_vals = None

            # Log mean to file only (no console progress bar) except for dice
            prog_bar_val = name == "Dice"  # Show dice on console
            self.log(
                f"val_{name}",
                mean_val,
                prog_bar=prog_bar_val,
                sync_dist=True,
            )

            # Track dice metric for console logging
            if name == "Dice":
                dice_found = True
                # Update best val_dice if this is a new best
                if mean_val > self.best_val_dice:
                    self.best_val_dice = mean_val

            # Log per-class metrics if available
            if per_class_vals is not None:
                for class_idx in range(per_class_vals.shape[0]):
                    class_val = per_class_vals[class_idx].item()
                    self.log(
                        f"val_{name}_class{class_idx}",
                        class_val,
                        prog_bar=False,
                        sync_dist=True,
                    )

            # Reset metric for next epoch
            metric_fn.reset()

        # Log best val_dice to console progress bar (only if dice metric exists)
        if dice_found:
            self.log(
                "best_val_dice",
                self.best_val_dice,
                prog_bar=True,
                sync_dist=False,
            )

    def on_before_optimizer_step(self, optimizer) -> None:  # type: ignore[override]
        """
        Gradient clipping before optimizer step.

        Matches nnU-Net v2.4.1 behavior with max_norm=12.
        Called automatically by PyTorch Lightning before each optimizer step.

        Args:
            optimizer: The optimizer being used
        """
        # Clip gradients to max norm of 12 (matches nnU-Net)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12)

    def configure_optimizers(self):  # type: ignore[override]
        """
        Configure optimizer and learning rate scheduler.

        Uses linear learning rate decay by default (0.00001 reduction per epoch).
        Can be overridden in config with 'lr_scheduler' settings.

        Returns:
            Dictionary with 'optimizer' and 'lr_scheduler' for PyTorch Lightning.
            This return format is valid per PyTorch Lightning documentation, but pyright
            doesn't recognize the dict return type as compatible with the base class signature.
        """
        optimizer = build_optimizer(self.model, self.cfg)

        # Get LR scheduler config (allows overriding defaults)
        lr_config = self.cfg.get("lr_scheduler", {})
        mode = lr_config.get("mode", "linear")
        decay_rate = lr_config.get("decay_rate", 0.00001)
        exponent = lr_config.get("exponent", 0.9)

        lr_scheduler = PolyLRScheduler(
            optimizer,
            initial_lr=self.cfg["training"]["learning_rate"],
            max_epochs=self.cfg["training"]["epochs"],
            exponent=exponent,
            mode=mode,
            decay_rate=decay_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",  # Step after each epoch
                "frequency": 1,  # Step every epoch
            },
        }
