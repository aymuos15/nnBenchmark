"""Ignite-based validation engine for nnBenchmark.

Uses PyTorch Ignite Engine for event-driven validation with handlers,
similar to the inference engine but specifically for post-training validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from ignite.engine import Engine
from monai.networks.utils import one_hot

from src.engines.inference.strategy import InferenceStrategy, create_inferer

if TYPE_CHECKING:
    from pathlib import Path


class ValidationEngine:
    """Ignite-based validation engine.

    Wraps PyTorch Ignite Engine to provide event-driven validation
    with support for handlers, metrics, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: dict[str, Any],
        metric_fns: dict[str, Any],
        data_dir: str | Path | None = None,
    ):
        """Initialize ValidationEngine.

        Args:
            model: PyTorch model for validation
            device: Device to run validation on (cuda or cpu)
            cfg: Configuration dictionary
            metric_fns: Dictionary of metric functions {name: metric_fn}
            data_dir: Optional dataset directory for loading class labels
        """
        self.model = model
        self.device = device
        self.cfg = cfg
        self.metric_fns = metric_fns
        self.data_dir = data_dir

        # Get configuration parameters
        self.num_classes = cfg["dataset"]["num_classes"]
        self.spatial_dims = cfg["model"].get("spatial_dims", 3)
        self.use_amp = cfg.get("training", {}).get("mixed_precision", False)
        self.deep_supervision = cfg.get("model", {}).get("deep_supervision", False)

        # Create inference strategy (sliding window or full volume)
        self.inferer: InferenceStrategy = create_inferer(cfg)

        # Create Ignite engine with custom iteration function
        self.engine = Engine(self._validation_iteration)

        # Initialize metrics dict and results storage in state
        self.engine.state.metrics = {}
        self.engine.state.batch_outputs = []  # For visualization

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor], non_blocking: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for validation.

        Args:
            batch: Dictionary with 'image' and 'label' keys
            non_blocking: Whether to use non-blocking transfer

        Returns:
            Tuple of (images, labels)
        """
        images = batch["image"].to(self.device, non_blocking=non_blocking)
        labels = batch["label"].to(self.device, non_blocking=non_blocking)
        return images, labels

    def _validation_iteration(self, engine: Engine, batch: Any) -> dict[str, Any]:
        """Single validation iteration.

        This is called by Ignite Engine for each batch. Performs:
        1. Model inference (using inferer strategy)
        2. Metric computation
        3. Returns outputs for handlers

        Args:
            engine: Ignite Engine instance
            batch: Batch data dictionary

        Returns:
            Dictionary with 'images', 'labels', 'predictions', 'outputs'
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare batch
            images, labels = self._prepare_batch(batch, non_blocking=True)

            # Ensure labels are integers and within valid range
            labels = torch.round(labels).long()
            labels = torch.clamp(labels, min=0, max=self.num_classes - 1)

            # Forward pass using inference strategy
            outputs_raw = self.inferer.infer(
                self.model, images, self.device, use_amp=self.use_amp
            )
            # Inferer returns union type, but in practice it's always a Tensor or list
            outputs = outputs_raw  # type: ignore[assignment]

            # Handle deep supervision output format (use only final output)
            expected_ds_ndim = 3 + self.spatial_dims
            if (
                self.deep_supervision
                and isinstance(outputs, torch.Tensor)
                and outputs.ndim == expected_ds_ndim
            ):
                final_output = outputs[:, 0, ...]  # First output is final
            elif self.deep_supervision and isinstance(outputs, list):
                final_output = outputs[-1]  # type: ignore[index]  # Last output
            else:
                final_output = outputs

            # Ensure final_output is a tensor
            assert isinstance(
                final_output, torch.Tensor
            ), f"Expected Tensor, got {type(final_output)}"

            # Get predictions (argmax for multi-class)
            preds = torch.argmax(final_output, dim=1, keepdim=True)

            # Convert to one-hot format for metrics
            preds_one_hot = one_hot(preds, num_classes=self.num_classes)
            labels_one_hot = one_hot(labels, num_classes=self.num_classes)

            # Accumulate metrics
            for metric_fn in self.metric_fns.values():
                metric_fn(preds_one_hot, labels_one_hot)

            return {
                "images": images,
                "labels": labels,
                "predictions": preds,
                "outputs": final_output,
            }

    def run(self, data_loader: Any) -> None:
        """Run validation on data loader.

        Args:
            data_loader: DataLoader with validation data
        """
        self.engine.run(data_loader)
