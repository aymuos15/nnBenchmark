"""Blob Loss implementation for instance segmentation.

This module implements a compound loss function that combines global segmentation loss
with per-instance (blob) loss computation. The blob loss computes loss for each
connected component separately, allowing for better handling of multi-instance segmentation tasks.

The implementation automatically converts binary labels to multi-instance labels using
GPU-accelerated connected components analysis.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai import losses

if TYPE_CHECKING:
    import cupy as cp  # type: ignore[import]
    from cucim.skimage import measure as cucim_measure  # type: ignore[import]
else:
    try:
        import cupy as cp
        from cucim.skimage import measure as cucim_measure
    except ImportError:
        cp = None  # type: ignore
        cucim_measure = None  # type: ignore

from src.factory.losses.cc import gpu_connected_components


class BlobLoss(nn.Module):
    """Blob Loss for multi-instance segmentation.

    This loss function combines global segmentation loss with per-instance (blob) loss.
    It automatically detects connected components in the ground truth and computes
    separate losses for each instance, which are then weighted and combined with
    the global loss.

    The binary label is used for the global part, and multi-instance labels are
    automatically generated from connected components for the blob part.

    Args:
        base_loss (str): Name of the base loss function to use (e.g., "DiceLoss", "DiceCELoss").
                        Default: "DiceLoss"
        main_weight (float): Weight for the main (global) loss. Default: 1.0
        blob_weight (float): Weight for the blob (per-instance) loss. Default: 0.0
        to_onehot_y (bool): Convert target to one-hot encoding. Default: False
        softmax (bool): Apply softmax to predictions. Default: False
        sigmoid (bool): Apply sigmoid to predictions. Default: True
        **base_loss_kwargs: Additional keyword arguments passed to the base loss function

    Example:
        >>> loss_fn = BlobLoss(
        ...     base_loss="DiceLoss",
        ...     main_weight=1.0,
        ...     blob_weight=1.0,
        ...     sigmoid=True
        ... )
        >>> pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        >>> target = torch.randint(0, 2, (2, 1, 64, 64))
        >>> loss = loss_fn(pred, target)
        >>> loss.backward()

    **YAML Configuration:**

        loss:
          type: BlobLoss
          base_loss: DiceLoss
          main_weight: 1.0
          blob_weight: 1.0
          sigmoid: true
          to_onehot_y: false

    Key features:
    - Combines global and per-instance loss computation
    - Automatically generates multi-instance labels using connected components
    - Supports any MONAI loss as the base loss function
    - Fully differentiable with gradient flow through both loss components
    """

    main_weight: float
    blob_weight: float
    to_onehot_y: bool
    softmax: bool
    sigmoid: bool
    main_criterion: nn.Module | None
    blob_criterion: nn.Module | None

    def __init__(
        self,
        base_loss: str = "DiceLoss",
        main_weight: float = 1.0,
        blob_weight: float = 0.0,
        to_onehot_y: bool = False,
        softmax: bool = False,
        sigmoid: bool = True,
        **base_loss_kwargs,
    ) -> None:
        """Initialize BlobLoss."""
        super().__init__()

        self.main_weight = main_weight  # type: ignore
        self.blob_weight = blob_weight  # type: ignore
        self.to_onehot_y = to_onehot_y  # type: ignore
        self.softmax = softmax  # type: ignore
        self.sigmoid = sigmoid  # type: ignore

        # Get the base loss class from MONAI
        base_loss_class = getattr(losses, base_loss)

        # Create base loss instance for main loss (global)
        self.main_criterion = (  # type: ignore
            base_loss_class(
                to_onehot_y=to_onehot_y,
                softmax=False,  # We handle activation ourselves
                sigmoid=False,
                **base_loss_kwargs,
            )
            if main_weight > 0
            else None
        )

        # Create base loss instance for blob loss (per-instance)
        self.blob_criterion = (  # type: ignore
            base_loss_class(
                to_onehot_y=False,  # We handle conversion ourselves
                softmax=False,
                sigmoid=False,
                **base_loss_kwargs,
            )
            if blob_weight > 0
            else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Blob Loss.

        Args:
            pred: Network predictions of shape (B, C, H, W) or (B, C, H, W, D)
                Raw logits that will be activated based on sigmoid/softmax settings.

            target: Ground truth labels of shape (B, H, W) or (B, H, W, D) with class indices,
                   or (B, C, H, W) or (B, C, H, W, D) if one-hot encoded

        Returns:
            Scalar loss tensor (differentiable)
        """
        # Apply activations to predictions
        if self.softmax:
            activated_pred = F.softmax(pred, dim=1)
        elif self.sigmoid:
            activated_pred = torch.sigmoid(pred)
        else:
            activated_pred = pred

        # Convert target to one-hot if needed
        if self.to_onehot_y and target.dim() != pred.dim():
            target = self._to_onehot(target, pred.shape[1])

        # Ensure target is float32
        if target.dtype != torch.float32:
            target = target.float()

        # Save original target for blob loss computation
        # We may need different shapes for main vs blob loss
        target_for_blob = target.clone()

        # Normalize target shape for class indices with spurious channel dimension
        # This is needed for blob loss computation
        if target.dim() == pred.dim() and target.shape[1] == 1:
            # Target is class indices with spurious channel: (B, 1, H, W) -> (B, H, W)
            target_for_blob = target.squeeze(1)

        # Initialize losses
        main_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        blob_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Compute main (global) loss
        # Pass target in its original form for main criterion
        if self.main_weight > 0 and self.main_criterion is not None:
            main_loss = self.main_criterion(activated_pred, target)

        # Compute blob (per-instance) loss
        if self.blob_weight > 0 and self.blob_criterion is not None:
            blob_loss = self._compute_blob_loss(activated_pred, target_for_blob)

        # Combine losses
        if self.blob_weight == 0 and self.main_weight > 0:
            loss = main_loss
        elif self.main_weight == 0 and self.blob_weight > 0:
            loss = blob_loss
        elif self.main_weight > 0 and self.blob_weight > 0:
            loss = main_loss * self.main_weight + blob_loss * self.blob_weight
        else:
            loss = main_loss + blob_loss

        return loss

    def _compute_blob_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute blob loss for multi-instance segmentation.

        This function:
        1. Loops through elements in the batch
        2. Converts binary labels to multi-instance labels using connected components
        3. Loops through blobs per element and computes loss
        4. Averages across blobs and batch

        Args:
            pred: Network prediction tensor (after activation)
            target: Ground truth label tensor

        Returns:
            torch.Tensor: Mean blob loss across the batch
        """
        # Early return if blob criterion is not set
        if self.blob_criterion is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        batch_size = pred.shape[0]
        element_blob_losses = []

        # Loop over elements in batch
        for b in range(batch_size):
            # Extract single sample
            pred_volume = pred[b]  # (C, H, W) or (C, H, W, D)
            target_volume = target[b]  # (C, H, W) or (C, H, W, D) or (H, W) etc.

            # Handle different target shapes - convert to binary mask
            if target_volume.dim() == pred_volume.dim() - 1:
                # Target is class indices: (H, W) or (H, W, D)
                binary_label = (target_volume > 0).float()
            elif target_volume.dim() == pred_volume.dim():
                # Target is one-hot: (C, H, W) or (C, H, W, D)
                # Sum over channels to get binary mask (foreground vs background)
                if target_volume.shape[0] > 1:
                    # Multi-class: exclude background (channel 0)
                    binary_label = target_volume[1:].sum(dim=0)
                else:
                    binary_label = target_volume[0]
            else:
                raise ValueError(f"Unexpected target shape: {target_volume.shape}")

            # Generate multi-instance labels using connected components
            multi_label, num_instances = gpu_connected_components(binary_label)

            if num_instances == 0:
                # No instances in this sample, skip
                continue

            # Get unique labels (excluding background 0)
            unique_labels = torch.unique(multi_label)
            unique_labels = unique_labels[unique_labels > 0]

            blob_losses = []

            # Loop through each instance (blob)
            for label_id in unique_labels:
                # Create masked prediction and target
                # We mask the output and use binary label for this instance
                label_mask = torch.ones_like(binary_label)
                label_mask[multi_label != label_id] = 0
                label_mask[multi_label == 0] = 0  # Exclude background

                # Prepare inputs for loss computation
                # For predictions, we use the full prediction volume
                # For targets, we create a binary mask for this specific instance
                instance_target = (multi_label == label_id).float()

                # Expand dimensions to match expected input shape
                # Add batch dimension: (C, H, W) -> (1, C, H, W)
                masked_pred = (pred_volume * label_mask.unsqueeze(0)).unsqueeze(0)
                masked_target = instance_target.unsqueeze(0).unsqueeze(0)

                # Compute loss for this blob
                blob_loss_value = self.blob_criterion(masked_pred, masked_target)
                blob_losses.append(blob_loss_value)

            # Average over blobs for this element
            if blob_losses:
                mean_blob_loss = torch.mean(torch.stack(blob_losses))
                element_blob_losses.append(mean_blob_loss)

        # Average over batch
        if element_blob_losses:
            mean_element_blob_loss = torch.mean(torch.stack(element_blob_losses))
        else:
            mean_element_blob_loss = torch.tensor(
                0.0, device=pred.device, dtype=pred.dtype
            )

        return mean_element_blob_loss

    @staticmethod
    def _to_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert class indices to one-hot encoding.

        Args:
            target: Tensor of shape (B, H, W) or (B, H, W, D) with class indices
            num_classes: Number of classes

        Returns:
            One-hot encoded tensor of shape (B, C, H, W) or (B, C, H, W, D)
        """
        return (
            F.one_hot(target.long(), num_classes=num_classes)
            .permute(0, -1, *range(1, target.dim()))
            .float()
        )


__all__ = ["BlobLoss"]
