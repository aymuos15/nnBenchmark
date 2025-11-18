"""Connected Components Loss for instance segmentation.

This module provides a differentiable loss function that evaluates predictions
per connected component, enabling better handling of multi-instance segmentation tasks.

Based on the nnunetv2 implementation but adapted for the nnBenchmark framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.factory.cc_utils import get_gt_regions


class CCLoss(nn.Module):
    """Connected Components Loss for multi-instance segmentation.

    This loss function computes loss by evaluating predictions at the region level
    using connected components from the ground truth. For each connected component,
    a dice score is computed, and the loss is derived from the mean dice score.

    This is particularly useful for multi-instance segmentation tasks where
    instances should be evaluated individually rather than globally.

    The loss is fully differentiable and supports backpropagation through
    the dice computation.

    Args:
        to_onehot_y (bool): Convert target to one-hot encoding. Default: False
        softmax (bool): Apply softmax to predictions. Default: False
        sigmoid (bool): Apply sigmoid to predictions. Default: True

    Example:
        >>> loss_fn = CCLoss(sigmoid=True)
        >>> pred = torch.randn(2, 3, 64, 64, requires_grad=True)
        >>> target = torch.randint(0, 3, (2, 64, 64))
        >>> loss = loss_fn(pred, target)
        >>> loss.backward()  # Gradients flow through the loss

    **YAML Configuration Notes for Switching from Other Losses:**

    When switching from DiceCELoss or other losses to CCLoss in your config file:

    BEFORE (DiceCELoss):
        loss:
          type: DiceCELoss
          to_onehot_y: true
          softmax: true
          batch: true  # ← Remove this, CCLoss doesn't support it

    AFTER (CCLoss):
        loss:
          type: CCLoss
          to_onehot_y: true
          sigmoid: true  # ← Change softmax to sigmoid
          # Do NOT include batch parameter

    Key differences:
    - Uses `sigmoid` instead of `softmax` for activation
    - No `batch` parameter (CCLoss evaluates regions, not batch-wise)
    - Better for multi-instance segmentation tasks
    """

    to_onehot_y: bool
    softmax: bool
    sigmoid: bool

    def __init__(
        self,
        to_onehot_y: bool = False,
        softmax: bool = False,
        sigmoid: bool = True,
        batch: bool = True,  # Ignored, kept for config compatibility
    ) -> None:
        """Initialize CCLoss."""
        super().__init__()
        self.to_onehot_y = to_onehot_y  # type: ignore
        self.softmax = softmax  # type: ignore
        self.sigmoid = sigmoid  # type: ignore
        # batch parameter is ignored - CCLoss evaluates per region, not batch-wise

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Connected Components Loss.

        Args:
            pred: Network predictions of shape (B, C, H, W) or (B, C, H, W, D)
                Raw logits that will be activated based on sigmoid/softmax settings.

            target: Ground truth labels
                If to_onehot_y=False: Shape (B, H, W) or (B, H, W, D) with class indices
                If to_onehot_y=True: Shape (B, C, H, W) or (B, C, H, W, D) one-hot encoded

        Returns:
            Scalar loss tensor (differentiable)
        """
        # Apply activations to predictions
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        elif self.sigmoid:
            pred = torch.sigmoid(pred)

        # Convert target to one-hot if needed
        if self.to_onehot_y and target.dim() != pred.dim():
            target = self._to_onehot(target, pred.shape[1])

        # Ensure target is float32
        if target.dtype != torch.float32:
            target = target.float()

        # Normalize target shape for class indices with spurious channel dimension
        # When LoadImaged with ensure_channel_first=true is applied to class index targets,
        # they arrive as (B, 1, H, W) or (B, 1, H, W, D) instead of the expected (B, H, W) or (B, H, W, D).
        # Following MONAI's philosophy of robustness, we squeeze this spurious single dimension.
        # This is consistent with how DiceCELoss handles target shape normalization internally.
        if target.dim() == pred.dim() and target.shape[1] == 1:
            # Target is class indices with spurious channel: (B, 1, H, W) -> (B, H, W)
            target = target.squeeze(1)

        # Process each sample in the batch
        batch_size = pred.shape[0]
        batch_losses = []

        for b in range(batch_size):
            # Extract single sample
            pred_volume = pred[b]  # (C, H, W) or (C, H, W, D)
            target_volume = target[b]  # (C, H, W) or (C, H, W, D) or (H, W) etc.

            # Handle different target shapes
            if target_volume.dim() == pred_volume.dim() - 1:
                # Target is class indices, convert to one-hot
                target_onehot = F.one_hot(
                    target_volume.long(), num_classes=pred_volume.shape[0]
                )
                # Permute from (H, W, C) to (C, H, W)
                target_onehot = target_onehot.permute(
                    -1, *range(target_onehot.dim() - 1)
                ).float()
            else:
                target_onehot = target_volume

            # Get connected components from target
            sample_loss = self._compute_region_loss(pred_volume, target_onehot)
            batch_losses.append(sample_loss)

        # Mean loss across batch
        loss = torch.mean(torch.stack(batch_losses))
        return loss

    def _compute_region_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for a single sample by evaluating regions separately.

        Args:
            pred: Predictions of shape (C, H, W) or (C, H, W, D)
            target: One-hot target of shape (C, H, W) or (C, H, W, D)

        Returns:
            Scalar loss tensor for this sample
        """
        num_classes = pred.shape[0]
        class_losses = []

        # Process each class separately
        for c in range(num_classes):
            pred_class = pred[c]  # (H, W) or (H, W, D)
            target_class = target[c]  # (H, W) or (H, W, D)

            # Get connected components in ground truth
            region_map, num_regions = get_gt_regions(target_class, pred.device)

            if num_regions == 0:
                # No ground truth for this class, loss = 1.0
                # Use pred to create a tensor that's part of the computation graph
                class_losses.append(pred_class.sum() * 0.0 + 1.0)
                continue

            # Compute dice score for each region
            region_dice_scores = []

            for region_id in range(1, num_regions + 1):
                region_mask = region_map == region_id

                # Extract predictions and target for this region
                pred_region = pred_class[region_mask]
                target_region = target_class[region_mask]

                # Handle empty regions
                if target_region.sum() == 0:
                    # Ground truth is empty for this region
                    # Use pred_region to create a tensor that's part of the computation graph
                    region_dice_scores.append(pred_region.sum() * 0.0 + 1.0)
                    continue

                # Compute differentiable dice score
                dice_score = self._dice_score(pred_region, target_region)
                region_dice_scores.append(dice_score)

            # Mean dice score across regions for this class
            if region_dice_scores:
                mean_dice = torch.mean(torch.stack(region_dice_scores))
            else:
                # Use pred_class to create a tensor that's part of the computation graph
                mean_dice = pred_class.sum() * 0.0 + 1.0

            # Convert to loss: loss = 1 - dice_score
            class_loss = 1.0 - mean_dice
            class_losses.append(class_loss)

        # Mean loss across all classes (skip background class 0 if only one class present)
        if class_losses:
            # Average from class 1 onwards (skip background)
            if len(class_losses) > 1:
                sample_loss = torch.mean(torch.stack(class_losses[1:]))
            else:
                sample_loss = class_losses[0]
        else:
            # Use pred to create a tensor that's part of the computation graph
            sample_loss = pred.sum() * 0.0 + 1.0

        return sample_loss

    @staticmethod
    def _dice_score(
        pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute differentiable dice score.

        Args:
            pred: Prediction tensor (flattened or 1D)
            target: Target tensor (flattened or 1D)
            smooth: Smoothing factor

        Returns:
            Scalar dice score tensor (differentiable)
        """
        intersection = torch.sum(pred * target)
        dice = (2.0 * intersection + smooth) / (
            torch.sum(pred) + torch.sum(target) + smooth
        )
        return dice

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


__all__ = ["CCLoss"]
