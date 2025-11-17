"""Connected Components Metric for instance segmentation evaluation.

This module provides a MONAI-compatible metric that evaluates predictions
per connected component, enabling better assessment of multi-instance segmentation tasks.

The metric can be used as a drop-in replacement for DiceMetric in configuration files.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch

if TYPE_CHECKING:
    import cupy as cp  # type: ignore[import]
    from cucim.skimage import measure as cucim_measure  # type: ignore[import]
    from cupyx.scipy.ndimage import distance_transform_edt  # type: ignore[import]
else:
    try:
        import cupy as cp
        from cucim.skimage import measure as cucim_measure
        from cupyx.scipy.ndimage import distance_transform_edt
    except ImportError:
        cp = None  # type: ignore
        cucim_measure = None  # type: ignore
        distance_transform_edt = None  # type: ignore


class CCMetric:
    """Connected Components Metric for multi-instance segmentation evaluation.

    This metric computes evaluation scores by analyzing predictions at the region level
    using connected components from the ground truth. For each connected component,
    a base metric (e.g., Dice) is computed, and the final score is the mean across regions.

    This is particularly useful for multi-instance segmentation tasks where
    instances should be evaluated individually rather than globally.

    Args:
        include_background (bool): Whether to include background class (class 0).
            Default: False
        reduction (str): Reduction method for batch aggregation.
            Options: "mean", "mean_batch", "sum", "none"
            Default: "mean_batch"
        num_classes (int): Number of classes in the segmentation.
            Required parameter.
        get_not_nans (bool): Whether to return only non-NaN values.
            Default: False
        metric_type (str): Type of metric to compute per region.
            Options: "dice", "surface_dice"
            Default: "dice"
        class_thresholds (list[float]): Distance thresholds for surface dice per class.
            Required if metric_type is "surface_dice".
            Example: [2.0] for single class, [2.0, 3.0] for two classes
        distance_metric (str): Distance metric for surface dice computation.
            Options: "euclidean", "chessboard", "taxicab"
            Default: "euclidean"

    Example:
        >>> # In YAML config, replace DiceMetric with CCMetric
        >>> # metrics:
        >>> #   - type: CCMetric
        >>> #     include_background: false
        >>> #     reduction: mean_batch
        >>> #     num_classes: 3
        >>>
        >>> # In Python code
        >>> metric = CCMetric(include_background=False, reduction="mean_batch", num_classes=3)
        >>> pred = torch.randn(2, 3, 64, 64).softmax(dim=1)  # (B, C, H, W)
        >>> target = torch.randint(0, 3, (2, 1, 64, 64))     # (B, 1, H, W)
        >>> score = metric(pred, target)
        >>> f"CC-Dice Score: {score.item():.4f}"  # doctest: +SKIP

    **YAML Configuration Examples:**

    Replace DiceMetric:
        BEFORE:
            - type: DiceMetric
              include_background: false
              reduction: mean_batch
              num_classes: 3

        AFTER:
            - type: CCMetric
              include_background: false
              reduction: mean_batch
              num_classes: 3

    Reference:
        Jaus, A., et al. "Every Component Counts: Rethinking the Measure of Success
        for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks."
        AAAI 2025.
    """

    def __init__(
        self,
        include_background: bool = False,
        reduction: str = "mean_batch",
        num_classes: Optional[int] = None,
        get_not_nans: bool = False,
        metric_type: str = "dice",
        class_thresholds: Optional[list[float]] = None,
        distance_metric: str = "euclidean",
    ) -> None:
        """Initialize CCMetric."""
        if num_classes is None:
            raise ValueError("num_classes is required for CCMetric")

        self.include_background = include_background
        self.reduction = reduction
        self.num_classes = num_classes
        self.get_not_nans = get_not_nans
        self.metric_type = metric_type
        self.class_thresholds = class_thresholds
        self.distance_metric = distance_metric

        # Validate reduction method
        valid_reductions = ["mean", "mean_batch", "sum", "none"]
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, got '{reduction}'"
            )

        # Validate metric_type
        valid_metric_types = ["dice", "surface_dice"]
        if metric_type not in valid_metric_types:
            raise ValueError(
                f"metric_type must be one of {valid_metric_types}, got '{metric_type}'"
            )

        # Validate surface dice parameters
        if metric_type == "surface_dice":
            if class_thresholds is None:
                raise ValueError(
                    f"class_thresholds is required when metric_type='{metric_type}'"
                )

        # State for accumulating results (MONAI pattern)
        self._scores: list[torch.Tensor] = []

    def __call__(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Accumulate Connected Components metric scores.

        This follows MONAI's accumulation pattern where __call__ stores results
        and aggregate() computes the final metric.

        Args:
            y_pred: Predictions of shape (B, C, H, W) or (B, C, H, W, D)
                Can be raw logits, softmax probabilities, or one-hot encoded.
                If not one-hot, softmax will be applied along channel dimension.

            y: Ground truth labels
                Shape (B, 1, H, W) or (B, H, W) with class indices, OR
                Shape (B, C, H, W) if already one-hot encoded
        """
        # Convert to one-hot if needed
        y_pred, y_onehot = self._prepare_inputs(y_pred, y)

        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]

        # Compute scores for each sample in batch
        for b in range(batch_size):
            pred_volume = y_pred[b]  # (C, H, W) or (C, H, W, D)
            target_volume = y_onehot[b]  # (C, H, W) or (C, H, W, D)

            # Compute per-class scores for this sample
            sample_scores = self._compute_sample_scores(pred_volume, target_volume)

            # Apply background exclusion
            if not self.include_background and num_classes > 1:
                # Exclude class 0 (background)
                sample_scores = sample_scores[1:]

            # Store scores for this sample
            self._scores.append(sample_scores)

    def _prepare_inputs(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare prediction and target tensors.

        Args:
            y_pred: Raw predictions
            y: Ground truth labels

        Returns:
            Tuple of (processed_pred, onehot_target)
        """
        # Apply softmax to predictions if they don't sum to 1 along channel dim
        # (i.e., they are raw logits)
        pred_sum = y_pred.sum(dim=1, keepdim=True)
        if not torch.allclose(pred_sum, torch.ones_like(pred_sum), atol=1e-3):
            y_pred = torch.softmax(y_pred, dim=1)

        # Convert target to one-hot if needed
        if y.shape[1] == 1 or y.dim() == y_pred.dim() - 1:
            # Target has class indices
            if y.dim() == y_pred.dim() and y.shape[1] == 1:
                y = y.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

            # One-hot encode
            y_onehot = torch.nn.functional.one_hot(
                y.long(), num_classes=self.num_classes
            )
            # Permute from (B, H, W, C) to (B, C, H, W)
            y_onehot = y_onehot.permute(0, -1, *range(1, y_onehot.dim() - 1)).float()
        else:
            y_onehot = y.float()

        return y_pred, y_onehot

    def _compute_sample_scores(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-class scores for a single sample.

        Args:
            pred: Predictions of shape (C, H, W) or (C, H, W, D)
            target: One-hot target of shape (C, H, W) or (C, H, W, D)

        Returns:
            Tensor of shape (C,) with per-class scores
        """
        num_classes = pred.shape[0]
        class_scores = []

        for c in range(num_classes):
            pred_class = pred[c]
            target_class = target[c]

            # Handle empty classes
            score, handled = self._handle_empty_classes(pred_class, target_class)
            if handled:
                class_scores.append(score)
                continue

            # Get regions from connected components
            region_map, num_regions = get_gt_regions(target_class, pred.device)

            if num_regions == 0:
                # No regions found
                class_scores.append(torch.tensor(1.0, device=pred.device))
                continue

            # Compute metric for each region
            region_scores = []
            for region_id in range(1, num_regions + 1):
                region_mask = region_map == region_id

                # Compute metric for this region based on metric_type
                if self.metric_type == "dice":
                    region_score = self._compute_region_dice(
                        pred_class, target_class, region_mask
                    )

                elif self.metric_type == "surface_dice":
                    # Use class-specific threshold if available, otherwise use first threshold
                    # class_thresholds is guaranteed to be non-None by __init__ validation
                    class_thresholds: list[float] = self.class_thresholds  # type: ignore[assignment]
                    threshold = (
                        class_thresholds[c]
                        if c < len(class_thresholds)
                        else class_thresholds[0]
                    )
                    pred_masked, target_masked = self._create_masked_region(
                        pred_class, target_class, region_mask
                    )
                    region_score = self._surface_dice_coefficient(
                        pred_masked,
                        target_masked,
                        threshold=threshold,
                        distance_metric=self.distance_metric,
                    )
                else:
                    raise ValueError(f"Unknown metric_type: {self.metric_type}")

                region_scores.append(region_score)

            # Mean across regions for this class
            if region_scores:
                mean_score = torch.mean(torch.stack(region_scores))
            else:
                mean_score = torch.tensor(1.0, device=pred.device)

            class_scores.append(mean_score)

        return torch.stack(class_scores)

    @staticmethod
    def _compute_region_dice(
        pred_class: torch.Tensor,
        target_class: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice coefficient for a specific region.

        Args:
            pred_class: Prediction tensor for a class
            target_class: Ground truth tensor for a class
            region_mask: Boolean mask identifying the region

        Returns:
            Dice coefficient score as tensor
        """
        pred_region = pred_class[region_mask]
        target_region = target_class[region_mask]
        pred_region = torch.clamp(pred_region, 0, 1)
        target_region = torch.clamp(target_region, 0, 1)
        return CCMetric._dice_coefficient(pred_region, target_region)

    @staticmethod
    def _create_masked_region(
        pred_class: torch.Tensor,
        target_class: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked versions of prediction and target for surface dice computation.

        Args:
            pred_class: Prediction tensor for a class
            target_class: Ground truth tensor for a class
            region_mask: Boolean mask identifying the region

        Returns:
            tuple: (pred_masked, target_masked) tensors with region isolated
        """
        pred_masked = pred_class.clone()
        pred_masked[~region_mask] = 0
        target_masked = target_class.clone()
        target_masked[~region_mask] = 0
        return pred_masked, target_masked

    @staticmethod
    def _handle_empty_classes(
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Handle edge cases where prediction or ground truth is empty.

        Returns:
            tuple: (score, handled) where handled is True if special case was handled
        """
        pred_empty = torch.sum(pred) == 0
        gt_empty = torch.sum(gt) == 0

        device = pred.device

        if pred_empty and gt_empty:
            return torch.tensor(1.0, device=device), True  # Both empty = perfect match
        elif gt_empty:
            return torch.tensor(1.0, device=device), True  # No ground truth = skip
        elif pred_empty:
            return torch.tensor(0.0, device=device), True  # Missed entire class

        return torch.tensor(0.0, device=device), False  # Not handled

    @staticmethod
    def _dice_coefficient(
        pred: torch.Tensor,
        gt: torch.Tensor,
        smooth: float = 1e-7,
    ) -> torch.Tensor:
        """
        Calculate Dice coefficient between prediction and ground truth.

        Args:
            pred: Prediction tensor
            gt: Ground truth tensor
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice coefficient score as tensor
        """
        intersection = torch.sum(pred * gt)
        dice = (2.0 * intersection + smooth) / (
            torch.sum(pred) + torch.sum(gt) + smooth
        )
        return dice

    @staticmethod
    def _surface_dice_coefficient(
        pred: torch.Tensor,
        gt: torch.Tensor,
        threshold: float = 2.0,
        distance_metric: str = "euclidean",
    ) -> torch.Tensor:
        """
        Calculate Surface Dice coefficient between prediction and ground truth.

        This computes the Normalized Surface Dice (NSD), which measures the fraction
        of the ground truth boundary that is correctly predicted within a distance threshold.

        Args:
            pred: Prediction tensor of shape (H, W) or (H, W, D)
            gt: Ground truth tensor of shape (H, W) or (H, W, D)
            threshold: Distance threshold in pixels/mm for boundary matching
            distance_metric: Distance metric ("euclidean", "chessboard", "taxicab")

        Returns:
            Surface Dice coefficient score as tensor (scalar between 0 and 1)
        """
        from monai.metrics import compute_surface_dice

        # Convert to binary if needed (thresholding at 0.5)
        pred_binary = (pred > 0.5).float()
        gt_binary = gt.float()

        # Add batch and channel dimensions for MONAI
        if pred_binary.dim() == 2:  # (H, W)
            pred_binary = pred_binary.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            gt_binary = gt_binary.unsqueeze(0).unsqueeze(0)
        elif pred_binary.dim() == 3:  # (H, W, D)
            pred_binary = pred_binary.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
            gt_binary = gt_binary.unsqueeze(0).unsqueeze(0)

        # Compute surface dice using MONAI
        surface_dice = compute_surface_dice(
            y_pred=pred_binary,
            y=gt_binary,
            class_thresholds=[threshold],
            include_background=True,
            distance_metric=distance_metric,
        )
        # Result is tensor, squeeze to scalar
        return surface_dice.squeeze()

    def reset(self) -> None:
        """Reset accumulated scores. Required for MONAI compatibility."""
        self._scores = []

    def aggregate(self) -> torch.Tensor:
        """
        Aggregate accumulated scores and apply reduction.

        This is called by the inference handler to get final results.

        Returns:
            Aggregated metric score based on reduction strategy
        """
        if len(self._scores) == 0:
            # No scores accumulated yet
            return torch.tensor(0.0)

        # Stack all accumulated scores: shape (N, C) where N=num_samples, C=num_classes
        all_scores = torch.stack(self._scores)

        # Filter NaNs if requested
        if self.get_not_nans:
            all_scores = all_scores[~torch.isnan(all_scores)]

        # Apply reduction strategy
        if self.reduction == "mean_batch":
            # Mean across all samples and classes (scalar)
            return torch.mean(all_scores)
        elif self.reduction == "mean":
            # Mean across classes for each sample: shape (N,)
            return torch.mean(all_scores, dim=1)
        elif self.reduction == "sum":
            # Sum across all dimensions
            return torch.sum(all_scores)
        elif self.reduction == "none":
            # Return per-sample, per-class scores: shape (N, C)
            return all_scores
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ============================================================================
# Helper Functions for Connected Components
# ============================================================================


def gpu_connected_components(
    img: torch.Tensor,
    connectivity: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    PyTorch wrapper for calculating connected components on a GPU using cupy and cucim.

    From: https://github.com/aymuos15/GPU-Connected-Components

    Args:
        img: Input image tensor
        connectivity: Connectivity defining the neighborhood. Default is None.

    Returns:
        tuple: (labeled_img, num_features)
            - labeled_img: Labeled image with each connected component having a unique label
            - num_features: Number of connected components found
    """
    if cp is None:
        raise ImportError(
            "cupy and cucim are required for GPU connected components. "
            "Install with: pip install cupy-cuda11x cucim"
        )

    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(
        img_cupy, connectivity=connectivity, return_num=True
    )
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
    return labeled_img_torch, num_features


def get_gt_regions(
    gt: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Divides the ground truth segmentation space into regions based on proximity to instances.

    This function uses GPU-accelerated distance transforms to create a Voronoi-like partition
    of the image space, where each pixel is assigned to the closest ground truth instance.

    Args:
        gt: Ground truth segmentation for a single class
        device: Device to place tensors on

    Returns:
        tuple: (region_map, num_features)
            - region_map: Tensor where each pixel is labeled with the nearest region ID
            - num_features: Number of distinct regions/connected components
    """
    if distance_transform_edt is None:
        raise ImportError(
            "cupyx is required for distance transforms. "
            "Install with: pip install cupy-cuda11x"
        )

    # Identify connected components in the ground truth
    labeled_gt, num_features = gpu_connected_components(gt)

    if num_features == 0:
        return torch.zeros_like(gt, dtype=torch.long), 0

    # Initialize distance map (stores distance to nearest instance)
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    # Initialize region map (stores region ID for each pixel)
    region_map = torch.zeros_like(gt, dtype=torch.long)

    # Process each connected component (region) in the ground truth
    for region_label in range(1, num_features + 1):
        # Create binary mask for current region
        region_mask = labeled_gt == region_label

        # Convert PyTorch tensor to CuPy array (stays on GPU)
        region_mask_cupy = cp.asarray(region_mask)

        # Calculate Euclidean distance transform using GPU-accelerated cupyx
        distance_cupy = distance_transform_edt(
            ~region_mask_cupy,
            float64_distances=False,  # Use float32 for speed
        )

        # Convert CuPy array back to PyTorch tensor (stays on GPU)
        distance = torch.as_tensor(distance_cupy, device=device)

        # Initialize or update based on closest distance
        if region_label == 1 or distance_map.max() == 0:
            distance_map = distance
            region_map = region_label * torch.ones_like(gt, dtype=torch.long)
        else:
            # Update pixels that are closer to this region
            update_mask = distance < distance_map
            distance_map[update_mask] = distance[update_mask]
            region_map[update_mask] = region_label

    return region_map, num_features


__all__ = ["CCMetric", "gpu_connected_components", "get_gt_regions"]
