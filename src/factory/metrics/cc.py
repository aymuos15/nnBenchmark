"""Connected Components Metric for instance segmentation evaluation.

This module provides a MONAI-compatible metric that evaluates predictions
per connected component, enabling better assessment of multi-instance segmentation tasks.

The metric can be used as a drop-in replacement for DiceMetric in configuration files.
"""

from typing import Optional, Tuple

import torch

from src.factory.cc_utils import get_gt_regions


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

        # State for tracking instance-size binned statistics
        # Each element is (score, instance_size) tuple
        self._instance_scores: list[tuple[float, int]] = []

        # Track instances per sample for per-sample binned statistics
        # Structure: list of dicts, one per sample, each containing {size_bin: [scores]}
        self._per_sample_instances: list[dict[str, list[float]]] = []
        self._current_sample_instances: dict[str, list[float]] = {
            "all": [],
            "0-2cc": [],
            "2-10cc": [],
            ">10cc": [],
        }

        self._binned_scores: dict[str, list[float]] = {
            "all": [],
            "0-2cc": [],
            "2-10cc": [],
            ">10cc": [],
        }

        # State for tracking FP/TP/FN instance counts by size
        self._tp_instances: list[int] = []  # List of TP GT instance sizes
        self._fn_instances: list[int] = []  # List of FN GT instance sizes
        self._fp_instances: list[int] = []  # List of FP predicted instance sizes

        # State for tracking per-sample FP/TP/FN counts
        self._per_sample_fp_tp_fn: list[dict[str, dict[str, int]]] = []
        self._current_sample_fp_tp_fn: dict[str, dict[str, int]] = {
            "all": {"TP": 0, "FN": 0, "FP": 0},
            "0-2cc": {"TP": 0, "FN": 0, "FP": 0},
            "2-10cc": {"TP": 0, "FN": 0, "FP": 0},
            ">10cc": {"TP": 0, "FN": 0, "FP": 0},
        }

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
            # Reset per-sample instance tracking for this sample
            self._current_sample_instances = {
                "all": [],
                "0-2cc": [],
                "2-10cc": [],
                ">10cc": [],
            }

            # Reset per-sample FP/TP/FN tracking for this sample
            self._current_sample_fp_tp_fn = {
                "all": {"TP": 0, "FN": 0, "FP": 0},
                "0-2cc": {"TP": 0, "FN": 0, "FP": 0},
                "2-10cc": {"TP": 0, "FN": 0, "FP": 0},
                ">10cc": {"TP": 0, "FN": 0, "FP": 0},
            }

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

            # Save per-sample instance data for later
            self._per_sample_instances.append(self._current_sample_instances.copy())

            # Save per-sample FP/TP/FN data for later
            self._per_sample_fp_tp_fn.append(self._current_sample_fp_tp_fn.copy())

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
            # Skip background class when tracking instance sizes (only track for foreground classes)
            should_track_instances = c > 0 or self.include_background

            pred_class = pred[c]
            target_class = target[c]

            # Handle empty classes
            score, handled = self._handle_empty_classes(pred_class, target_class)
            if handled:
                class_scores.append(score)
                # Still need to classify FP instances even when GT is empty or pred is empty
                if should_track_instances:
                    # When GT is empty, we need to count all predictions as FP
                    # When pred is empty, there are no FP to count
                    # When both are empty, no FP/TP/FN to count
                    pred_empty = torch.sum(pred_class) == 0

                    if not pred_empty:  # Only classify FP if predictions exist
                        from src.factory.cc_utils import gpu_connected_components

                        # Binarize predictions at threshold 0.5
                        pred_binary = (pred_class > 0.5).float()

                        # Run connected components on binarized predictions
                        labeled_pred, num_pred = gpu_connected_components(pred_binary)

                        # All predicted instances are FP when GT is empty
                        for pred_id in range(1, num_pred + 1):
                            pred_mask = labeled_pred == pred_id
                            pred_size = int(torch.sum(pred_mask).item())

                            # False Positive: Predicted instance with no GT
                            self._fp_instances.append(pred_size)
                            self._current_sample_fp_tp_fn["all"]["FP"] += 1
                            if pred_size < 2:
                                self._current_sample_fp_tp_fn["0-2cc"]["FP"] += 1
                            elif pred_size < 10:
                                self._current_sample_fp_tp_fn["2-10cc"]["FP"] += 1
                            else:
                                self._current_sample_fp_tp_fn[">10cc"]["FP"] += 1
                continue

            # Get regions from connected components
            region_map, labeled_gt, num_regions = get_gt_regions(
                target_class, pred.device
            )

            if num_regions == 0:
                # No regions found
                class_scores.append(torch.tensor(1.0, device=pred.device))
                # Still need to classify FP instances even when no GT regions found
                if should_track_instances:
                    pred_empty = torch.sum(pred_class) == 0

                    if not pred_empty:  # Only classify FP if predictions exist
                        from src.factory.cc_utils import gpu_connected_components

                        # Binarize predictions at threshold 0.5
                        pred_binary = (pred_class > 0.5).float()

                        # Run connected components on binarized predictions
                        labeled_pred, num_pred = gpu_connected_components(pred_binary)

                        # All predicted instances are FP when no GT regions
                        for pred_id in range(1, num_pred + 1):
                            pred_mask = labeled_pred == pred_id
                            pred_size = int(torch.sum(pred_mask).item())

                            # False Positive: Predicted instance with no GT
                            self._fp_instances.append(pred_size)
                            self._current_sample_fp_tp_fn["all"]["FP"] += 1
                            if pred_size < 2:
                                self._current_sample_fp_tp_fn["0-2cc"]["FP"] += 1
                            elif pred_size < 10:
                                self._current_sample_fp_tp_fn["2-10cc"]["FP"] += 1
                            else:
                                self._current_sample_fp_tp_fn[">10cc"]["FP"] += 1
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

                # Track instance size only for non-background classes
                # (number of pixels/voxels in original connected component)
                if should_track_instances:
                    # Use labeled_gt to get the actual size of the ground truth instance, not the Voronoi-expanded region
                    original_component_mask = labeled_gt == region_id
                    instance_size = int(torch.sum(original_component_mask).item())
                    score_value = float(region_score.item())

                    # Track globally
                    self._instance_scores.append((score_value, instance_size))

                    # Track per-sample
                    self._current_sample_instances["all"].append(score_value)
                    if instance_size < 2:
                        self._current_sample_instances["0-2cc"].append(score_value)
                    elif instance_size < 10:
                        self._current_sample_instances["2-10cc"].append(score_value)
                    else:
                        self._current_sample_instances[">10cc"].append(score_value)

            # Mean across regions for this class
            if region_scores:
                mean_score = torch.mean(torch.stack(region_scores))
            else:
                mean_score = torch.tensor(1.0, device=pred.device)

            class_scores.append(mean_score)

            # Classify instances as TP/FN/FP (only for non-background classes)
            if should_track_instances and num_regions > 0:
                self._classify_instances(
                    pred_class, target_class, labeled_gt, num_regions
                )

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

    def _classify_instances(
        self,
        pred_class: torch.Tensor,
        target_class: torch.Tensor,
        labeled_gt: torch.Tensor,
        num_gt_regions: int,
    ) -> None:
        """
        Classify instances as TP/FN/FP based on overlap detection.

        Args:
            pred_class: Prediction probabilities for a class (soft, 0-1)
            target_class: Ground truth for a class (binary, 0 or 1)
            labeled_gt: Labeled ground truth components (each component has unique ID)
            num_gt_regions: Number of ground truth regions

        Classification logic:
            - TP: GT instance has ANY overlap with predictions (intersection > 0)
            - FN: GT instance has NO overlap with predictions (intersection == 0)
            - FP: Predicted instance has NO overlap with any GT (intersection == 0)
        """
        from src.factory.cc_utils import gpu_connected_components

        # Binarize predictions at threshold 0.5
        pred_binary = (pred_class > 0.5).float()

        # Run connected components on binarized predictions
        labeled_pred, num_pred = gpu_connected_components(pred_binary)

        # Classify GT instances as TP or FN
        for gt_id in range(1, num_gt_regions + 1):
            gt_mask = labeled_gt == gt_id
            gt_size = int(torch.sum(gt_mask).item())

            # Check if this GT instance has ANY intersection with predictions
            intersection = torch.sum(gt_mask & (pred_binary > 0)).item()

            if intersection > 0:
                # True Positive: GT instance overlaps with predictions
                self._tp_instances.append(gt_size)
                # Update per-sample counts
                self._current_sample_fp_tp_fn["all"]["TP"] += 1
                if gt_size < 2:
                    self._current_sample_fp_tp_fn["0-2cc"]["TP"] += 1
                elif gt_size < 10:
                    self._current_sample_fp_tp_fn["2-10cc"]["TP"] += 1
                else:
                    self._current_sample_fp_tp_fn[">10cc"]["TP"] += 1
            else:
                # False Negative: GT instance has no overlap with predictions
                self._fn_instances.append(gt_size)
                # Update per-sample counts
                self._current_sample_fp_tp_fn["all"]["FN"] += 1
                if gt_size < 2:
                    self._current_sample_fp_tp_fn["0-2cc"]["FN"] += 1
                elif gt_size < 10:
                    self._current_sample_fp_tp_fn["2-10cc"]["FN"] += 1
                else:
                    self._current_sample_fp_tp_fn[">10cc"]["FN"] += 1

        # Classify predicted instances as FP (those with no GT overlap)
        for pred_id in range(1, num_pred + 1):
            pred_mask = labeled_pred == pred_id
            pred_size = int(torch.sum(pred_mask).item())

            # Check if this predicted instance has ANY intersection with GT
            intersection = torch.sum(pred_mask & (target_class > 0)).item()

            if intersection == 0:
                # False Positive: Predicted instance has no overlap with any GT
                self._fp_instances.append(pred_size)
                # Update per-sample counts
                self._current_sample_fp_tp_fn["all"]["FP"] += 1
                if pred_size < 2:
                    self._current_sample_fp_tp_fn["0-2cc"]["FP"] += 1
                elif pred_size < 10:
                    self._current_sample_fp_tp_fn["2-10cc"]["FP"] += 1
                else:
                    self._current_sample_fp_tp_fn[">10cc"]["FP"] += 1

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

    def _compute_binned_statistics(self) -> None:
        """Compute binned statistics from accumulated instance scores.

        Categorizes instance scores into bins based on instance SIZE (pixel/voxel count):
        - all: all instances
        - 0-2cc: instances with size [0, 2) pixels/voxels
        - 2-10cc: instances with size [2, 10) pixels/voxels
        - >10cc: instances with size >= 10 pixels/voxels
        """
        # Clear previous binned scores
        for key in self._binned_scores:
            self._binned_scores[key] = []

        # Bin all instance scores by instance size
        for score, instance_size in self._instance_scores:
            # All instances go to "all" bin
            self._binned_scores["all"].append(score)

            # Also add to category bin based on instance size
            if instance_size < 2:
                self._binned_scores["0-2cc"].append(score)
            elif instance_size < 10:
                self._binned_scores["2-10cc"].append(score)
            else:
                self._binned_scores[">10cc"].append(score)

    def get_binned_statistics(self) -> dict[str, dict[str, float]]:
        """Get CC loss statistics binned by instance size.

        Bins instances based on their size (pixel/voxel count) and returns
        CC loss statistics for each size bin.

        Returns:
            Dictionary with bin names as keys and statistics dicts as values.
            Each statistics dict contains: mean, std, min, max, count of CC loss scores
            for instances in that size bin.

            Example:
            {
                "all": {"mean": 0.75, "std": 0.15, "min": 0.2, "max": 0.98, "count": 500},
                "0-2cc": {"mean": 0.45, "std": 0.20, "min": 0.1, "max": 0.95, "count": 150},
                "2-10cc": {"mean": 0.80, "std": 0.12, "min": 0.3, "max": 0.99, "count": 250},
                ">10cc": {"mean": 0.85, "std": 0.10, "min": 0.4, "max": 1.0, "count": 100}
            }

            Where:
            - 0-2cc: CC loss scores for instances with 0-2 pixels/voxels
            - 2-10cc: CC loss scores for instances with 2-10 pixels/voxels
            - >10cc: CC loss scores for instances with >10 pixels/voxels
        """
        import numpy as np

        # Compute bins first
        self._compute_binned_statistics()

        # Compute statistics for each bin
        binned_stats: dict[str, dict[str, float]] = {}

        for bin_name, bin_scores in self._binned_scores.items():
            if len(bin_scores) == 0:
                # Empty bin: create default stats
                binned_stats[bin_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                }
            else:
                # Non-empty bin: compute statistics
                scores_array = np.array(bin_scores)
                binned_stats[bin_name] = {
                    "mean": float(np.mean(scores_array)),
                    "std": float(np.std(scores_array)),
                    "min": float(np.min(scores_array)),
                    "max": float(np.max(scores_array)),
                    "count": int(len(bin_scores)),
                }

        return binned_stats

    def get_fp_tp_fn_statistics(self) -> dict[str, dict[str, int]]:
        """Get FP/TP/FN instance counts binned by instance size.

        Bins instances based on their size (pixel/voxel count) and returns
        TP/FN/FP counts for each size bin.

        Returns:
            Dictionary with bin names as keys and count dicts as values.
            Each count dict contains: TP, FN, FP counts for that size bin.

            Example:
            {
                "0-2cc": {"TP": 120, "FN": 30, "FP": 25},
                "2-10cc": {"TP": 180, "FN": 20, "FP": 15},
                ">10cc": {"TP": 45, "FN": 5, "FP": 3},
                "all": {"TP": 345, "FN": 55, "FP": 43}
            }

            Where:
            - TP: True Positive GT instances (have overlap with predictions)
            - FN: False Negative GT instances (no overlap with predictions)
            - FP: False Positive predicted instances (no overlap with GT)
            - Sizes are binned as: 0-2cc, 2-10cc, >10cc pixels/voxels
        """
        # Initialize bins
        bins = {
            "0-2cc": {"TP": 0, "FN": 0, "FP": 0},
            "2-10cc": {"TP": 0, "FN": 0, "FP": 0},
            ">10cc": {"TP": 0, "FN": 0, "FP": 0},
            "all": {"TP": 0, "FN": 0, "FP": 0},
        }

        # Bin TP instances by GT size
        for size in self._tp_instances:
            bins["all"]["TP"] += 1
            if size < 2:
                bins["0-2cc"]["TP"] += 1
            elif size < 10:
                bins["2-10cc"]["TP"] += 1
            else:
                bins[">10cc"]["TP"] += 1

        # Bin FN instances by GT size
        for size in self._fn_instances:
            bins["all"]["FN"] += 1
            if size < 2:
                bins["0-2cc"]["FN"] += 1
            elif size < 10:
                bins["2-10cc"]["FN"] += 1
            else:
                bins[">10cc"]["FN"] += 1

        # Bin FP instances by predicted size
        for size in self._fp_instances:
            bins["all"]["FP"] += 1
            if size < 2:
                bins["0-2cc"]["FP"] += 1
            elif size < 10:
                bins["2-10cc"]["FP"] += 1
            else:
                bins[">10cc"]["FP"] += 1

        return bins

    def get_per_sample_fp_tp_fn_statistics(self) -> list[dict[str, dict[str, int]]]:
        """Get per-sample FP/TP/FN instance counts binned by instance size.

        Returns:
            List of dicts, one per sample, each containing FP/TP/FN counts per size bin.
            Each sample dict has structure:
            {
                "all": {"TP": 10, "FN": 2, "FP": 5},
                "0-2cc": {"TP": 0, "FN": 1, "FP": 3},
                "2-10cc": {"TP": 2, "FN": 1, "FP": 2},
                ">10cc": {"TP": 8, "FN": 0, "FP": 0}
            }

            Example:
            >>> stats = metric.get_per_sample_fp_tp_fn_statistics()
            >>> sample_0 = stats[0]
            >>> # Access: sample_0['all']['TP'], sample_0['all']['FP']
        """
        return self._per_sample_fp_tp_fn

    def reset(self) -> None:
        """Reset accumulated scores. Required for MONAI compatibility.

        Note: Only resets _scores (per-sample scores) for MONAI compatibility.
        Instance scores are NOT reset here to allow accumulation across the entire
        validation/inference run for final binned statistics computation.
        """
        self._scores = []
        # Do NOT reset _instance_scores - they need to accumulate across all samples
        # to compute final binned statistics at the end of validation/inference

    def reset_instance_scores(self) -> None:
        """Reset instance scores for a new validation/inference run.

        This should be called when starting a fresh validation or inference
        to clear accumulated instance scores from previous runs.
        """
        self._instance_scores = []
        self._per_sample_instances = []
        self._current_sample_instances = {
            "all": [],
            "0-2cc": [],
            "2-10cc": [],
            ">10cc": [],
        }
        self._binned_scores = {
            "all": [],
            "0-2cc": [],
            "2-10cc": [],
            ">10cc": [],
        }
        # Also reset FP/TP/FN tracking
        self._tp_instances = []
        self._fn_instances = []
        self._fp_instances = []
        self._per_sample_fp_tp_fn = []
        self._current_sample_fp_tp_fn = {
            "all": {"TP": 0, "FN": 0, "FP": 0},
            "0-2cc": {"TP": 0, "FN": 0, "FP": 0},
            "2-10cc": {"TP": 0, "FN": 0, "FP": 0},
            ">10cc": {"TP": 0, "FN": 0, "FP": 0},
        }

    def get_per_sample_binned_statistics(self) -> list[dict[str, dict[str, float]]]:
        """Get per-sample binned statistics for all samples.

        Returns:
            List of dicts, one per sample, each containing binned statistics.
            Each sample dict has keys: all, 0-2cc, 2-10cc, >10cc
            Each bin contains: mean, std, min, max, count

            Example:
            [
                {
                    "all": {"mean": 0.85, "std": 0.12, "min": 0.5, "max": 0.98, "count": 5},
                    "0-2cc": {"mean": 0.45, "std": 0.0, "min": 0.45, "max": 0.45, "count": 1},
                    ...
                },
                {...}  # next sample
            ]
        """
        import numpy as np

        per_sample_stats = []

        for sample_instances in self._per_sample_instances:
            sample_stats = {}

            for bin_name, bin_scores in sample_instances.items():
                if len(bin_scores) == 0:
                    # Empty bin
                    sample_stats[bin_name] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "count": 0,
                    }
                else:
                    # Non-empty bin
                    scores_array = np.array(bin_scores)
                    sample_stats[bin_name] = {
                        "mean": float(np.mean(scores_array)),
                        "std": float(np.std(scores_array)),
                        "min": float(np.min(scores_array)),
                        "max": float(np.max(scores_array)),
                        "count": int(len(bin_scores)),
                    }

            per_sample_stats.append(sample_stats)

        return per_sample_stats

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


__all__ = ["CCMetric"]
