"""
Inference evaluation module for model evaluation and testing using MONAI directly.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from monai.networks.utils import one_hot
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from src.inference.strategy import create_inferer
from src.logging import (
    get_gpu_memory_string,
    log_gpu_memory,
)
from src.plotting.validation import save_validation_visualizations
from src.utils.data import get_class_labels

# Suppress MONAI deprecation warnings for get_mask_edges (used internally by SurfaceDiceMetric)
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    metric_fns: dict[str, Any],
    device: torch.device | None = None,
    cfg: dict[str, Any] | None = None,
    save_dir: str | None = None,
    epoch: int | None = None,
    save_viz: bool = False,
    verbose: bool = True,
    data_dicts: list[dict[str, str]] | None = None,
    logger: Any = None,
    log_gpu_mem: bool = False,
    data_dir: str | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation on a data loader and return scores.

    Args:
        model: The model to evaluate
        data_loader: DataLoader with evaluation data
        metric_fns: Dict of metric functions (name -> metric) for computing multiple metrics
        device: torch device
        cfg: Optional config dict (needed if save_viz=True for spatial_dims)
        save_dir: Optional directory to save visualizations
        epoch: Optional epoch number (for visualization naming)
        save_viz: Whether to save visualizations of first batch
        verbose: Whether to print per-sample scores
        data_dicts: Optional list of data dictionaries with case paths
        logger: Optional logger instance for file logging
        log_gpu_mem: Whether to log GPU memory to file (default: False)
        data_dir: Optional dataset directory for loading class labels
        use_amp: Whether to use automatic mixed precision (FP16) for inference

    Returns:
        Dict mapping metric_name -> {'mean', 'std', 'min', 'max', 'all_scores', 'per_class'}
        'per_class' contains: {class_name: {'mean', 'std', 'min', 'max', 'all_scores'}} for each class
    """
    metrics_dict: dict[str, Any] = metric_fns

    # Load class labels if data_dir is provided
    class_labels: dict[int, str] | None = None
    include_background = False  # Default value
    if data_dir is not None and cfg is not None:
        # Check if include_background is set in the first metric config
        if "metrics" in cfg and len(cfg["metrics"]) > 0:
            include_background = cfg["metrics"][0].get("include_background", False)
        class_labels = get_class_labels(data_dir, include_background=include_background)

    model.eval()
    # Track scores per metric - will store either scalars or per-class arrays
    all_scores: dict[str, list] = {name: [] for name in metrics_dict.keys()}
    first_batch_saved = False
    batch_size = data_loader.batch_size if data_loader.batch_size else 1

    # Initialize inference strategy (sliding window or full-volume)
    inferer = None
    if cfg is not None and device is not None:
        try:
            inferer = create_inferer(cfg)
        except ValueError as e:
            warnings.warn(
                f"Failed to initialize sliding window inferer: {e}. Using full-volume inference."
            )
            inferer = None

    # Log GPU memory at start of evaluation
    if log_gpu_mem and logger is not None and device is not None:
        log_gpu_memory(logger, "Evaluation Start", device)

    with torch.no_grad():
        for batch_idx, data_item in enumerate(data_loader):
            inputs: torch.Tensor = data_item["image"].to(device)
            labels: torch.Tensor = data_item["label"].to(device)

            # Perform inference using appropriate strategy
            if inferer is not None and device is not None:
                outputs: torch.Tensor = inferer.infer(
                    model, inputs, device, use_amp=use_amp
                )
            else:
                # Fallback to direct forward pass
                if use_amp:
                    if device is None:
                        raise RuntimeError("AMP enabled but no device specified")
                    with autocast(device.type):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

            # Convert predictions to one-hot format for metrics that require it
            # Get class predictions first
            preds_argmax: torch.Tensor = torch.argmax(outputs, dim=1, keepdim=True)
            # Convert to one-hot: (B, num_classes, H, W, D)
            num_classes = outputs.shape[1]
            preds_onehot: torch.Tensor = one_hot(preds_argmax, num_classes=num_classes)

            # Also convert labels to one-hot format for metrics that require it
            labels_onehot: torch.Tensor = one_hot(labels, num_classes=num_classes)

            # Compute all metrics for this batch
            batch_scores = {}
            batch_scores_per_class = {}
            for name, metric in metrics_dict.items():
                metric(y_pred=preds_onehot, y=labels_onehot)
                result = metric.aggregate()

                # Check if result is per-class (tensor with multiple elements) or scalar
                if isinstance(result, torch.Tensor) and result.numel() > 1:
                    # Per-class scores
                    per_class_scores = result.cpu().numpy()
                    all_scores[name].append(per_class_scores)

                    # Compute mean across classes for batch display
                    mean_score = float(np.mean(per_class_scores))
                    batch_scores[name] = mean_score
                    batch_scores_per_class[name] = per_class_scores
                else:
                    # Scalar score (backward compatibility)
                    score: float = result.item()
                    all_scores[name].append(score)
                    batch_scores[name] = score

                metric.reset()

            if verbose:
                # Extract case path from data_dicts if available
                case_path = "unknown"
                if data_dicts is not None:
                    sample_idx = batch_idx * batch_size
                    if sample_idx < len(data_dicts):
                        image_path = data_dicts[sample_idx].get("image", "unknown")
                        case_path = Path(image_path).name

                # Print scores for all metrics
                for name, score in batch_scores.items():
                    if name in batch_scores_per_class and class_labels is not None:
                        # Per-class scores available
                        per_class = batch_scores_per_class[name]
                        class_scores_str = ", ".join(
                            [
                                f"{class_labels[idx]}: {per_class[i]:.4f}"
                                for i, idx in enumerate(sorted(class_labels.keys()))
                            ]
                        )
                        print(
                            f"{case_path} [{name}]: {class_scores_str}, Mean: {score:.4f}"
                        )

                        # Log to file with optional GPU memory
                        if logger is not None:
                            log_msg = f"Sample {batch_idx + 1}: {case_path} [{name}]: {class_scores_str}, Mean: {score:.4f}"
                            if log_gpu_mem and device is not None:
                                log_msg += get_gpu_memory_string(device)
                            logger.info(log_msg)
                    else:
                        # Scalar score (backward compatibility)
                        scores_str = ", ".join(
                            [f"{n} = {s:.4f}" for n, s in batch_scores.items()]
                        )
                        print(f"{case_path}: {scores_str}")

                        # Log to file with optional GPU memory in same line
                        if logger is not None:
                            log_msg = (
                                f"Sample {batch_idx + 1}: {case_path}: {scores_str}"
                            )
                            if log_gpu_mem and device is not None:
                                log_msg += get_gpu_memory_string(device)
                            logger.info(log_msg)

            # Save visualization of first batch only
            if (
                save_viz
                and batch_idx == 0
                and not first_batch_saved
                and save_dir
                and epoch is not None
                and cfg
            ):
                save_validation_visualizations(
                    images=inputs,
                    labels=labels,
                    predictions=preds_argmax,  # Use argmax for visualization
                    save_dir=save_dir,
                    epoch=epoch,
                    spatial_dims=cfg["model"]["spatial_dims"],
                )
                first_batch_saved = True

    # Compute statistics for each metric
    results = {}
    for name, scores in all_scores.items():
        if len(scores) > 0 and isinstance(scores[0], np.ndarray):
            # Per-class scores
            scores_array = np.array(scores)  # Shape: (num_samples, num_classes)

            # Overall statistics (mean across all classes and samples)
            all_values = scores_array.flatten()
            results[name] = {
                "mean": float(np.mean(all_values)),
                "std": float(np.std(all_values)),
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
                "all_scores": scores,
            }

            # Per-class statistics
            per_class_stats = {}
            num_classes = scores_array.shape[1]
            for class_idx in range(num_classes):
                class_scores = scores_array[:, class_idx]

                # Find the class name
                if class_labels is not None:
                    # Map array index to class label index
                    sorted_class_indices = sorted(class_labels.keys())
                    actual_class_idx = sorted_class_indices[class_idx]
                    class_name = class_labels[actual_class_idx]
                else:
                    class_name = f"Class {class_idx + 1}"

                per_class_stats[class_name] = {
                    "mean": float(np.mean(class_scores)),
                    "std": float(np.std(class_scores)),
                    "min": float(np.min(class_scores)),
                    "max": float(np.max(class_scores)),
                    "all_scores": class_scores.tolist(),
                }

            results[name]["per_class"] = per_class_stats
        else:
            # Scalar scores (backward compatibility)
            results[name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "all_scores": scores,
            }

    # Log GPU memory at end of evaluation
    if log_gpu_mem and logger is not None and device is not None:
        log_gpu_memory(logger, "Evaluation End", device)

    return results
