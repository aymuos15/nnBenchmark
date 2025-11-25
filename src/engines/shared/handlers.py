"""Shared base handlers for inference and validation engines.

Provides common functionality to eliminate code duplication between
inference and validation handler implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ignite.engine import Engine, Events

from src.utils.data import get_class_labels

if TYPE_CHECKING:
    from loguru._logger import Logger


class BaseMetricsHandler(ABC):
    """Base class for metrics computation handlers.

    Provides shared logic for accumulating per-sample scores and computing
    summary statistics. Subclasses implement context-specific logging.
    """

    def __init__(
        self,
        metric_fns: dict[str, Any],
        logger: Logger | None = None,
        data_dir: str | None = None,
        include_background: bool = False,
        verbose: bool = True,
        device: torch.device | None = None,
        data_dicts: list[dict[str, str]] | None = None,
    ):
        """Initialize BaseMetricsHandler.

        Args:
            metric_fns: Dictionary of metric functions {name: metric_fn}
            logger: Optional logger instance for file logging
            data_dir: Optional dataset directory for loading class labels
            include_background: Whether metrics include background class
            verbose: Whether to print per-case scores to console
            device: Device for GPU memory logging
            data_dicts: Optional list of data dictionaries with case paths
        """
        self.metric_fns = metric_fns
        self.logger = logger
        self.data_dir = data_dir
        self.verbose = verbose
        self.device = device
        self.data_dicts = data_dicts

        # Load class labels if available
        self.class_labels: dict[int, str] | None = None
        if data_dir is not None:
            self.class_labels = get_class_labels(
                data_dir, include_background=include_background
            )

        # Track scores per metric
        self.all_scores: dict[str, list] = {name: [] for name in metric_fns.keys()}

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.STARTED, self._reset_instance_scores)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._log_iteration_scores)
        engine.add_event_handler(Events.COMPLETED, self._compute_final_metrics)

    def _reset_instance_scores(self, engine: Engine) -> None:
        """Reset instance scores for CCMetrics at start."""
        for metric in self.metric_fns.values():
            if hasattr(metric, "reset_instance_scores"):
                metric.reset_instance_scores()

    @abstractmethod
    def _log_iteration_scores(self, engine: Engine) -> None:
        """Log scores for current iteration. Subclasses implement specific logging."""
        raise NotImplementedError

    def _compute_batch_scores(self) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Compute and store batch scores from accumulated metrics.

        Returns:
            Tuple of (batch_scores dict, batch_scores_per_class dict)
        """
        batch_scores: dict[str, float] = {}
        batch_scores_per_class: dict[str, np.ndarray] = {}

        for name, metric in self.metric_fns.items():
            result = metric.aggregate()

            if isinstance(result, torch.Tensor) and result.numel() > 1:
                per_class_scores = result.cpu().numpy()
                self.all_scores[name].append(per_class_scores)
                batch_scores[name] = float(np.mean(per_class_scores))
                batch_scores_per_class[name] = per_class_scores
            else:
                score: float = result.item()
                self.all_scores[name].append(score)
                batch_scores[name] = score

            metric.reset()

        return batch_scores, batch_scores_per_class

    def _get_case_path(self, batch_idx: int) -> str:
        """Extract case path from data_dicts if available."""
        if self.data_dicts is not None and batch_idx < len(self.data_dicts):
            image_path = self.data_dicts[batch_idx].get("image", "unknown")
            return Path(image_path).name
        return "unknown"

    def _compute_final_metrics(self, engine: Engine) -> None:
        """Compute final statistics from all scores."""
        results = {}

        for name, scores in self.all_scores.items():
            if len(scores) > 0 and isinstance(scores[0], np.ndarray):
                scores_array = np.array(scores)
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
                    if self.class_labels is not None:
                        sorted_class_indices = sorted(self.class_labels.keys())
                        actual_class_idx = sorted_class_indices[class_idx]
                        class_name = self.class_labels[actual_class_idx]
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
                results[name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "all_scores": scores,
                }

            # Get extended statistics from metrics that support them
            if name in self.metric_fns:
                metric = self.metric_fns[name]
                self._add_extended_statistics(results[name], metric)

        engine.state.metrics = results

    def _add_extended_statistics(self, result: dict, metric: Any) -> None:
        """Add extended statistics (bins, FP/TP/FN) if metric supports them."""
        stat_methods = [
            ("get_binned_statistics", "bins"),
            ("get_per_sample_binned_statistics", "per_sample_bins"),
            ("get_fp_tp_fn_statistics", "fp_tp_fn"),
            ("get_per_sample_fp_tp_fn_statistics", "per_sample_fp_tp_fn"),
        ]
        for method_name, key in stat_methods:
            if hasattr(metric, method_name):
                try:
                    result[key] = getattr(metric, method_name)()
                except Exception:
                    pass


class BaseProgressHandler:
    """Base class for progress tracking handlers.

    Provides shared logic for displaying progress during evaluation.
    """

    # Subclasses set this to "inference" or "validation"
    context_name: str = "evaluation"

    def __init__(
        self,
        logger: Logger | None = None,
        total_samples: int | None = None,
        data_dicts: list[dict[str, str]] | None = None,
    ):
        """Initialize BaseProgressHandler.

        Args:
            logger: Optional logger instance
            total_samples: Total number of samples (for progress percentage)
            data_dicts: Optional list of data dictionaries with case paths
        """
        self.logger = logger
        self.total_samples = total_samples
        self.data_dicts = data_dicts

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.STARTED, self._on_started)
        engine.add_event_handler(Events.ITERATION_STARTED, self._on_iteration_started)
        engine.add_event_handler(Events.COMPLETED, self._on_completed)

    def _on_started(self, engine: Engine) -> None:
        """Log start."""
        if self.logger is not None:
            self.logger.info("=" * 50)
            self.logger.info(f"Starting {self.context_name}...")
            if self.total_samples is not None:
                self.logger.info(f"Total samples: {self.total_samples}")
            self.logger.info("=" * 50)

    def _on_iteration_started(self, engine: Engine) -> None:
        """Log iteration start."""
        batch_idx = engine.state.iteration - 1

        case_path = "unknown"
        if self.data_dicts is not None and batch_idx < len(self.data_dicts):
            image_path = self.data_dicts[batch_idx].get("image", "unknown")
            case_path = Path(image_path).name

        if self.total_samples is not None:
            progress = (batch_idx + 1) / self.total_samples * 100
            msg = f"[{batch_idx + 1}/{self.total_samples}] ({progress:.1f}%) Processing: {case_path}"
        else:
            msg = f"[{batch_idx + 1}] Processing: {case_path}"

        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def _on_completed(self, engine: Engine) -> None:
        """Log completion."""
        completion_msg = f"{self.context_name.capitalize()} completed!"
        print("\n" + "=" * 50)
        print(completion_msg)
        print("=" * 50)

        if self.logger is not None:
            self.logger.info("=" * 50)
            self.logger.info(completion_msg)
            self.logger.info("=" * 50)
