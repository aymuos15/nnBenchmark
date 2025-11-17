"""Custom Ignite handlers for validation in nnBenchmark.

Provides event-driven handlers for metrics computation, progress tracking,
logging, results saving, and visualization during validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ignite.engine import Engine, Events

from src.plotting.validation import save_validation_visualizations
from src.utils.data import get_class_labels
from src.utils.files import save_json

if TYPE_CHECKING:
    from loguru._logger import Logger


class ValidationMetricsHandler:
    """Computes and logs metrics during validation.

    Accumulates per-sample scores during validation and computes
    summary statistics at the end.
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
        """Initialize ValidationMetricsHandler.

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
        # Log per-sample scores after each iteration
        engine.add_event_handler(
            Events.ITERATION_COMPLETED, self._log_iteration_scores
        )

        # Compute final statistics at end of validation
        engine.add_event_handler(Events.COMPLETED, self._compute_final_metrics)

    def _log_iteration_scores(self, engine: Engine) -> None:
        """Log scores for current iteration."""
        batch_idx = engine.state.iteration - 1

        # Compute batch scores from accumulated metrics
        batch_scores = {}
        batch_scores_per_class = {}

        for name, metric in self.metric_fns.items():
            result = metric.aggregate()

            # Check if result is per-class or scalar
            if isinstance(result, torch.Tensor) and result.numel() > 1:
                # Per-class scores
                per_class_scores = result.cpu().numpy()
                self.all_scores[name].append(per_class_scores)

                # Compute mean across classes for display
                mean_score = float(np.mean(per_class_scores))
                batch_scores[name] = mean_score
                batch_scores_per_class[name] = per_class_scores
            else:
                # Scalar score
                score: float = result.item()
                self.all_scores[name].append(score)
                batch_scores[name] = score

            # Reset metric for next sample
            metric.reset()

        if self.verbose:
            # Extract case path from data_dicts if available
            case_path = "unknown"
            if self.data_dicts is not None and batch_idx < len(self.data_dicts):
                image_path = self.data_dicts[batch_idx].get("image", "unknown")
                case_path = Path(image_path).name

            # Build a single log message with all metrics
            scores_str = ", ".join(
                [f"{n} = {s:.4f}" for n, s in batch_scores.items()]
            )
            print(f"{case_path}: {scores_str}")

            # Log to file
            if self.logger is not None:
                log_msg = f"Sample {batch_idx + 1}: {case_path}: {scores_str}"
                self.logger.info(log_msg)

    def _compute_final_metrics(self, engine: Engine) -> None:
        """Compute final statistics from all scores."""
        results = {}

        for name, scores in self.all_scores.items():
            if len(scores) > 0 and isinstance(scores[0], np.ndarray):
                # Per-class scores
                scores_array = np.array(scores)  # Shape: (num_cases, num_classes)

                # Overall statistics (mean across all classes and cases)
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
                # Scalar scores
                results[name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "all_scores": scores,
                }

        # Store results in engine state for other handlers
        engine.state.metrics = results


class ValidationProgressHandler:
    """Displays progress during validation.

    Shows sample-by-sample progress with case names.
    """

    def __init__(
        self,
        logger: Logger | None = None,
        total_samples: int | None = None,
        data_dicts: list[dict[str, str]] | None = None,
    ):
        """Initialize ValidationProgressHandler.

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
        """Log validation start."""
        if self.logger is not None:
            self.logger.info("=" * 50)
            self.logger.info("Starting validation...")
            if self.total_samples is not None:
                self.logger.info(f"Total samples: {self.total_samples}")
            self.logger.info("=" * 50)

    def _on_iteration_started(self, engine: Engine) -> None:
        """Log iteration start."""
        batch_idx = engine.state.iteration - 1

        # Extract case path if available
        case_path = "unknown"
        if self.data_dicts is not None and batch_idx < len(self.data_dicts):
            image_path = self.data_dicts[batch_idx].get("image", "unknown")
            case_path = Path(image_path).name

        # Compute progress
        if self.total_samples is not None:
            progress = (batch_idx + 1) / self.total_samples * 100
            msg = f"[{batch_idx + 1}/{self.total_samples}] ({progress:.1f}%) Processing: {case_path}"
        else:
            msg = f"[{batch_idx + 1}] Processing: {case_path}"

        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def _on_completed(self, engine: Engine) -> None:
        """Log validation completion."""
        print("\n" + "=" * 50)
        print("Validation completed!")
        print("=" * 50)

        if self.logger is not None:
            self.logger.info("=" * 50)
            self.logger.info("Validation completed!")
            self.logger.info("=" * 50)


class ValidationResultsHandler:
    """Saves validation results to validation_history.json.

    Creates structured results file with metrics, per-sample scores,
    and configuration information.
    """

    def __init__(
        self,
        results_dir: str,
        config_name: str,
        cfg: dict[str, Any],
        fold: int | None,
        checkpoint_path: str,
        epoch: int | None,
        data_dicts: list[dict[str, str]] | None = None,
    ):
        """Initialize ValidationResultsHandler.

        Args:
            results_dir: Directory to save results
            config_name: Name of configuration file
            cfg: Configuration dictionary
            fold: Fold number
            checkpoint_path: Path to model checkpoint
            epoch: Epoch number from checkpoint
            data_dicts: Optional list of data dictionaries with case paths
        """
        self.results_dir = results_dir
        self.config_name = config_name
        self.cfg = cfg
        self.fold = fold
        self.checkpoint_path = checkpoint_path
        self.epoch = epoch
        self.data_dicts = data_dicts

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.COMPLETED, self._save_results)

    def _save_results(self, engine: Engine) -> None:
        """Save results to validation_history.json."""
        # Get metrics from engine state (set by ValidationMetricsHandler)
        all_results = engine.state.metrics

        metric_names = list(all_results.keys())

        # Prepare summary statistics for all metrics
        summary = {}
        per_sample_scores = {}

        for metric_name, results in all_results.items():
            metric_summary = {
                "mean": results["mean"],
                "std": results["std"],
                "min": results["min"],
                "max": results["max"],
                "num_cases": len(results["all_scores"]),
            }

            # Add per-class statistics if available
            if "per_class" in results:
                metric_summary["per_class"] = results["per_class"]

            summary[metric_name] = metric_summary

            # Convert per_sample_scores to JSON-serializable format
            metric_per_sample_scores = results["all_scores"]
            if len(metric_per_sample_scores) > 0 and isinstance(
                metric_per_sample_scores[0], np.ndarray
            ):
                # Per-class scores - convert numpy arrays to lists
                metric_per_sample_scores = [
                    score.tolist() for score in metric_per_sample_scores
                ]

            per_sample_scores[metric_name] = metric_per_sample_scores

        # Create validation history JSON
        validation_history = {
            "config_name": self.config_name,
            "dataset_name": self.cfg["dataset"]["name"],
            "fold": self.fold,
            "checkpoint_path": self.checkpoint_path,
            "epoch": self.epoch,
            "metrics": metric_names,
            "summary": summary,
            "per_sample_scores": per_sample_scores,
            "sample_names": (
                [Path(d.get("image", "unknown")).name for d in self.data_dicts]
                if self.data_dicts
                else []
            ),
        }

        # Save validation history JSON with epoch number
        if self.epoch is not None:
            validation_history_path = str(
                Path(self.results_dir) / f"validation_history_epoch_{self.epoch:03d}.json"
            )
        else:
            validation_history_path = str(
                Path(self.results_dir) / "validation_history.json"
            )

        save_json(validation_history, validation_history_path)

        print(f"\nResults saved to: {self.results_dir}")
        print(f"Validation history: {validation_history_path}")


class ValidationVisualizationHandler:
    """Saves validation visualizations (input, label, prediction images).

    Generates side-by-side visualization plots for the first batch
    of validation data.
    """

    def __init__(
        self,
        results_dir: str,
        spatial_dims: int,
        epoch: int,
        save_first_n_batches: int = 1,
    ):
        """Initialize ValidationVisualizationHandler.

        Args:
            results_dir: Directory to save visualizations
            spatial_dims: Number of spatial dimensions (2 or 3)
            epoch: Epoch number for naming visualizations
            save_first_n_batches: Number of initial batches to visualize
        """
        self.results_dir = Path(results_dir)
        self.spatial_dims = spatial_dims
        self.epoch = epoch
        self.save_first_n_batches = save_first_n_batches
        self.batches_saved = 0

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._save_visualization)

    def _save_visualization(self, engine: Engine) -> None:
        """Save visualization for current batch."""
        # Only save first N batches
        if self.batches_saved >= self.save_first_n_batches:
            return

        # Get outputs from current iteration
        outputs = engine.state.output

        images = outputs["images"]
        labels = outputs["labels"]
        preds = outputs["predictions"]

        # Create visualizations directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Save visualization
        save_validation_visualizations(
            images=images,
            labels=labels,
            predictions=preds,
            save_dir=str(viz_dir),
            epoch=self.epoch,
            spatial_dims=self.spatial_dims,
        )

        self.batches_saved += 1
