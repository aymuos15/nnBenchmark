"""Custom Ignite handlers for validation in nnBenchmark.

Provides event-driven handlers for metrics computation, progress tracking,
logging, results saving, and visualization during validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ignite.engine import Engine, Events

from src.engines.shared.handlers import BaseMetricsHandler, BaseProgressHandler
from src.plotting.validation import save_validation_visualizations
from src.utils.files import save_json


class ValidationMetricsHandler(BaseMetricsHandler):
    """Computes and logs metrics during validation.

    Uses simple scalar logging (all metrics in one line).
    """

    def _log_iteration_scores(self, engine: Engine) -> None:
        """Log scores for current iteration."""
        batch_idx = engine.state.iteration - 1
        batch_scores, _ = self._compute_batch_scores()

        if self.verbose:
            case_path = self._get_case_path(batch_idx)
            scores_str = ", ".join([f"{n} = {s:.4f}" for n, s in batch_scores.items()])
            print(f"{case_path}: {scores_str}")
            if self.logger is not None:
                log_msg = f"Sample {batch_idx + 1}: {case_path}: {scores_str}"
                self.logger.info(log_msg)


class ValidationProgressHandler(BaseProgressHandler):
    """Displays progress during validation."""

    context_name = "validation"


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
        all_results = engine.state.metrics
        metric_names = list(all_results.keys())

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

            # Add optional extended statistics
            for key in ["per_class", "bins", "per_sample_bins", "fp_tp_fn", "per_sample_fp_tp_fn"]:
                if key in results:
                    metric_summary[key] = results[key]

            summary[metric_name] = metric_summary

            # Convert per_sample_scores to JSON-serializable format
            metric_per_sample_scores = results["all_scores"]
            if len(metric_per_sample_scores) > 0 and isinstance(
                metric_per_sample_scores[0], np.ndarray
            ):
                metric_per_sample_scores = [score.tolist() for score in metric_per_sample_scores]
            per_sample_scores[metric_name] = metric_per_sample_scores

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

        history_dir = Path(self.results_dir) / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        if self.epoch is not None:
            validation_history_path = str(history_dir / f"validation_epoch_{self.epoch:03d}.json")
        else:
            validation_history_path = str(history_dir / "validation.json")

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
        if self.batches_saved >= self.save_first_n_batches:
            return

        outputs = engine.state.output
        images = outputs["images"]
        labels = outputs["labels"]
        preds = outputs["predictions"]

        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        save_validation_visualizations(
            images=images,
            labels=labels,
            predictions=preds,
            save_dir=str(viz_dir),
            epoch=self.epoch,
            spatial_dims=self.spatial_dims,
        )

        self.batches_saved += 1
