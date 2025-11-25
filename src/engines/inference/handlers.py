"""Custom Ignite handlers for inference in nnBenchmark.

Provides event-driven handlers for metrics computation, progress tracking,
logging, and results saving during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ignite.engine import Engine, Events

from src.engines.shared.handlers import BaseMetricsHandler, BaseProgressHandler
from src.utils.files import save_json


class InferenceMetricsHandler(BaseMetricsHandler):
    """Computes and logs metrics during inference.

    Extends BaseMetricsHandler with inference-specific per-class logging.
    """

    def _log_iteration_scores(self, engine: Engine) -> None:
        """Log scores for current iteration with per-class detail."""
        batch_idx = engine.state.iteration - 1
        batch_scores, batch_scores_per_class = self._compute_batch_scores()

        if self.verbose:
            case_path = self._get_case_path(batch_idx)
            has_per_class = (
                len(batch_scores_per_class) > 0 and self.class_labels is not None
            )

            if has_per_class and self.class_labels is not None:
                for name, score in batch_scores.items():
                    if name in batch_scores_per_class:
                        per_class = batch_scores_per_class[name]
                        class_scores_str = ", ".join(
                            [
                                f"{self.class_labels[idx]}: {per_class[i]:.4f}"
                                for i, idx in enumerate(
                                    sorted(self.class_labels.keys())
                                )
                            ]
                        )
                        print(
                            f"{case_path} [{name}]: {class_scores_str}, Mean: {score:.4f}"
                        )
                        if self.logger is not None:
                            log_msg = f"Sample {batch_idx + 1}: {case_path} [{name}]: {class_scores_str}, Mean: {score:.4f}"
                            self.logger.info(log_msg)
            else:
                scores_str = ", ".join(
                    [f"{n} = {s:.4f}" for n, s in batch_scores.items()]
                )
                print(f"{case_path}: {scores_str}")
                if self.logger is not None:
                    log_msg = f"Sample {batch_idx + 1}: {case_path}: {scores_str}"
                    self.logger.info(log_msg)


class InferenceProgressHandler(BaseProgressHandler):
    """Displays progress during inference."""

    context_name = "inference"


class InferenceResultsHandler:
    """Saves inference results to test_history.json.

    Creates structured results file with metrics, per-sample scores,
    and configuration information.
    """

    def __init__(
        self,
        results_dir: str,
        config_name: str,
        cfg: dict[str, Any],
        fold: int | None,
        use_test_set: bool,
        model_path: str,
        data_dicts: list[dict[str, str]] | None = None,
    ):
        """Initialize InferenceResultsHandler.

        Args:
            results_dir: Directory to save results
            config_name: Name of configuration file
            cfg: Configuration dictionary
            fold: Fold number (None if using test set)
            use_test_set: Whether using dedicated test set
            model_path: Path to model checkpoint
            data_dicts: Optional list of data dictionaries with case paths
        """
        self.results_dir = results_dir
        self.config_name = config_name
        self.cfg = cfg
        self.fold = fold
        self.use_test_set = use_test_set
        self.model_path = model_path
        self.data_dicts = data_dicts

    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.COMPLETED, self._save_results)

    def _save_results(self, engine: Engine) -> None:
        """Save results to test_history.json."""
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
            for key in [
                "per_class",
                "bins",
                "per_sample_bins",
                "fp_tp_fn",
                "per_sample_fp_tp_fn",
            ]:
                if key in results:
                    metric_summary[key] = results[key]

            summary[metric_name] = metric_summary

            # Convert per_sample_scores to JSON-serializable format
            metric_per_sample_scores = results["all_scores"]
            if len(metric_per_sample_scores) > 0 and isinstance(
                metric_per_sample_scores[0], np.ndarray
            ):
                metric_per_sample_scores = [
                    score.tolist() for score in metric_per_sample_scores
                ]
            per_sample_scores[metric_name] = metric_per_sample_scores

        test_history = {
            "config_name": self.config_name,
            "dataset_name": self.cfg["dataset"]["name"],
            "fold": self.fold,
            "use_test_set": self.use_test_set,
            "model_path": self.model_path,
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
        test_history_path = str(history_dir / "test.json")
        save_json(test_history, test_history_path)

        print(f"\nResults saved to: {self.results_dir}")
        print(f"Test history: {test_history_path}")
