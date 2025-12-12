"""Custom Ignite handlers for inference in nnBenchmark.

Provides event-driven handlers for metrics computation, progress tracking,
logging, and results saving during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ignite.engine import Engine

from src.engines.shared.handlers import (
    BaseMetricsHandler,
    BaseProgressHandler,
    BaseResultsHandler,
)


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

            if has_per_class:
                for name, score in batch_scores.items():
                    if name in batch_scores_per_class:
                        per_class = batch_scores_per_class[name]
                        class_scores_str = ", ".join(
                            [
                                f"{self.class_labels[idx]}: {per_class[i]:.4f}"
                                for i, idx in enumerate(
                                    sorted(self.class_labels.keys())
                                )
                                if i < len(per_class)
                            ]
                        )
                        if self.logger is not None:
                            log_msg = f"Sample {batch_idx + 1}: {case_path} [{name}]: {class_scores_str}, Mean: {score:.4f}"
                            self.logger.info(log_msg)
                        else:
                            from loguru import logger

                            logger.info(
                                f"{case_path} [{name}]: {class_scores_str}, Mean: {score:.4f}"
                            )
            else:
                scores_str = ", ".join(
                    [f"{n} = {s:.4f}" for n, s in batch_scores.items()]
                )
                if self.logger is not None:
                    log_msg = f"Sample {batch_idx + 1}: {case_path}: {scores_str}"
                    self.logger.info(log_msg)
                else:
                    from loguru import logger

                    logger.info(f"{case_path}: {scores_str}")


class InferenceProgressHandler(BaseProgressHandler):
    """Displays progress during inference."""

    context_name = "inference"


class InferenceResultsHandler(BaseResultsHandler):
    """Saves inference results to test.json.

    Creates structured results file with metrics, per-sample scores,
    and configuration information specific to inference/testing.
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
        super().__init__(results_dir, config_name, cfg, fold, data_dicts)
        self.use_test_set = use_test_set
        self.model_path = model_path

    def _get_history_dict(
        self,
        summary: dict[str, Any],
        per_sample_scores: dict[str, Any],
        metric_names: list[str],
        sample_names: list[str],
    ) -> dict[str, Any]:
        """Build inference-specific history dictionary."""
        return {
            "config_name": self.config_name,
            "dataset_name": self.cfg["dataset"]["name"],
            "fold": self.fold,
            "use_test_set": self.use_test_set,
            "model_path": self.model_path,
            "metrics": metric_names,
            "summary": summary,
            "per_sample_scores": per_sample_scores,
            "sample_names": sample_names,
        }

    def _get_output_path(self) -> str:
        """Get inference output file path."""
        history_dir = Path(self.results_dir) / "history"
        return str(history_dir / "test.json")
