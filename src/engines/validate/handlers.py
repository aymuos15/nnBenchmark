"""Custom Ignite handlers for validation in nnBenchmark.

Provides event-driven handlers for metrics computation, progress tracking,
logging, results saving, and visualization during validation.
"""


from pathlib import Path
from typing import Any

from ignite.engine import Engine, Events

from src.engines.shared.handlers import (
    BaseMetricsHandler,
    BaseProgressHandler,
    BaseResultsHandler,
)
from src.plotting.validation import save_validation_visualizations


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
            if self.logger is not None:
                log_msg = f"Sample {batch_idx + 1}: {case_path}: {scores_str}"
                self.logger.info(log_msg)
            else:
                from loguru import logger

                logger.info(f"{case_path}: {scores_str}")


class ValidationProgressHandler(BaseProgressHandler):
    """Displays progress during validation."""

    context_name = "validation"


class ValidationResultsHandler(BaseResultsHandler):
    """Saves validation results to validation_history.json.

    Creates structured results file with metrics, per-sample scores,
    and configuration information specific to validation.
    """

    def __init__(
        self,
        results_dir: str,
        config_name: str,
        cfg: Any,
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
        super().__init__(results_dir, config_name, cfg, fold, data_dicts)
        self.checkpoint_path = checkpoint_path
        self.epoch = epoch

    def _get_history_dict(
        self,
        summary: dict[str, Any],
        per_sample_scores: dict[str, Any],
        metric_names: list[str],
        sample_names: list[str],
    ) -> dict[str, Any]:
        """Build validation-specific history dictionary."""
        return {
            "config_name": self.config_name,
            "dataset_name": self.cfg["dataset"]["name"],
            "fold": self.fold,
            "checkpoint_path": self.checkpoint_path,
            "epoch": self.epoch,
            "metrics": metric_names,
            "summary": summary,
            "per_sample_scores": per_sample_scores,
            "sample_names": sample_names,
        }

    def _get_output_path(self) -> str:
        """Get validation output file path."""
        history_dir = Path(self.results_dir) / "history"
        if self.epoch is not None:
            return str(history_dir / f"validation_epoch_{self.epoch:03d}.json")
        return str(history_dir / "validation.json")


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

        outputs = engine.state.output  # type: ignore[attr-defined]
        images = outputs["images"]  # type: ignore[index]
        labels = outputs["labels"]  # type: ignore[index]
        preds = outputs["predictions"]  # type: ignore[index]

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
