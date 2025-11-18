"""
Tests for src/engines/validate/handlers.py - Validation event handlers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from ignite.engine import Engine


class TestValidationMetricsHandler:
    """Test ValidationMetricsHandler for metrics computation and logging."""

    def test_metrics_handler_initialization(self) -> None:
        """Test ValidationMetricsHandler initializes correctly."""
        from src.engines.validate.handlers import ValidationMetricsHandler

        metric_fns = {"DiceMetric": MagicMock()}
        handler = ValidationMetricsHandler(
            metric_fns=metric_fns,
            logger=None,
            data_dir=None,
            include_background=False,
            verbose=True,
        )

        assert handler.metric_fns == metric_fns
        assert handler.verbose is True
        assert "DiceMetric" in handler.all_scores
        assert handler.all_scores["DiceMetric"] == []

    def test_metrics_handler_scalar_results(self) -> None:
        """Test that ValidationMetricsHandler handles scalar metric results."""
        from src.engines.validate.handlers import ValidationMetricsHandler

        # Create mock metric that returns scalar
        mock_metric = MagicMock()
        mock_metric.aggregate.return_value = torch.tensor(0.85)

        metric_fns = {"DiceMetric": mock_metric}
        handler = ValidationMetricsHandler(
            metric_fns=metric_fns, verbose=False, logger=None
        )

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.iteration = 1

        # Call iteration handler
        handler._log_iteration_scores(engine)

        # Check scalar was stored
        assert len(handler.all_scores["DiceMetric"]) == 1
        assert isinstance(handler.all_scores["DiceMetric"][0], float)
        assert abs(handler.all_scores["DiceMetric"][0] - 0.85) < 1e-6

    def test_metrics_handler_per_class_results(self) -> None:
        """Test that ValidationMetricsHandler handles per-class tensor results."""
        from src.engines.validate.handlers import ValidationMetricsHandler

        # Create mock metric that returns per-class scores
        mock_metric = MagicMock()
        mock_metric.aggregate.return_value = torch.tensor([0.80, 0.85, 0.90])

        metric_fns = {"DiceMetric": mock_metric}
        handler = ValidationMetricsHandler(
            metric_fns=metric_fns, verbose=False, logger=None
        )

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.iteration = 1

        # Call iteration handler
        handler._log_iteration_scores(engine)

        # Check per-class scores were stored as numpy array
        assert len(handler.all_scores["DiceMetric"]) == 1
        assert isinstance(handler.all_scores["DiceMetric"][0], np.ndarray)
        assert handler.all_scores["DiceMetric"][0].shape == (3,)
        np.testing.assert_array_almost_equal(
            handler.all_scores["DiceMetric"][0], [0.80, 0.85, 0.90]
        )

    def test_metrics_handler_per_class_statistics(self) -> None:
        """Test that ValidationMetricsHandler computes per-class statistics correctly."""
        from src.engines.validate.handlers import ValidationMetricsHandler

        # Create mock metric
        mock_metric = MagicMock()

        metric_fns = {"DiceMetric": mock_metric}
        handler = ValidationMetricsHandler(
            metric_fns=metric_fns, verbose=False, logger=None
        )

        # Manually add per-class scores for 3 samples, 2 classes
        handler.all_scores["DiceMetric"] = [
            np.array([0.8, 0.9]),
            np.array([0.7, 0.85]),
            np.array([0.75, 0.95]),
        ]

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.metrics = {}

        # Compute final metrics
        handler._compute_final_metrics(engine)

        # Check results stored in engine state
        assert "DiceMetric" in engine.state.metrics
        results = engine.state.metrics["DiceMetric"]

        # Check overall statistics
        assert "mean" in results
        assert "std" in results
        assert "min" in results
        assert "max" in results

        # Check per-class statistics
        assert "per_class" in results
        per_class = results["per_class"]
        assert len(per_class) == 2  # 2 classes

    def test_metrics_handler_with_class_labels(self, tmp_path: Path) -> None:
        """Test that ValidationMetricsHandler loads and uses class labels."""
        from src.engines.validate.handlers import ValidationMetricsHandler

        # Create mock dataset.json with labels
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        import json

        dataset_json = {
            "labels": {"0": "background", "1": "class1", "2": "class2"},
            "name": "TestDataset",
        }
        with open(dataset_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        # Create handler with data_dir
        metric_fns = {"DiceMetric": MagicMock()}
        handler = ValidationMetricsHandler(
            metric_fns=metric_fns,
            data_dir=str(dataset_dir),
            include_background=False,
            verbose=False,
        )

        # Check class labels loaded (excluding background)
        assert handler.class_labels is not None

    def test_metrics_handler_resets_metrics_after_iteration(self) -> None:
        """Test that ValidationMetricsHandler resets metrics after each iteration."""
        from src.engines.validate.handlers import ValidationMetricsHandler

        # Create mock metric
        mock_metric = MagicMock()
        mock_metric.aggregate.return_value = torch.tensor(0.85)

        metric_fns = {"DiceMetric": mock_metric}
        handler = ValidationMetricsHandler(
            metric_fns=metric_fns, verbose=False, logger=None
        )

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.iteration = 1

        # Call iteration handler
        handler._log_iteration_scores(engine)

        # Check that metric.reset() was called
        mock_metric.reset.assert_called_once()


class TestValidationProgressHandler:
    """Test ValidationProgressHandler for progress logging."""

    def test_progress_handler_initialization(self) -> None:
        """Test ValidationProgressHandler initializes correctly."""
        from src.engines.validate.handlers import ValidationProgressHandler

        handler = ValidationProgressHandler(
            logger=None, total_samples=10, data_dicts=None
        )

        assert handler.total_samples == 10

    def test_progress_handler_batch_logging(self, capsys) -> None:
        """Test that ValidationProgressHandler logs batch progress correctly."""
        from src.engines.validate.handlers import ValidationProgressHandler

        # Create data dicts with case paths
        data_dicts = [
            {"image": "/path/to/case_001.nii.gz"},
            {"image": "/path/to/case_002.nii.gz"},
        ]

        handler = ValidationProgressHandler(
            logger=None, total_samples=2, data_dicts=data_dicts
        )

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()

        # Test first iteration
        engine.state.iteration = 1
        handler._on_iteration_started(engine)

        captured = capsys.readouterr()
        assert "case_001.nii.gz" in captured.out
        assert "1/2" in captured.out or "50" in captured.out  # Progress

        # Test second iteration
        engine.state.iteration = 2
        handler._on_iteration_started(engine)

        captured = capsys.readouterr()
        assert "case_002.nii.gz" in captured.out

    def test_progress_handler_percentage_calculation(self) -> None:
        """Test that ValidationProgressHandler calculates percentages correctly."""
        from src.engines.validate.handlers import ValidationProgressHandler

        handler = ValidationProgressHandler(logger=None, total_samples=10)

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()

        # Test at 50% completion
        engine.state.iteration = 5
        handler._on_iteration_started(engine)

        # We can't easily test console output, but we can verify no errors


class TestValidationResultsHandler:
    """Test ValidationResultsHandler for saving results."""

    def test_results_handler_initialization(self, tmp_path: Path) -> None:
        """Test ValidationResultsHandler initializes correctly."""
        from src.engines.validate.handlers import ValidationResultsHandler

        cfg = {"dataset": {"name": "Dataset001_Test"}}
        handler = ValidationResultsHandler(
            results_dir=str(tmp_path),
            config_name="fold_0.yaml",
            cfg=cfg,
            fold=0,
            checkpoint_path="checkpoint.pt",
            epoch=1,
        )

        assert handler.results_dir == str(tmp_path)
        assert handler.epoch == 1

    def test_results_handler_json_serialization(self, tmp_path: Path) -> None:
        """Test that ValidationResultsHandler serializes numpy arrays to JSON."""
        from src.engines.validate.handlers import ValidationResultsHandler

        cfg = {"dataset": {"name": "Dataset001_Test"}}
        handler = ValidationResultsHandler(
            results_dir=str(tmp_path),
            config_name="fold_0.yaml",
            cfg=cfg,
            fold=0,
            checkpoint_path="checkpoint.pt",
            epoch=1,
        )

        # Create mock engine with numpy array results
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.metrics = {
            "DiceMetric": {
                "mean": 0.85,
                "std": 0.05,
                "min": 0.75,
                "max": 0.95,
                "per_class": {
                    "class1": {"mean": 0.80, "std": 0.04, "all_scores": [0.8, 0.81]},
                    "class2": {"mean": 0.90, "std": 0.03, "all_scores": [0.89, 0.91]},
                },
                "all_scores": [np.array([0.8, 0.9]), np.array([0.81, 0.91])],
            }
        }

        # Save results
        handler._save_results(engine)

        # Check file was created in history/ subdirectory
        result_file = tmp_path / "history" / "validation_epoch_001.json"
        assert result_file.exists()

        # Check JSON is valid and arrays converted to lists
        import json

        with open(result_file) as f:
            data = json.load(f)

        assert "summary" in data
        assert "DiceMetric" in data["summary"]
        assert "mean" in data["summary"]["DiceMetric"]
        assert data["summary"]["DiceMetric"]["mean"] == 0.85

    def test_results_handler_epoch_filename(self, tmp_path: Path) -> None:
        """Test that ValidationResultsHandler uses correct epoch number in filename."""
        from src.engines.validate.handlers import ValidationResultsHandler

        cfg = {"dataset": {"name": "Dataset001_Test"}}
        handler = ValidationResultsHandler(
            results_dir=str(tmp_path),
            config_name="fold_0.yaml",
            cfg=cfg,
            fold=0,
            checkpoint_path="checkpoint.pt",
            epoch=42,
        )

        # Create mock engine with state
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.metrics = {
            "DiceMetric": {
                "mean": 0.85,
                "std": 0.05,
                "min": 0.75,
                "max": 0.95,
                "all_scores": [0.8, 0.85, 0.9],
            }
        }

        # Save results
        handler._save_results(engine)

        # Check correct filename in history/ subdirectory
        result_file = tmp_path / "history" / "validation_epoch_042.json"
        assert result_file.exists()


class TestValidationVisualizationHandler:
    """Test ValidationVisualizationHandler for saving visualizations."""

    def test_visualization_handler_initialization(self, tmp_path: Path) -> None:
        """Test ValidationVisualizationHandler initializes correctly."""
        from src.engines.validate.handlers import ValidationVisualizationHandler

        handler = ValidationVisualizationHandler(
            results_dir=str(tmp_path),
            spatial_dims=3,
            epoch=1,
            save_first_n_batches=5,
        )

        assert handler.results_dir == Path(tmp_path)
        assert handler.save_first_n_batches == 5
        assert handler.batches_saved == 0

    def test_visualization_handler_first_n_batches_limit(self, tmp_path: Path) -> None:
        """Test that ValidationVisualizationHandler only saves first N batches."""
        from src.engines.validate.handlers import ValidationVisualizationHandler

        handler = ValidationVisualizationHandler(
            results_dir=str(tmp_path),
            spatial_dims=3,
            epoch=1,
            save_first_n_batches=2,
        )

        # Create mock engine with batch output
        engine = MagicMock(spec=Engine)
        engine.state = MagicMock()
        engine.state.batch = {
            "image": torch.randn(1, 1, 8, 8, 8),
            "label": torch.randint(0, 2, (1, 1, 8, 8, 8)),
        }
        engine.state.output = {
            "images": torch.randn(1, 1, 8, 8, 8),
            "labels": torch.randint(0, 2, (1, 1, 8, 8, 8)),
            "predictions": torch.randn(1, 2, 8, 8, 8),
        }

        # Mock visualization function
        with patch(
            "src.engines.validate.handlers.save_validation_visualizations"
        ) as mock_save:
            # Save batch 1 - should call visualization
            engine.state.iteration = 1
            handler._save_visualization(engine)
            assert mock_save.call_count == 1
            assert handler.batches_saved == 1

            # Save batch 2 - should call visualization
            engine.state.iteration = 2
            handler._save_visualization(engine)
            assert mock_save.call_count == 2
            assert handler.batches_saved == 2

            # Save batch 3 - should NOT call visualization (limit reached)
            engine.state.iteration = 3
            handler._save_visualization(engine)
            assert mock_save.call_count == 2  # Still 2, no new call
            assert handler.batches_saved == 2

            # Save batch 4 - should NOT call visualization
            engine.state.iteration = 4
            handler._save_visualization(engine)
            assert mock_save.call_count == 2
            assert handler.batches_saved == 2

    def test_visualization_handler_attaches_to_engine(self) -> None:
        """Test that ValidationVisualizationHandler attaches to correct event."""
        from src.engines.validate.handlers import ValidationVisualizationHandler

        handler = ValidationVisualizationHandler(
            results_dir="/tmp",
            spatial_dims=3,
            epoch=1,
            save_first_n_batches=5,
        )

        # Create mock engine
        engine = MagicMock(spec=Engine)

        # Attach handler
        handler.attach(engine)

        # Check that handler was attached to ITERATION_COMPLETED event
        engine.add_event_handler.assert_called()
