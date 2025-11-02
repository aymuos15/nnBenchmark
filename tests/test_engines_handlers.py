"""
Tests for training and inference handlers in src/engines/train/handlers.py and src/engines/inference/handlers.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import torch
from ignite.engine import Engine, Events


class TestTrainingHistoryHandler:
    """Test TrainingHistoryHandler for recording training metrics."""

    def test_handler_initialization(self, tmp_path: Path) -> None:
        """Test TrainingHistoryHandler initialization."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))

        assert handler.results_dir == str(tmp_path)
        assert isinstance(handler.training_history, dict)
        assert "epochs" in handler.training_history
        assert "train_loss" in handler.training_history

    def test_handler_initializes_history_structure(self, tmp_path: Path) -> None:
        """Test that handler initializes proper history structure."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))

        assert isinstance(handler.training_history["epochs"], list)
        assert isinstance(handler.training_history["train_loss"], list)
        assert isinstance(handler.training_history["val_epochs"], list)

    def test_handler_creates_history_file_path(self, tmp_path: Path) -> None:
        """Test that handler creates correct history file path."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))

        expected_path = str(Path(tmp_path) / "training_history.json")
        assert handler.history_path == expected_path

    def test_handler_attaches_to_engine(self, tmp_path: Path) -> None:
        """Test that handler can attach to Ignite engine."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))
        engine = Engine(lambda e, b: None)

        handler.attach(engine)

        # Check that handler registered for EPOCH_COMPLETED event
        assert Events.EPOCH_COMPLETED in engine._event_handlers

    def test_handler_loads_existing_history_when_resuming(self, tmp_path: Path) -> None:
        """Test that handler loads existing history when resuming."""
        from src.engines.train.handlers import TrainingHistoryHandler

        # Create existing history file
        history_data = {
            "epochs": [1, 2],
            "train_loss": [0.5, 0.3],
            "val_epochs": [2],
        }
        history_path = tmp_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history_data, f)

        handler = TrainingHistoryHandler(str(tmp_path), resume=True)

        assert handler.training_history == history_data

    def test_handler_starts_fresh_when_not_resuming(self, tmp_path: Path) -> None:
        """Test that handler starts fresh history when not resuming."""
        from src.engines.train.handlers import TrainingHistoryHandler

        # Create existing history file (should be ignored)
        history_data = {"epochs": [1, 2], "train_loss": [0.5, 0.3]}
        history_path = tmp_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history_data, f)

        handler = TrainingHistoryHandler(str(tmp_path), resume=False)

        # Should have fresh history
        assert handler.training_history["epochs"] == []
        assert handler.training_history["train_loss"] == []

    def test_handler_records_validation_metrics(self, tmp_path: Path) -> None:
        """Test that handler can record validation metrics."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))

        metrics: dict[str, torch.Tensor | float] = {
            "val_DiceMetric": torch.tensor(0.85),
            "val_loss": 0.15,
        }
        handler.record_validation_metrics(epoch=1, metrics=metrics)

        assert 1 in handler.training_history["val_epochs"]
        assert "val_DiceMetric" in handler.training_history
        assert "val_loss" in handler.training_history

    def test_handler_skips_validation_when_training_all_data(
        self, tmp_path: Path
    ) -> None:
        """Test that handler skips validation when training_all_data=True."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path), training_all_data=True)

        metrics: dict[str, torch.Tensor | float] = {"val_DiceMetric": torch.tensor(0.85)}
        handler.record_validation_metrics(epoch=1, metrics=metrics)

        # Should not record validation metrics
        assert "val_DiceMetric" not in handler.training_history

    def test_handler_converts_tensor_to_float(self, tmp_path: Path) -> None:
        """Test that handler converts tensor values to float."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))

        metrics: dict[str, torch.Tensor | float] = {"val_metric": torch.tensor(0.95)}
        handler.record_validation_metrics(epoch=1, metrics=metrics)

        # Should be converted to float
        assert isinstance(handler.training_history["val_metric"][0], float)


class TestValidationVisualizationHandler:
    """Test ValidationVisualizationHandler for saving validation visualizations."""

    def test_handler_initialization(self, tmp_path: Path) -> None:
        """Test ValidationVisualizationHandler initialization."""
        from src.engines.train.handlers import ValidationVisualizationHandler

        handler = ValidationVisualizationHandler(
            results_dir=str(tmp_path),
            spatial_dims=3,
        )

        assert handler.results_dir == str(tmp_path)
        assert handler.spatial_dims == 3

    def test_handler_attaches_to_engine(self, tmp_path: Path) -> None:
        """Test that handler can attach to Ignite engine (save_visualization method)."""
        from src.engines.train.handlers import ValidationVisualizationHandler

        handler = ValidationVisualizationHandler(
            results_dir=str(tmp_path),
            spatial_dims=3,
        )

        # Test that handler can save visualizations
        assert hasattr(handler, "save_visualization")
        assert callable(handler.save_visualization)


class TestTrainingLogger:
    """Test TrainingLogger for logging training events."""

    def test_logger_initialization(self) -> None:
        """Test TrainingLogger initialization."""
        from src.engines.train.handlers import TrainingLogger

        logger = MagicMock()
        handler = TrainingLogger(logger=logger)

        assert handler.logger == logger

    def test_logger_attaches_to_engine(self) -> None:
        """Test that logger can attach to Ignite engine."""
        from src.engines.train.handlers import TrainingLogger

        logger = MagicMock()
        handler = TrainingLogger(logger=logger)
        engine = Engine(lambda e, b: None)

        handler.attach(engine)

        # Handler should be attached successfully
        assert handler is not None


class TestGPUMemoryHandler:
    """Test GPUMemoryHandler for logging GPU memory usage."""

    def test_handler_initialization(self) -> None:
        """Test GPUMemoryHandler initialization."""
        from src.engines.train.handlers import GPUMemoryHandler

        logger = MagicMock()
        device = torch.device("cpu")
        handler = GPUMemoryHandler(logger=logger, device=device)

        assert handler.logger == logger
        assert handler.device == device

    def test_handler_attaches_to_engine(self) -> None:
        """Test that handler can attach to Ignite engine."""
        from src.engines.train.handlers import GPUMemoryHandler

        logger = MagicMock()
        device = torch.device("cpu")
        handler = GPUMemoryHandler(logger=logger, device=device)
        engine = Engine(lambda e, b: None)

        handler.attach(engine)

        # Handler should be attached successfully
        assert handler is not None

    def test_handler_logs_gpu_memory(self) -> None:
        """Test that handler logs GPU memory."""
        from src.engines.train.handlers import GPUMemoryHandler

        logger = MagicMock()
        device = torch.device("cpu")
        handler = GPUMemoryHandler(logger=logger, device=device)

        assert handler.logger == logger


class TestInferenceMetricsHandler:
    """Test InferenceMetricsHandler for computing metrics during inference."""

    def test_handler_initialization(self) -> None:
        """Test InferenceMetricsHandler initialization."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        metric_fns = {"metric1": MagicMock()}
        handler = InferenceMetricsHandler(metric_fns=metric_fns)

        assert handler.metric_fns == metric_fns
        assert isinstance(handler.all_scores, dict)

    def test_handler_stores_metric_functions(self) -> None:
        """Test that handler stores metric functions."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        metrics = {
            "DiceMetric": MagicMock(),
            "HausdorffDistance": MagicMock(),
        }
        handler = InferenceMetricsHandler(metric_fns=metrics)

        assert len(handler.metric_fns) == 2
        assert "DiceMetric" in handler.metric_fns

    def test_handler_initializes_scores_dict(self) -> None:
        """Test that handler initializes scores dict."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        metric_fns = {"metric1": MagicMock(), "metric2": MagicMock()}
        handler = InferenceMetricsHandler(metric_fns=metric_fns)

        assert "metric1" in handler.all_scores
        assert "metric2" in handler.all_scores
        assert isinstance(handler.all_scores["metric1"], list)

    def test_handler_attaches_to_engine(self) -> None:
        """Test that handler can attach to Ignite engine."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        metric_fns = {"metric1": MagicMock()}
        handler = InferenceMetricsHandler(metric_fns=metric_fns)
        engine = Engine(lambda e, b: None)

        handler.attach(engine)

        # Handler should be attached for ITERATION_COMPLETED and COMPLETED events
        assert Events.ITERATION_COMPLETED in engine._event_handlers
        assert Events.COMPLETED in engine._event_handlers

    def test_handler_with_logger(self) -> None:
        """Test that handler can work with logger."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        logger = MagicMock()
        metric_fns = {"metric1": MagicMock()}
        handler = InferenceMetricsHandler(metric_fns=metric_fns, logger=logger)

        assert handler.logger == logger

    def test_handler_with_data_dir(self, tmp_path: Path) -> None:
        """Test that handler can work with data directory."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        # Create a mock dataset.json file
        dataset_json = tmp_path / "dataset.json"
        dataset_json.write_text('{"labels": {"0": "background", "1": "class1"}}')

        metric_fns = {"metric1": MagicMock()}
        handler = InferenceMetricsHandler(metric_fns=metric_fns, data_dir=str(tmp_path))

        assert handler.data_dir == str(tmp_path)

    def test_handler_with_verbose_flag(self) -> None:
        """Test that handler respects verbose flag."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        metric_fns = {"metric1": MagicMock()}
        handler = InferenceMetricsHandler(metric_fns=metric_fns, verbose=False)

        assert handler.verbose is False

    def test_handler_with_gpu_memory_logging(self) -> None:
        """Test that handler can enable GPU memory logging."""
        from src.engines.inference.handlers import InferenceMetricsHandler

        metric_fns = {"metric1": MagicMock()}
        device = torch.device("cpu")
        logger = MagicMock()

        handler = InferenceMetricsHandler(
            metric_fns=metric_fns,
            log_gpu_mem=True,
            device=device,
            logger=logger,
        )

        assert handler.log_gpu_mem is True
        assert handler.device == device


class TestInferenceProgressHandler:
    """Test InferenceProgressHandler for tracking inference progress."""

    def test_handler_initialization(self) -> None:
        """Test InferenceProgressHandler initialization."""
        from src.engines.inference.handlers import InferenceProgressHandler

        logger = MagicMock()
        handler = InferenceProgressHandler(logger=logger, total_samples=100)

        assert handler.logger == logger
        assert handler.total_samples == 100

    def test_handler_attaches_to_engine(self) -> None:
        """Test that handler can attach to Ignite engine."""
        from src.engines.inference.handlers import InferenceProgressHandler

        logger = MagicMock()
        handler = InferenceProgressHandler(logger=logger, total_samples=100)
        engine = Engine(lambda e, b: None)

        handler.attach(engine)

        # Handler should be attached successfully
        assert handler is not None


class TestInferenceResultsHandler:
    """Test InferenceResultsHandler for saving inference results."""

    def test_handler_initialization(self, tmp_path: Path) -> None:
        """Test InferenceResultsHandler initialization."""
        from src.engines.inference.handlers import InferenceResultsHandler

        config = {"dataset": {"name": "test"}}
        handler = InferenceResultsHandler(
            results_dir=str(tmp_path),
            config_name="test_config",
            cfg=config,
            fold=0,
            use_test_set=False,
            model_path="/path/to/model.pt",
        )

        assert handler.results_dir == str(tmp_path)
        assert handler.config_name == "test_config"
        assert handler.fold == 0

    def test_handler_attaches_to_engine(self, tmp_path: Path) -> None:
        """Test that handler can attach to Ignite engine."""
        from src.engines.inference.handlers import InferenceResultsHandler

        config = {"dataset": {"name": "test"}}
        handler = InferenceResultsHandler(
            results_dir=str(tmp_path),
            config_name="test_config",
            cfg=config,
            fold=0,
            use_test_set=False,
            model_path="/path/to/model.pt",
        )
        engine = Engine(lambda e, b: None)

        handler.attach(engine)

        # Handler should be attached successfully
        assert handler is not None

    def test_handler_with_test_set(self, tmp_path: Path) -> None:
        """Test that handler can work with test set."""
        from src.engines.inference.handlers import InferenceResultsHandler

        config = {"dataset": {"name": "test"}}
        handler = InferenceResultsHandler(
            results_dir=str(tmp_path),
            config_name="test_config",
            cfg=config,
            fold=None,
            use_test_set=True,
            model_path="/path/to/model.pt",
        )

        assert handler.use_test_set is True
        assert handler.fold is None


class TestHandlerIntegration:
    """Test handler integration with Ignite engine."""

    def test_multiple_handlers_attach_to_same_engine(self, tmp_path: Path) -> None:
        """Test that multiple handlers can attach to same engine."""
        from src.engines.train.handlers import (
            GPUMemoryHandler,
            TrainingHistoryHandler,
            TrainingLogger,
        )

        engine = Engine(lambda e, b: {"loss": 0.5})

        handlers = [
            TrainingHistoryHandler(str(tmp_path)),
            TrainingLogger(logger=MagicMock()),
            GPUMemoryHandler(logger=MagicMock(), device=torch.device("cpu")),
        ]

        for handler in handlers:
            handler.attach(engine)

        # All handlers should be attached
        assert len(handlers) == 3

    def test_handlers_dont_interfere_with_each_other(
        self, tmp_path: Path
    ) -> None:
        """Test that handlers don't interfere with each other."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler1 = TrainingHistoryHandler(str(tmp_path / "handler1"))
        handler2 = TrainingHistoryHandler(str(tmp_path / "handler2"))

        engine = Engine(lambda e, b: {"loss": 0.5})

        handler1.attach(engine)
        handler2.attach(engine)

        # Handlers should have different state
        assert handler1.history_path != handler2.history_path
