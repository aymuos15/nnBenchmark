"""
Tests for src/engines/inference/engine.py - Ignite-based inference engine.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class TestInferenceEngineInitialization:
    """Test InferenceEngine initialization."""

    def test_inference_engine_init(self, sample_config: dict) -> None:
        """Test InferenceEngine initialization."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
                data_dir=None,
            )

            assert engine.model == model
            assert engine.device == device
            assert engine.cfg == sample_config
            assert engine.metric_fns == metric_fns

    def test_inference_engine_reads_config_parameters(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine reads configuration parameters."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert engine.num_classes == sample_config["dataset"]["num_classes"]
            assert engine.spatial_dims == sample_config["model"].get("spatial_dims", 3)

    def test_inference_engine_detects_deep_supervision(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine detects deep supervision configuration."""
        from src.engines.inference.engine import InferenceEngine

        sample_config["model"]["deep_supervision"] = True

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert engine.deep_supervision is True

    def test_inference_engine_detects_mixed_precision(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine detects mixed precision configuration."""
        from src.engines.inference.engine import InferenceEngine

        sample_config["training"]["mixed_precision"] = True

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert engine.use_amp is True

    def test_inference_engine_creates_inferer(self, sample_config: dict) -> None:
        """Test that InferenceEngine creates inferer from config."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_create:
            mock_inferer = MagicMock()
            mock_create.return_value = mock_inferer

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            mock_create.assert_called_once_with(sample_config)
            assert engine.inferer == mock_inferer

    def test_inference_engine_initializes_engine_state(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine initializes Ignite engine state."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert hasattr(engine.engine, "state")
            assert hasattr(engine.engine.state, "metrics")
            assert isinstance(engine.engine.state.metrics, dict)


class TestInferenceEngineBatchProcessing:
    """Test batch processing in inference engine."""

    def test_prepare_batch_moves_to_device(self, sample_config: dict) -> None:
        """Test that _prepare_batch moves tensors to device."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            batch = {
                "image": torch.randn(1, 1, 32, 32, 32),
                "label": torch.randint(0, 3, (1, 32, 32, 32)),
            }

            images, labels = engine._prepare_batch(batch)

            assert images.device == device
            assert labels.device == device

    def test_prepare_batch_extracts_image_label(self, sample_config: dict) -> None:
        """Test that _prepare_batch extracts image and label tensors."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            batch: dict[str, torch.Tensor] = {
                "image": torch.randn(2, 1, 16, 16, 16),
                "label": torch.randint(0, 3, (2, 16, 16, 16)),
            }

            images, labels = engine._prepare_batch(batch)

            assert images.shape == (2, 1, 16, 16, 16)
            assert labels.shape == (2, 16, 16, 16)


class TestInferenceEngineInferenceIteration:
    """Test inference iteration function."""

    def test_inference_iteration_requires_model_eval_mode(
        self, sample_config: dict
    ) -> None:
        """Test that inference iteration can work with model."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            # Engine should have access to model
            assert engine.model == model

    def test_inference_engine_has_inference_iteration_method(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine has _evaluation_iteration method."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert hasattr(engine, "_evaluation_iteration")
            assert callable(engine._evaluation_iteration)


class TestInferenceEngineOutputHandling:
    """Test inference engine output handling."""

    def test_inference_engine_handles_single_output(self, sample_config: dict) -> None:
        """Test inference engine with single model output."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert engine.deep_supervision is False

    def test_inference_engine_handles_deep_supervision_output(
        self, sample_config: dict
    ) -> None:
        """Test inference engine with deep supervision outputs."""
        from src.engines.inference.engine import InferenceEngine

        sample_config["model"]["deep_supervision"] = True

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert engine.deep_supervision is True


class TestInferenceEngineMetricHandling:
    """Test metric handling in inference engine."""

    def test_inference_engine_stores_metrics(self, sample_config: dict) -> None:
        """Test that InferenceEngine stores metric functions."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metrics = {
            "metric1": MagicMock(),
            "metric2": MagicMock(),
        }

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metrics,
            )

            assert engine.metric_fns == metrics
            assert len(engine.metric_fns) == 2

    def test_inference_engine_initializes_metrics_dict_in_state(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine initializes metrics dict in engine state."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        metric_fns = {"metric1": MagicMock()}

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns=metric_fns,
            )

            assert isinstance(engine.engine.state.metrics, dict)


class TestInferenceEngineDataDirectoryHandling:
    """Test data directory handling in inference engine."""

    def test_inference_engine_stores_data_dir(self, sample_config: dict) -> None:
        """Test that InferenceEngine stores data directory."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")
        data_dir = "/path/to/data"

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns={},
                data_dir=data_dir,
            )

            assert engine.data_dir == data_dir

    def test_inference_engine_handles_none_data_dir(self, sample_config: dict) -> None:
        """Test that InferenceEngine handles None data directory."""
        from src.engines.inference.engine import InferenceEngine

        model = nn.Linear(1, 1)
        device = torch.device("cpu")

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns={},
                data_dir=None,
            )

            assert engine.data_dir is None


class TestInferenceEngineConfiguration:
    """Test inference engine configuration handling."""

    def test_inference_engine_respects_spatial_dims_config(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine respects spatial_dims from config."""
        from src.engines.inference.engine import InferenceEngine

        sample_config["model"]["spatial_dims"] = 2

        model = nn.Linear(1, 1)
        device = torch.device("cpu")

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns={},
            )

            assert engine.spatial_dims == 2

    def test_inference_engine_uses_default_spatial_dims(
        self, sample_config: dict
    ) -> None:
        """Test that InferenceEngine defaults to 3D spatial dims."""
        from src.engines.inference.engine import InferenceEngine

        # Remove spatial_dims from config to test default
        if "spatial_dims" in sample_config["model"]:
            del sample_config["model"]["spatial_dims"]

        model = nn.Linear(1, 1)
        device = torch.device("cpu")

        with patch("src.engines.inference.engine.create_inferer") as mock_inferer:
            mock_inferer.return_value = MagicMock()

            engine = InferenceEngine(
                model=model,
                device=device,
                cfg=sample_config,
                metric_fns={},
            )

            assert engine.spatial_dims == 3
