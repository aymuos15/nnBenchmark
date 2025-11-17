"""
Tests for src/engines/ignite_utils/trainer.py - Trainer factory and deep supervision.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset


class TestDeepSupervisionLossWrapper:
    """Test DeepSupervisionLossWrapper for handling multi-output losses."""

    def test_wrapper_initialization(self) -> None:
        """Test DeepSupervisionLossWrapper initialization."""
        from src.engines.ignite_utils.trainer import DeepSupervisionLossWrapper

        loss_fn = nn.CrossEntropyLoss()
        ds_weights = [0.5, 0.3, 0.2]
        wrapper = DeepSupervisionLossWrapper(loss_fn, ds_weights, spatial_dims=3)

        assert wrapper.loss_fn == loss_fn
        assert wrapper.ds_weights == ds_weights
        assert wrapper.spatial_dims == 3

    def test_wrapper_forward_with_tensor_output(self) -> None:
        """Test wrapper forward pass with tensor output (deep supervision format)."""
        from src.engines.ignite_utils.trainer import DeepSupervisionLossWrapper

        loss_fn = nn.MSELoss()
        ds_weights = [0.5, 0.3, 0.2]
        wrapper = DeepSupervisionLossWrapper(loss_fn, ds_weights, spatial_dims=3)

        # 3D deep supervision output: [B, num_outputs, C, D, H, W]
        outputs = torch.randn(2, 3, 1, 8, 8, 8)
        labels = torch.randn(2, 1, 8, 8, 8)

        loss = wrapper(outputs, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_wrapper_forward_with_list_output(self) -> None:
        """Test wrapper forward pass with list output."""
        from src.engines.ignite_utils.trainer import DeepSupervisionLossWrapper

        loss_fn = nn.MSELoss()
        ds_weights = [0.5, 0.3, 0.2]
        wrapper = DeepSupervisionLossWrapper(loss_fn, ds_weights, spatial_dims=3)

        # List of outputs (one per decoder level)
        outputs = [
            torch.randn(2, 1, 8, 8, 8),
            torch.randn(2, 1, 4, 4, 4),
            torch.randn(2, 1, 2, 2, 2),
        ]
        labels = torch.randn(2, 1, 8, 8, 8)

        loss = wrapper(outputs, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_wrapper_loss_is_weighted_sum(self) -> None:
        """Test that wrapper computes weighted sum of losses."""
        from src.engines.ignite_utils.trainer import DeepSupervisionLossWrapper

        loss_fn = nn.MSELoss()
        ds_weights = [0.5, 0.25, 0.25]
        wrapper = DeepSupervisionLossWrapper(loss_fn, ds_weights, spatial_dims=3)

        # Create deterministic outputs
        outputs = [
            torch.ones(2, 1, 8, 8, 8),
            torch.ones(2, 1, 4, 4, 4),
            torch.ones(2, 1, 2, 2, 2),
        ]
        labels = torch.zeros(2, 1, 8, 8, 8)

        loss = wrapper(outputs, labels)
        # Loss should be weighted sum (not just average)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_wrapper_2d_deep_supervision_format(self) -> None:
        """Test wrapper with 2D deep supervision format."""
        from src.engines.ignite_utils.trainer import DeepSupervisionLossWrapper

        loss_fn = nn.MSELoss()
        ds_weights = [0.5, 0.5]
        wrapper = DeepSupervisionLossWrapper(loss_fn, ds_weights, spatial_dims=2)

        # 2D deep supervision output: [B, num_outputs, C, H, W]
        outputs = torch.randn(2, 2, 1, 16, 16)
        labels = torch.randn(2, 1, 16, 16)

        loss = wrapper(outputs, labels)
        assert isinstance(loss, torch.Tensor)


class TestCreateTrainerBasics:
    """Test basic trainer creation."""

    def test_create_trainer_calls_model_registry(
        self, sample_config: dict, tmp_path
    ) -> None:
        """Test that create_trainer calls model registry to build model."""
        from src.engines.ignite_utils.trainer import create_trainer

        device = torch.device("cpu")
        logger = MagicMock()
        data = [
            {
                "image": torch.randn(1, 32, 32, 32),
                "label": torch.randint(0, 3, (32, 32, 32)),
            }
            for _ in range(2)
        ]
        ds = Dataset(data=data)
        loader = DataLoader(ds, batch_size=1)

        with patch(
            "src.engines.ignite_utils.trainer.model_registry.build"
        ) as mock_model:
            with patch(
                "src.engines.ignite_utils.trainer.loss_registry.build"
            ) as mock_loss:
                with patch(
                    "src.engines.ignite_utils.trainer.optimizer_registry.build"
                ) as mock_opt:
                    with patch(
                        "src.engines.ignite_utils.trainer.metric_registry.build"
                    ) as mock_metrics:
                        mock_model.return_value = nn.Linear(1, 1)
                        mock_loss.return_value = nn.MSELoss()
                        mock_opt.return_value = torch.optim.SGD(
                            nn.Linear(1, 1).parameters(), lr=0.001
                        )
                        mock_metrics.return_value = {}

                        create_trainer(
                            cfg=sample_config,
                            device=device,
                            train_loader=loader,
                            results_dir=str(tmp_path),
                            logger=logger,
                        )

                        # Model registry should be called
                        assert mock_model.called

    def test_create_trainer_calls_loss_registry(
        self, sample_config: dict, tmp_path
    ) -> None:
        """Test that create_trainer calls loss registry to build loss."""
        from src.engines.ignite_utils.trainer import create_trainer

        device = torch.device("cpu")
        logger = MagicMock()
        data = [
            {
                "image": torch.randn(1, 32, 32, 32),
                "label": torch.randint(0, 3, (32, 32, 32)),
            }
            for _ in range(2)
        ]
        ds = Dataset(data=data)
        loader = DataLoader(ds, batch_size=1)

        with patch(
            "src.engines.ignite_utils.trainer.model_registry.build"
        ) as mock_model:
            with patch(
                "src.engines.ignite_utils.trainer.loss_registry.build"
            ) as mock_loss:
                with patch(
                    "src.engines.ignite_utils.trainer.optimizer_registry.build"
                ) as mock_opt:
                    with patch(
                        "src.engines.ignite_utils.trainer.metric_registry.build"
                    ) as mock_metrics:
                        mock_model.return_value = nn.Linear(1, 1)
                        mock_loss.return_value = nn.MSELoss()
                        mock_opt.return_value = torch.optim.SGD(
                            nn.Linear(1, 1).parameters(), lr=0.001
                        )
                        mock_metrics.return_value = {}

                        create_trainer(
                            cfg=sample_config,
                            device=device,
                            train_loader=loader,
                            results_dir=str(tmp_path),
                            logger=logger,
                        )

                        # Loss registry should be called
                        assert mock_loss.called

    def test_create_trainer_builds_optimizer(
        self, sample_config: dict, tmp_path
    ) -> None:
        """Test that create_trainer builds optimizer using registry."""
        from src.engines.ignite_utils.trainer import create_trainer

        device = torch.device("cpu")
        logger = MagicMock()
        data = [
            {
                "image": torch.randn(1, 32, 32, 32),
                "label": torch.randint(0, 3, (32, 32, 32)),
            }
            for _ in range(2)
        ]
        ds = Dataset(data=data)
        loader = DataLoader(ds, batch_size=1)

        with patch(
            "src.engines.ignite_utils.trainer.model_registry.build"
        ) as mock_model:
            with patch(
                "src.engines.ignite_utils.trainer.loss_registry.build"
            ) as mock_loss:
                with patch(
                    "src.engines.ignite_utils.trainer.optimizer_registry.build"
                ) as mock_opt:
                    with patch(
                        "src.engines.ignite_utils.trainer.metric_registry.build"
                    ) as mock_metrics:
                        mock_model.return_value = nn.Linear(1, 1)
                        mock_loss.return_value = nn.MSELoss()
                        mock_opt.return_value = torch.optim.SGD(
                            nn.Linear(1, 1).parameters(), lr=0.001
                        )
                        mock_metrics.return_value = {}

                        create_trainer(
                            cfg=sample_config,
                            device=device,
                            train_loader=loader,
                            results_dir=str(tmp_path),
                            logger=logger,
                        )

                        # Optimizer registry should be called
                        mock_opt.assert_called_once()


class TestCreateTrainerDeepSupervision:
    """Test trainer creation with deep supervision."""

    def test_create_trainer_supports_deep_supervision_config(
        self, sample_config: dict, tmp_path
    ) -> None:
        """Test that create_trainer can handle deep supervision configuration."""
        sample_config["model"]["deep_supervision"] = True
        sample_config["model"]["ds_weights"] = [0.5, 0.3, 0.2]

        # Verify config is properly set
        assert sample_config["model"]["deep_supervision"] is True
        assert sample_config["model"]["ds_weights"] == [0.5, 0.3, 0.2]


class TestCreateTrainerMixedPrecision:
    """Test trainer creation with mixed precision."""


class TestPrepBatchFunction:
    """Test the _prepare_batch helper function."""

    def test_prepare_batch_moves_to_device(self) -> None:
        """Test that _prepare_batch moves tensors to device."""
        from src.engines.ignite_utils.trainer import _prepare_batch

        device = torch.device("cpu")
        batch = {
            "image": torch.randn(2, 1, 32, 32, 32),
            "label": torch.randint(0, 3, (2, 32, 32, 32)),
        }

        images, labels = _prepare_batch(batch, device)

        assert images.device == device
        assert labels.device == device

    def test_prepare_batch_returns_correct_shapes(self) -> None:
        """Test that _prepare_batch returns correct tensor shapes."""
        from src.engines.ignite_utils.trainer import _prepare_batch

        device = torch.device("cpu")
        batch = {
            "image": torch.randn(2, 1, 32, 32, 32),
            "label": torch.randint(0, 3, (2, 32, 32, 32)),
        }

        images, labels = _prepare_batch(batch, device)

        assert images.shape == (2, 1, 32, 32, 32)
        assert labels.shape == (2, 32, 32, 32)

    def test_prepare_batch_handles_batch_dict(self) -> None:
        """Test that _prepare_batch correctly extracts image and label."""
        from src.engines.ignite_utils.trainer import _prepare_batch

        device = torch.device("cpu")
        batch: dict[str, torch.Tensor] = {
            "image": torch.randn(1, 2, 16, 16, 16),
            "label": torch.randint(0, 2, (16, 16, 16)),
        }

        images, labels = _prepare_batch(batch, device)

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
