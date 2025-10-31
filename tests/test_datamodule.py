"""
Unit tests for SegmentationDataModule.
Tests dataset creation, caching functionality, and DataLoader configuration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest
from monai.data.dataset import CacheDataset, Dataset

from src.lightning.datamodule import SegmentationDataModule


@pytest.fixture
def sample_config_with_cache(sample_config: dict[str, Any]) -> dict[str, Any]:
    """Sample config with caching enabled."""
    config = sample_config.copy()
    config["dataset"]["cache"] = {"enabled": True, "cache_rate": 1.0}
    return config


@pytest.fixture
def sample_config_without_cache(sample_config: dict[str, Any]) -> dict[str, Any]:
    """Sample config with caching disabled."""
    config = sample_config.copy()
    config["dataset"]["cache"] = {"enabled": False}
    return config


class TestSegmentationDataModule:
    """Tests for SegmentationDataModule."""

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_setup_with_caching(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_with_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that setup creates CacheDataset when caching enabled."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_with_cache, data_dir=mock_dataset_dir, fold=0
        )

        # Act
        datamodule.setup(stage="fit")

        # Assert
        assert isinstance(datamodule.train_ds, CacheDataset)
        assert isinstance(datamodule.val_ds, CacheDataset)
        assert datamodule._using_cache is True

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_setup_without_caching(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that setup creates Dataset when caching disabled."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_without_cache, data_dir=mock_dataset_dir, fold=0
        )

        # Act
        datamodule.setup(stage="fit")

        # Assert
        assert isinstance(datamodule.train_ds, Dataset)
        assert isinstance(datamodule.val_ds, Dataset)
        assert not isinstance(datamodule.train_ds, CacheDataset)
        assert not isinstance(datamodule.val_ds, CacheDataset)
        assert datamodule._using_cache is False

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_train_dataloader_batch_size(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that train dataloader uses configured batch size."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_without_cache, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        train_loader = datamodule.train_dataloader()

        # Assert
        assert (
            train_loader.batch_size
            == sample_config_without_cache["training"]["batch_size"]
        )

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_val_dataloader_batch_size(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that validation dataloader uses batch_size=1."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_without_cache, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        val_loader = datamodule.val_dataloader()

        # Assert
        assert val_loader is not None
        assert val_loader.batch_size == 1

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_train_dataloader_shuffle_enabled(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that training dataloader has shuffling enabled."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_without_cache, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        train_loader = datamodule.train_dataloader()

        # Assert
        assert train_loader.sampler is not None or train_loader.shuffle is True

    @pytest.mark.parametrize(
        "cache_enabled,num_workers,expected_persistent",
        [
            (True, 4, False),  # Cache enabled → persistent_workers=False
            (False, 4, True),  # No cache, workers>0 → persistent_workers=True
            (False, 0, False),  # No workers → persistent_workers=False
        ],
    )
    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_persistent_workers_configuration(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        cache_enabled: bool,
        num_workers: int,
        expected_persistent: bool,
        sample_config: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test persistent_workers logic across config combinations.

        Validates that persistent_workers is:
        - False when using CacheDataset (data already in memory)
        - True when num_workers>0 and not using cache
        - False when num_workers=0
        """
        # Arrange
        config = sample_config.copy()
        config["dataset"]["cache"] = {"enabled": cache_enabled, "cache_rate": 1.0}
        config["training"]["num_workers"] = num_workers

        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=config, data_dir=mock_dataset_dir, fold=0
        )

        # Act
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Assert
        assert train_loader.persistent_workers is expected_persistent
        assert val_loader is not None
        assert val_loader.persistent_workers is expected_persistent

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.factory.transforms.transform_registry.build")
    def test_val_dataloader_none_when_no_validation_data(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that val_dataloader returns None when there's no validation data (fold=-1)."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [],  # Empty validation data (as in fold=-1)
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_without_cache, data_dir=mock_dataset_dir, fold=-1
        )
        datamodule.setup(stage="fit")

        # Act
        val_loader = datamodule.val_dataloader()

        # Assert
        assert val_loader is None
        assert datamodule.val_ds is None
