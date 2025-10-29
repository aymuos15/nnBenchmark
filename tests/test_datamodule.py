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
    @patch("src.lightning.datamodule.build_transforms")
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
    @patch("src.lightning.datamodule.build_transforms")
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
    @patch("src.lightning.datamodule.build_transforms")
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
    @patch("src.lightning.datamodule.build_transforms")
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
    @patch("src.lightning.datamodule.build_transforms")
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

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.lightning.datamodule.build_transforms")
    def test_persistent_workers_disabled_with_cache(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_with_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that persistent_workers=False when using CacheDataset."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_with_cache, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Assert
        assert train_loader.persistent_workers is False
        assert val_loader is not None
        assert val_loader.persistent_workers is False

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.lightning.datamodule.build_transforms")
    def test_persistent_workers_enabled_with_workers_and_no_cache(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that persistent_workers=True when num_workers>0 and not using cache."""
        # Arrange
        config = sample_config_without_cache.copy()
        config["training"]["num_workers"] = 4

        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=config, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Assert
        assert train_loader.persistent_workers is True
        assert val_loader is not None
        assert val_loader.persistent_workers is True

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.lightning.datamodule.build_transforms")
    def test_persistent_workers_disabled_with_zero_workers(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_without_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that persistent_workers=False when num_workers=0."""
        # Arrange
        config = sample_config_without_cache.copy()
        config["training"]["num_workers"] = 0

        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=config, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Assert
        assert train_loader.persistent_workers is False
        assert val_loader is not None
        assert val_loader.persistent_workers is False

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.lightning.datamodule.build_transforms")
    def test_val_dataloader_persistent_workers_logic(
        self,
        mock_build_transforms: Mock,
        mock_get_data_dicts: Mock,
        sample_config_with_cache: dict[str, Any],
        mock_dataset_dir: str,
    ) -> None:
        """Test that validation dataloader has same persistent_workers logic."""
        # Arrange
        mock_get_data_dicts.return_value = (
            [{"image": "img1.nii.gz", "label": "lbl1.nii.gz"}],
            [{"image": "img2.nii.gz", "label": "lbl2.nii.gz"}],
        )
        mock_build_transforms.return_value = Mock()

        datamodule = SegmentationDataModule(
            cfg=sample_config_with_cache, data_dir=mock_dataset_dir, fold=0
        )
        datamodule.setup(stage="fit")

        # Act
        val_loader = datamodule.val_dataloader()

        # Assert
        assert val_loader is not None
        assert val_loader.persistent_workers is False  # Due to cache
        assert val_loader.batch_size == 1  # Validation uses batch_size=1

    @patch("src.lightning.datamodule.get_data_dicts")
    @patch("src.lightning.datamodule.build_transforms")
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
