"""
Tests for src.utils.data module.
Tests data loading utilities including creating data dictionaries from dataset.json and splits.
"""


import os

import pytest

from src.utils.data import get_class_labels, get_data_dicts, get_test_data_dicts


class TestGetDataDicts:
    """Tests for get_data_dicts function."""

    def test_get_data_dicts_fold_0(self, mock_dataset_dir: str) -> None:
        """Test getting train/val data dicts for fold 0."""
        train_data, val_data = get_data_dicts(mock_dataset_dir, fold=0)

        # Check counts
        assert len(train_data) == 2
        assert len(val_data) == 2

        # Check structure
        assert all("image" in item and "label" in item for item in train_data)
        assert all("image" in item and "label" in item for item in val_data)

        # Check correct split (fold 0: train on 001,002, val on 003,004)
        train_images = [os.path.basename(item["image"]) for item in train_data]
        assert "Hippo_001_0000.nii.gz" in train_images
        assert "Hippo_002_0000.nii.gz" in train_images

        val_images = [os.path.basename(item["image"]) for item in val_data]
        assert "Hippo_003_0000.nii.gz" in val_images
        assert "Hippo_004_0000.nii.gz" in val_images

    def test_get_data_dicts_fold_1(self, mock_dataset_dir: str) -> None:
        """Test getting train/val data dicts for fold 1."""
        train_data, val_data = get_data_dicts(mock_dataset_dir, fold=1)

        # Check correct split (fold 1: train on 003,004, val on 001,002)
        train_images = [os.path.basename(item["image"]) for item in train_data]
        assert "Hippo_003_0000.nii.gz" in train_images
        assert "Hippo_004_0000.nii.gz" in train_images

        val_images = [os.path.basename(item["image"]) for item in val_data]
        assert "Hippo_001_0000.nii.gz" in val_images
        assert "Hippo_002_0000.nii.gz" in val_images

    def test_get_data_dicts_file_paths_correct(self, mock_dataset_dir: str) -> None:
        """Test that file paths in data dicts are correct and absolute."""
        train_data, _ = get_data_dicts(mock_dataset_dir, fold=0)

        # Check first training item
        first_train = train_data[0]
        assert os.path.isabs(first_train["image"])
        assert os.path.isabs(first_train["label"])
        assert "imagesTr" in first_train["image"]
        assert "labelsTr" in first_train["label"]


class TestGetTestDataDicts:
    """Tests for get_test_data_dicts function."""

    def test_get_test_data_dicts_validation_split(self, mock_dataset_dir: str) -> None:
        """Test getting test data using validation split (use_test_set=False)."""
        test_data = get_test_data_dicts(mock_dataset_dir, fold=0, use_test_set=False)

        # Should return validation split for fold 0
        assert len(test_data) == 2

        test_images = [os.path.basename(item["image"]) for item in test_data]
        assert "Hippo_003_0000.nii.gz" in test_images
        assert "Hippo_004_0000.nii.gz" in test_images

    def test_get_test_data_dicts_no_fold_when_using_val_split(
        self, mock_dataset_dir: str
    ) -> None:
        """Test that fold is required when use_test_set=False."""
        with pytest.raises(ValueError) as excinfo:
            get_test_data_dicts(mock_dataset_dir, fold=None, use_test_set=False)

        assert "fold parameter is required" in str(excinfo.value)

    def test_get_test_data_dicts_missing_test_set(self, mock_dataset_dir: str) -> None:
        """Test error when trying to use test set that doesn't exist."""
        # Remove test set directories
        import shutil

        shutil.rmtree(os.path.join(mock_dataset_dir, "imagesTs"))
        shutil.rmtree(os.path.join(mock_dataset_dir, "labelsTs"))

        with pytest.raises(FileNotFoundError) as excinfo:
            get_test_data_dicts(mock_dataset_dir, fold=None, use_test_set=True)

        assert "Test set directory not found" in str(excinfo.value)
        assert "imagesTs" in str(excinfo.value)
        assert "--use-val-split" in str(excinfo.value)

    def test_get_test_data_dicts_empty_test_set(self, mock_dataset_dir: str) -> None:
        """Test error when test set directory exists but is empty."""
        # imagesTs exists but is empty (no files created in fixture for test set)
        # We need to verify it handles this case

        # First, let's create a test image to avoid the "no images found" error
        # but not create its corresponding label
        test_img_path = os.path.join(
            mock_dataset_dir, "imagesTs", "Test_001_0000.nii.gz"
        )
        open(test_img_path, "a").close()

        with pytest.raises(ValueError) as excinfo:
            get_test_data_dicts(mock_dataset_dir, fold=None, use_test_set=True)

        assert "No valid image-label pairs found" in str(excinfo.value)

    def test_get_test_data_dicts_with_valid_test_set(
        self, mock_dataset_dir: str
    ) -> None:
        """Test getting test data with valid test set."""
        # Create test set files
        test_img_path = os.path.join(
            mock_dataset_dir, "imagesTs", "Test_001_0000.nii.gz"
        )
        test_label_path = os.path.join(mock_dataset_dir, "labelsTs", "Test_001.nii.gz")

        open(test_img_path, "a").close()
        open(test_label_path, "a").close()

        test_data = get_test_data_dicts(mock_dataset_dir, fold=None, use_test_set=True)

        assert len(test_data) == 1
        assert "Test_001_0000.nii.gz" in test_data[0]["image"]
        assert "Test_001.nii.gz" in test_data[0]["label"]


class TestGetClassLabels:
    """Tests for get_class_labels function."""

    def test_get_class_labels_without_background(
        self, mock_dataset_dir: str, mock_dataset_json: dict
    ) -> None:
        """Test getting class labels without background class."""
        labels = get_class_labels(mock_dataset_dir, include_background=False)

        assert len(labels) == 2
        assert 0 not in labels
        assert labels[1] == "Anterior"
        assert labels[2] == "Posterior"

    def test_get_class_labels_with_background(
        self, mock_dataset_dir: str, mock_dataset_json: dict
    ) -> None:
        """Test getting class labels with background class."""
        labels = get_class_labels(mock_dataset_dir, include_background=True)

        assert len(labels) == 3
        assert labels[0] == "background"
        assert labels[1] == "Anterior"
        assert labels[2] == "Posterior"

    def test_get_class_labels_returns_dict_with_int_keys(
        self, mock_dataset_dir: str
    ) -> None:
        """Test that returned dictionary has integer keys."""
        labels = get_class_labels(mock_dataset_dir, include_background=True)

        for key in labels.keys():
            assert isinstance(key, int)
