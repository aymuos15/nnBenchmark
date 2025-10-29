"""
Tests for src.splits module.
Tests k-fold cross-validation split generation for datasets.
"""

from __future__ import annotations

import os

import pytest

from src.planning.splits import (
    create_all_split,
    create_splits,
    extract_case_identifiers,
    load_dataset_json,
    save_splits,
)


class TestLoadDatasetJson:
    """Tests for load_dataset_json function."""

    def test_load_valid_dataset_json(self, mock_dataset_dir: str) -> None:
        """Test loading valid dataset.json."""
        dataset_json = load_dataset_json(mock_dataset_dir)

        assert "name" in dataset_json
        assert "training" in dataset_json
        assert dataset_json["name"] == "Dataset001_Hippo"

    def test_load_dataset_json_training_items(self, mock_dataset_dir: str) -> None:
        """Test that dataset.json has training items."""
        dataset_json = load_dataset_json(mock_dataset_dir)

        training = dataset_json["training"]
        assert len(training) > 0
        assert all("image" in item and "label" in item for item in training)

    def test_load_missing_dataset_json(self, temp_dir: str) -> None:
        """Test error when dataset.json is missing."""
        empty_dir = os.path.join(temp_dir, "empty_dataset")
        os.makedirs(empty_dir, exist_ok=True)

        with pytest.raises(FileNotFoundError):
            load_dataset_json(empty_dir)


class TestExtractCaseIdentifiers:
    """Tests for extract_case_identifiers function."""

    def test_extract_case_identifiers(self, mock_dataset_dir: str) -> None:
        """Test extracting case identifiers from dataset.json."""
        dataset_json = load_dataset_json(mock_dataset_dir)
        case_ids = extract_case_identifiers(dataset_json, mock_dataset_dir)

        # Should have 4 cases
        assert len(case_ids) == 4

        # Should contain expected filenames
        assert "Hippo_001_0000.nii.gz" in case_ids
        assert "Hippo_002_0000.nii.gz" in case_ids
        assert "Hippo_003_0000.nii.gz" in case_ids
        assert "Hippo_004_0000.nii.gz" in case_ids

    def test_extract_case_identifiers_uses_filenames(
        self, mock_dataset_dir: str
    ) -> None:
        """Test that extracted IDs are full filenames, not paths."""
        dataset_json = load_dataset_json(mock_dataset_dir)
        case_ids = extract_case_identifiers(dataset_json, mock_dataset_dir)

        # Should be filenames without directory paths
        for case_id in case_ids:
            assert "/" not in case_id
            assert "imagesTr" not in case_id

    def test_extract_case_identifiers_empty_dataset(self, temp_dir: str) -> None:
        """Test extracting from dataset with no training cases."""
        dataset_json = {"name": "EmptyDataset", "training": []}

        case_ids = extract_case_identifiers(dataset_json, temp_dir)
        assert len(case_ids) == 0


class TestCreateSplits:
    """Tests for create_splits function."""

    def test_create_splits_basic(self) -> None:
        """Test creating basic k-fold splits."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_splits(case_identifiers, n_folds=5, stratified=False)

        # Should have 5 folds
        assert len(splits) == 5
        assert all(f"fold_{i}" in splits for i in range(5))

    def test_create_splits_fold_structure(self) -> None:
        """Test that each fold has train and val splits."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_splits(case_identifiers, n_folds=5)

        for fold_idx in range(5):
            fold = splits[f"fold_{fold_idx}"]
            assert "train" in fold
            assert "val" in fold
            assert len(fold["train"]) > 0
            assert len(fold["val"]) > 0

    def test_create_splits_no_overlap(self) -> None:
        """Test that train and val splits don't overlap."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(20)]

        splits = create_splits(case_identifiers, n_folds=5)

        for fold_idx in range(5):
            fold = splits[f"fold_{fold_idx}"]
            train_set = set(fold["train"])
            val_set = set(fold["val"])

            # No overlap
            assert len(train_set & val_set) == 0

    def test_create_splits_all_cases_used(self) -> None:
        """Test that all cases appear in exactly one val split per fold."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(20)]

        splits = create_splits(case_identifiers, n_folds=5)

        for fold_idx in range(5):
            fold = splits[f"fold_{fold_idx}"]
            # Each case should be in either train or val
            all_cases = set(fold["train"]) | set(fold["val"])
            assert len(all_cases) == len(case_identifiers)

    def test_create_splits_reproducible_with_seed(self) -> None:
        """Test that splits are reproducible with same seed."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(20)]

        splits1 = create_splits(case_identifiers, n_folds=5, seed=42)
        splits2 = create_splits(case_identifiers, n_folds=5, seed=42)

        # Should be identical
        assert splits1 == splits2

    def test_create_splits_different_with_different_seed(self) -> None:
        """Test that splits differ with different seeds."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(20)]

        splits1 = create_splits(case_identifiers, n_folds=5, seed=42)
        splits2 = create_splits(case_identifiers, n_folds=5, seed=123)

        # At least one fold should differ
        differs = False
        for fold_idx in range(5):
            if splits1[f"fold_{fold_idx}"] != splits2[f"fold_{fold_idx}"]:
                differs = True
                break
        assert differs

    def test_create_splits_2_folds(self) -> None:
        """Test creating 2-fold splits."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_splits(case_identifiers, n_folds=2)

        assert len(splits) == 2
        assert "fold_0" in splits
        assert "fold_1" in splits

    def test_create_splits_many_folds(self) -> None:
        """Test creating many folds (leave-one-out style)."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_splits(case_identifiers, n_folds=10)

        assert len(splits) == 10
        for fold_idx in range(10):
            fold = splits[f"fold_{fold_idx}"]
            # With 10 folds from 10 cases, each val split should have 1 case
            assert len(fold["val"]) == 1

    def test_create_splits_stratified_requires_dataset_path(self) -> None:
        """Test that stratified splitting requires dataset_path."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        with pytest.raises(ValueError) as excinfo:
            create_splits(case_identifiers, n_folds=5, stratified=True)

        assert "dataset_path is required" in str(excinfo.value)


class TestSaveSplits:
    """Tests for save_splits function."""

    def test_save_splits_creates_file(self, temp_dir: str) -> None:
        """Test that save_splits creates a JSON file."""
        splits = {
            "fold_0": {
                "train": ["case_001.nii.gz", "case_002.nii.gz"],
                "val": ["case_003.nii.gz", "case_004.nii.gz"],
            },
            "fold_1": {
                "train": ["case_003.nii.gz", "case_004.nii.gz"],
                "val": ["case_001.nii.gz", "case_002.nii.gz"],
            },
        }

        output_path = os.path.join(temp_dir, "splits.json")
        save_splits(splits, output_path)

        assert os.path.exists(output_path)

    def test_save_splits_file_contents(self, temp_dir: str) -> None:
        """Test that saved splits file has correct contents."""
        import json

        splits = {
            "fold_0": {
                "train": ["case_001.nii.gz", "case_002.nii.gz"],
                "val": ["case_003.nii.gz"],
            }
        }

        output_path = os.path.join(temp_dir, "splits.json")
        save_splits(splits, output_path)

        # Load and verify
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert loaded == splits


class TestCreateAllSplit:
    """Tests for create_all_split function."""

    def test_create_all_split_basic(self) -> None:
        """Test creating an 'all' split with all cases in training."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_all_split(case_identifiers)

        # Should have single fold_-1 entry
        assert len(splits) == 1
        assert "fold_-1" in splits

    def test_create_all_split_structure(self) -> None:
        """Test that 'all' split has correct structure."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_all_split(case_identifiers)
        fold = splits["fold_-1"]

        # Should have train and val keys
        assert "train" in fold
        assert "val" in fold

        # All cases should be in train
        assert len(fold["train"]) == 10
        # Val should be empty
        assert len(fold["val"]) == 0

    def test_create_all_split_includes_all_cases(self) -> None:
        """Test that 'all' split includes all case identifiers."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]

        splits = create_all_split(case_identifiers)

        # All cases should be in train set
        train_cases = set(splits["fold_-1"]["train"])
        expected_cases = set(case_identifiers)
        assert train_cases == expected_cases

    def test_create_all_split_empty_validation(self) -> None:
        """Test that validation set is always empty."""
        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(5)]

        splits = create_all_split(case_identifiers)

        assert splits["fold_-1"]["val"] == []

    def test_create_all_split_single_case(self) -> None:
        """Test creating 'all' split with single case."""
        case_identifiers = ["single_case.nii.gz"]

        splits = create_all_split(case_identifiers)

        assert len(splits["fold_-1"]["train"]) == 1
        assert splits["fold_-1"]["val"] == []

    def test_create_all_split_many_cases(self) -> None:
        """Test creating 'all' split with many cases."""
        case_identifiers = [f"case_{i:04d}.nii.gz" for i in range(100)]

        splits = create_all_split(case_identifiers)

        assert len(splits["fold_-1"]["train"]) == 100
        assert len(splits["fold_-1"]["val"]) == 0

    def test_save_all_split(self, temp_dir: str) -> None:
        """Test saving 'all' split to file."""
        import json

        case_identifiers = [f"case_{i:03d}.nii.gz" for i in range(10)]
        splits = create_all_split(case_identifiers)

        output_path = os.path.join(temp_dir, "splits.json")
        save_splits(splits, output_path)

        # Verify file exists and has correct content
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert loaded == splits
        assert "fold_-1" in loaded
        assert len(loaded["fold_-1"]["train"]) == 10
        assert loaded["fold_-1"]["val"] == []

    def test_create_all_split_from_mock_dataset(self, mock_dataset_dir: str) -> None:
        """Test creating 'all' split from mock dataset."""
        dataset_json = load_dataset_json(mock_dataset_dir)
        case_ids = extract_case_identifiers(dataset_json, mock_dataset_dir)

        splits = create_all_split(case_ids)

        # Should have 4 cases (from mock dataset)
        assert len(splits["fold_-1"]["train"]) == 4
        assert splits["fold_-1"]["val"] == []

        # Verify all case IDs are in training
        for case_id in case_ids:
            assert case_id in splits["fold_-1"]["train"]

    def test_save_splits_overwrites_existing(self, temp_dir: str) -> None:
        """Test that save_splits overwrites existing file."""
        import json

        output_path = os.path.join(temp_dir, "splits.json")

        # Save first version
        splits1 = {"fold_0": {"train": ["case_001.nii.gz"], "val": ["case_002.nii.gz"]}}
        save_splits(splits1, output_path)

        # Save second version
        splits2 = {"fold_0": {"train": ["case_003.nii.gz"], "val": ["case_004.nii.gz"]}}
        save_splits(splits2, output_path)

        # Should have second version
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert loaded == splits2


class TestIntegrationWithMockDataset:
    """Integration tests using mock dataset."""

    def test_load_and_extract_from_mock_dataset(self, mock_dataset_dir: str) -> None:
        """Test loading and extracting from mock dataset."""
        dataset_json = load_dataset_json(mock_dataset_dir)
        case_ids = extract_case_identifiers(dataset_json, mock_dataset_dir)

        assert len(case_ids) == 4
        assert all(case_id.endswith(".nii.gz") for case_id in case_ids)

    def test_create_splits_from_mock_dataset(self, mock_dataset_dir: str) -> None:
        """Test creating splits from mock dataset."""
        dataset_json = load_dataset_json(mock_dataset_dir)
        case_ids = extract_case_identifiers(dataset_json, mock_dataset_dir)

        splits = create_splits(case_ids, n_folds=2)

        # Should have 2 folds
        assert len(splits) == 2

        # Each fold should split the 4 cases
        for fold_idx in range(2):
            fold = splits[f"fold_{fold_idx}"]
            assert len(fold["train"]) == 2
            assert len(fold["val"]) == 2

    def test_save_splits_from_mock_dataset(
        self, mock_dataset_dir: str, temp_dir: str
    ) -> None:
        """Test complete workflow: load, create, save splits."""
        dataset_json = load_dataset_json(mock_dataset_dir)
        case_ids = extract_case_identifiers(dataset_json, mock_dataset_dir)
        splits = create_splits(case_ids, n_folds=2)

        output_path = os.path.join(temp_dir, "splits.json")
        save_splits(splits, output_path)

        # Verify file exists and has correct content
        import json

        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert loaded == splits
