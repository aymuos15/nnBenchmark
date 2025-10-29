"""
Tests for src.config.load module.
Tests configuration loading utilities including YAML configs, training history, and splits.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from src.config.load import load_config, load_splits, load_training_history


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(
        self, mock_config_file: str, sample_config: dict[str, Any]
    ) -> None:
        """Test loading valid YAML configuration file."""
        result = load_config(mock_config_file)

        assert result == sample_config
        assert result["dataset"]["name"] == "Dataset001_Hippo"
        assert result["model"]["type"] == "UNet"
        assert result["training"]["epochs"] == 5

    def test_load_missing_config_file(self, temp_dir: str) -> None:
        """Test loading non-existent config file raises FileNotFoundError."""
        missing_config = os.path.join(temp_dir, "missing_config.yaml")

        with pytest.raises(FileNotFoundError):
            load_config(missing_config)

    def test_load_invalid_yaml(self, temp_dir: str) -> None:
        """Test loading invalid YAML raises appropriate error."""
        invalid_config = os.path.join(temp_dir, "invalid.yaml")

        # Write invalid YAML
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: content:\n  - malformed\n  - [")

        with pytest.raises(Exception):  # yaml.YAMLError or similar
            load_config(invalid_config)


class TestLoadTrainingHistory:
    """Tests for load_training_history function."""

    def test_load_valid_training_history(
        self, mock_results_dir: str, mock_training_history: dict[str, Any]
    ) -> None:
        """Test loading valid training_history.json."""
        result = load_training_history(mock_results_dir)

        assert result == mock_training_history
        assert result["epochs"] == [1, 2, 3]
        assert result["train_loss"] == [0.5, 0.3, 0.2]
        assert result["val_epochs"] == [2, 3]

    def test_load_missing_training_history(self, temp_dir: str) -> None:
        """Test loading non-existent training_history.json raises FileNotFoundError."""
        empty_results_dir = os.path.join(temp_dir, "empty_results")
        os.makedirs(empty_results_dir, exist_ok=True)

        with pytest.raises(FileNotFoundError) as excinfo:
            load_training_history(empty_results_dir)

        assert "Training history not found" in str(excinfo.value)
        assert "training_history.json" in str(excinfo.value)


class TestLoadSplits:
    """Tests for load_splits function."""

    def test_load_valid_splits_fold_0(self, mock_dataset_dir: str) -> None:
        """Test loading valid splits for fold 0."""
        train_cases, val_cases = load_splits(mock_dataset_dir, fold=0)

        assert len(train_cases) == 2
        assert len(val_cases) == 2
        assert "Hippo_001_0000.nii.gz" in train_cases
        assert "Hippo_002_0000.nii.gz" in train_cases
        assert "Hippo_003_0000.nii.gz" in val_cases
        assert "Hippo_004_0000.nii.gz" in val_cases

    def test_load_valid_splits_fold_1(self, mock_dataset_dir: str) -> None:
        """Test loading valid splits for fold 1."""
        train_cases, val_cases = load_splits(mock_dataset_dir, fold=1)

        assert len(train_cases) == 2
        assert len(val_cases) == 2
        assert "Hippo_003_0000.nii.gz" in train_cases
        assert "Hippo_004_0000.nii.gz" in train_cases
        assert "Hippo_001_0000.nii.gz" in val_cases
        assert "Hippo_002_0000.nii.gz" in val_cases

    def test_load_missing_splits_file(self, temp_dir: str) -> None:
        """Test loading splits from directory without splits.json raises FileNotFoundError."""
        empty_dir = os.path.join(temp_dir, "no_splits")
        os.makedirs(empty_dir, exist_ok=True)

        with pytest.raises(FileNotFoundError) as excinfo:
            load_splits(empty_dir, fold=0)

        assert "splits.json not found" in str(excinfo.value)
        assert "nnBench.plan" in str(excinfo.value)

    def test_load_invalid_fold(self, mock_dataset_dir: str) -> None:
        """Test loading non-existent fold raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            load_splits(mock_dataset_dir, fold=5)

        assert "Fold 5 not found" in str(excinfo.value)
        assert "Available folds" in str(excinfo.value)

    def test_load_all_split_fold_minus_1(self, mock_dataset_dir: str) -> None:
        """Test loading fold=-1 (all data split) if available."""
        import json

        # First, we need to add fold_-1 to the splits.json
        from pathlib import Path

        splits_path = Path(mock_dataset_dir) / "splits.json"
        with open(splits_path, "r") as f:
            splits = json.load(f)

        # Add fold_-1 with all cases in training
        all_cases = []
        for fold_data in splits.values():
            all_cases.extend(fold_data["train"])
            all_cases.extend(fold_data["val"])

        all_cases = list(set(all_cases))  # Remove duplicates

        splits["fold_-1"] = {"train": all_cases, "val": []}

        with open(splits_path, "w") as f:
            json.dump(splits, f)

        # Now load fold=-1
        train_cases, val_cases = load_splits(mock_dataset_dir, fold=-1)

        # Training should have all 4 cases
        assert len(train_cases) == 4
        # Validation should be empty
        assert len(val_cases) == 0
        # All cases should be in training
        assert all(
            case in train_cases
            for case in [
                "Hippo_001_0000.nii.gz",
                "Hippo_002_0000.nii.gz",
                "Hippo_003_0000.nii.gz",
                "Hippo_004_0000.nii.gz",
            ]
        )
