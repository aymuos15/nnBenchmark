"""
Shared pytest fixtures for nnBenchmark tests.
Provides mock configs, dataset files, and temporary directories.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest
import yaml


@pytest.fixture(scope="session", autouse=True)
def _setup_nnunet_env_vars(tmp_path_factory: pytest.TempPathFactory) -> None:  # noqa: ARG001
    """
    Set up nnUNet environment variables for testing.

    Creates temporary directories and sets environment variables
    required by the nnUNet pipeline. This fixture runs automatically
    for all tests in the session.

    Note: Named with leading underscore to indicate internal fixture.
    Using autouse=True so it's automatically applied without being called.
    """
    # Create temporary base directory for nnUNet paths
    base_tmp = tmp_path_factory.mktemp("nnunet_env")

    # Create and set environment variables
    nnunet_raw = base_tmp / "nnUNet_raw"
    nnunet_preprocessed = base_tmp / "nnUNet_preprocessed"
    nnunet_results = base_tmp / "nnUNet_results"

    nnunet_raw.mkdir(exist_ok=True)
    nnunet_preprocessed.mkdir(exist_ok=True)
    nnunet_results.mkdir(exist_ok=True)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)


@pytest.fixture
def mock_dataset_json() -> dict[str, Any]:
    """Mock dataset.json for testing (Hippo-style)."""
    return {
        "name": "Dataset001_Hippo",
        "description": "Hippocampus segmentation test dataset",
        "labels": {"0": "background", "1": "Anterior", "2": "Posterior"},
        "numTraining": 4,
        "file_ending": ".nii.gz",
        "training": [
            {
                "image": "./imagesTr/Hippo_001_0000.nii.gz",
                "label": "./labelsTr/Hippo_001.nii.gz",
            },
            {
                "image": "./imagesTr/Hippo_002_0000.nii.gz",
                "label": "./labelsTr/Hippo_002.nii.gz",
            },
            {
                "image": "./imagesTr/Hippo_003_0000.nii.gz",
                "label": "./labelsTr/Hippo_003.nii.gz",
            },
            {
                "image": "./imagesTr/Hippo_004_0000.nii.gz",
                "label": "./labelsTr/Hippo_004.nii.gz",
            },
        ],
    }


@pytest.fixture
def mock_splits_json() -> dict[str, Any]:
    """Mock splits.json for testing (2 folds)."""
    return {
        "fold_0": {
            "train": ["Hippo_001_0000.nii.gz", "Hippo_002_0000.nii.gz"],
            "val": ["Hippo_003_0000.nii.gz", "Hippo_004_0000.nii.gz"],
        },
        "fold_1": {
            "train": ["Hippo_003_0000.nii.gz", "Hippo_004_0000.nii.gz"],
            "val": ["Hippo_001_0000.nii.gz", "Hippo_002_0000.nii.gz"],
        },
    }


@pytest.fixture
def mock_training_history() -> dict[str, Any]:
    """Mock training_history.json for testing."""
    return {
        "epochs": [1, 2, 3],
        "train_loss": [0.5, 0.3, 0.2],
        "val_epochs": [2, 3],
        "DiceMetric_mean": [0.7, 0.8],
        "DiceMetric_per_class": {"Anterior": [0.65, 0.75], "Posterior": [0.75, 0.85]},
    }


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "dataset": {
            "name": "Dataset001_Hippo",
            "spatial_size": [40, 56, 40],
            "num_classes": 3,
            "fold": 0,
        },
        "model": {
            "type": "DynUNet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 3,
            "filters": [16, 32, 64],
            "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "upsample_kernel_size": [[2, 2, 2], [2, 2, 2]],
            "norm_name": ["INSTANCE", {"affine": True}],
            "act_name": ["leakyrelu", {"inplace": True, "negative_slope": 0.01}],
            "deep_supervision": False,
            "deep_supr_num": 1,
            "res_block": False,
        },
        "training": {
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "val_interval": 2,
            "num_workers": 4,
            "checkpoint_metric": "DiceMetric",
            "plot_metrics": ["DiceMetric"],
        },
        "optimizer": {"type": "Adam", "weight_decay": 0.0001},
        "loss": {"type": "DiceCELoss", "to_onehot_y": True, "softmax": True},
        "metrics": [
            {
                "type": "DiceMetric",
                "include_background": False,
                "reduction": "mean_batch",
                "num_classes": 3,
            }
        ],
        "transforms": {
            "common": [
                {"type": "LoadImaged", "keys": ["image", "label"]},
                {"type": "EnsureChannelFirstd", "keys": ["image", "label"]},
                {"type": "ScaleIntensityd", "keys": ["image"]},
                {"type": "ToTensord", "keys": ["image", "label"]},
            ],
            "train": [
                {
                    "type": "RandFlipd",
                    "keys": ["image", "label"],
                    "prob": 0.5,
                    "spatial_axis": 0,
                }
            ],
            "val": [],
            "test": [],
        },
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_dataset_dir(
    temp_dir: str, mock_dataset_json: dict[str, Any], mock_splits_json: dict[str, Any]
) -> str:
    """
    Create a mock dataset directory with dataset.json and splits.json.

    Also creates corresponding preprocessed directories in nnUNet_preprocessed
    to support tests that expect preprocessed data.

    Returns:
        Path to the mock dataset directory
    """
    # Create dataset directory structure
    dataset_dir = os.path.join(temp_dir, "Dataset001_Hippo")
    os.makedirs(dataset_dir, exist_ok=True)

    # Create subdirectories
    for subdir in [
        "imagesTr",
        "labelsTr",
        "imagesTs",
        "labelsTs",
        "imagesTr_cropped",
        "labelsTr_cropped",
    ]:
        os.makedirs(os.path.join(dataset_dir, subdir), exist_ok=True)

    # Write dataset.json
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(mock_dataset_json, f, indent=2)

    # Note: splits.json is now saved to the preprocessed directory, not the raw dataset directory
    # This is created below in the preprocessed dataset structure section

    # Create empty placeholder files for images/labels
    for item in mock_dataset_json["training"]:
        img_path = os.path.join(dataset_dir, item["image"].replace("./", ""))
        label_path = os.path.join(dataset_dir, item["label"].replace("./", ""))

        # Touch files (create empty files)
        open(img_path, "a").close()
        open(label_path, "a").close()

        # Also create cropped versions
        img_cropped = img_path.replace("imagesTr/", "imagesTr_cropped/")
        label_cropped = label_path.replace("labelsTr/", "labelsTr_cropped/")
        open(img_cropped, "a").close()
        open(label_cropped, "a").close()

    # Create preprocessed dataset structure in nnUNet_preprocessed
    # This is required for get_data_dicts() to work
    preprocessed_root = os.environ.get("nnUNet_preprocessed")
    if preprocessed_root:
        preprocessed_dataset_dir = os.path.join(preprocessed_root, "Dataset001_Hippo")
        for subdir in ["imagesTr", "labelsTr"]:
            os.makedirs(os.path.join(preprocessed_dataset_dir, subdir), exist_ok=True)

        # Create splits.json in preprocessed location
        splits_path = os.path.join(preprocessed_dataset_dir, "splits.json")
        with open(splits_path, "w") as f:
            json.dump(mock_splits_json, f, indent=2)

        # Create empty placeholder files in preprocessed directories
        for item in mock_dataset_json["training"]:
            img_name = os.path.basename(item["image"])
            label_name = os.path.basename(item["label"])

            img_path = os.path.join(preprocessed_dataset_dir, "imagesTr", img_name)
            label_path = os.path.join(preprocessed_dataset_dir, "labelsTr", label_name)

            open(img_path, "a").close()
            open(label_path, "a").close()

    return dataset_dir


@pytest.fixture
def mock_config_file(temp_dir: str, sample_config: dict[str, Any]) -> str:
    """
    Create a mock YAML config file.

    Returns:
        Path to the mock config file
    """
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def mock_results_dir(temp_dir: str, mock_training_history: dict[str, Any]) -> str:
    """
    Create a mock results directory with training_history.json.

    Returns:
        Path to the mock results directory
    """
    results_dir = os.path.join(temp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Write training_history.json
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(mock_training_history, f, indent=2)

    return results_dir
