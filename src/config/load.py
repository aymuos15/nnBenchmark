"""
Configuration and split management utilities for nnBenchmark.
Provides helpers for loading YAML configs, training history, and CV splits.
"""

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_history(results_dir: str) -> dict[str, list[float]]:
    """
    Load training history from JSON file.

    Args:
        results_dir: Directory containing training_history.json

    Returns:
        Dictionary with training history (epochs, losses, metrics)

    Raises:
        FileNotFoundError: If training_history.json doesn't exist
    """
    history_path = str(Path(results_dir) / "training_history.json")

    if not Path(history_path).exists():
        raise FileNotFoundError(
            f"Training history not found at {history_path}\n"
            f"Make sure you've trained a model and saved the training history."
        )

    with open(history_path, "r") as f:
        history: dict[str, list[float]] = json.load(f)

    return history


def load_splits(data_dir: str, fold: int) -> tuple[list[str], list[str]]:
    """
    Load train/val splits from splits.json.

    Args:
        data_dir: Dataset directory (used to determine preprocessed directory)
        fold: Fold number to load (e.g., 0 for fold_0, or -1 for all data)

    Returns:
        Tuple of (train_cases, val_cases) as lists of case IDs.
        For fold=-1 (training on all data), val_cases will be an empty list.

    Raises:
        FileNotFoundError: If splits.json doesn't exist
        ValueError: If fold doesn't exist in splits.json
    """
    from src.config.paths import get_preprocessed_root

    # Get splits.json from preprocessed directory
    data_dir_path = Path(data_dir)
    dataset_name = data_dir_path.name
    preprocessed_root = get_preprocessed_root()
    splits_path = str(preprocessed_root / dataset_name / "splits.json")

    if not Path(splits_path).exists():
        raise FileNotFoundError(
            f"splits.json not found at {splits_path}. "
            f"Please run: nnBench.plan --dataset {data_dir}"
        )

    with open(splits_path, "r") as f:
        splits = json.load(f)

    # Handle fold=-1: train on all data
    if fold == -1:
        all_cases = set()
        for fold_data in splits.values():
            all_cases.update(fold_data["train"])
            all_cases.update(fold_data["val"])
        return sorted(list(all_cases)), []

    fold_key = f"fold_{fold}"
    if fold_key not in splits:
        raise ValueError(
            f"Fold {fold} not found in splits.json. Available folds: {list(splits.keys())}"
        )

    return splits[fold_key]["train"], splits[fold_key]["val"]
