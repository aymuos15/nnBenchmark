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
    Load YAML configuration file with optional base_config inheritance.

    Supports minimal override configs that reference a base config:
        base_config: path/to/base_fold_0.yaml
        overrides:
          training:
            epochs: 400

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration (merged if base_config specified)

    Raises:
        ConfigValidationError: If override keys don't exist in base config
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check if this config uses inheritance
    if "base_config" not in config:
        return config

    # Import here to avoid circular dependency
    from src.config.merge import load_config_with_inheritance

    # Resolve base_config path relative to current config file
    base_config_path = config["base_config"]
    config_dir = Path(config_path).parent

    # If base_config is relative, resolve it relative to current config
    if not Path(base_config_path).is_absolute():
        base_config_path = str(config_dir / base_config_path)

    # Use load_config_with_inheritance, passing this function as loader
    # (creates a proper config dict for the merge function)
    config_with_resolved_path = config.copy()
    config_with_resolved_path["base_config"] = base_config_path

    return load_config_with_inheritance(config_with_resolved_path, load_config)


def load_training_history(results_dir: str) -> dict[str, list[float]]:
    """
    Load training history from JSON file.

    Args:
        results_dir: Directory containing history/training_history.json

    Returns:
        Dictionary with training history (epochs, losses, metrics)

    Raises:
        FileNotFoundError: If training_history.json doesn't exist
    """
    history_path = str(Path(results_dir) / "history" / "training.json")

    if not Path(history_path).exists():
        raise FileNotFoundError(
            f"Training history not found at {history_path}\n"
            f"Make sure you've trained a model and saved the training history."
        )

    with open(history_path, "r") as f:
        history: dict[str, list[float]] = json.load(f)

    return history


def load_validation_histories(results_dir: str) -> dict[str, list[float]]:
    """
    Load and aggregate validation histories from multiple validation_history_epoch_*.json files.

    Finds all validation history files in results_dir/history/, sorts by epoch, and aggregates
    metrics into time-series format suitable for plotting.

    Args:
        results_dir: Directory containing history/validation_history_epoch_*.json files

    Returns:
        Dictionary with aggregated validation data:
        {
            "val_epochs": [1, 5, 10, ...],
            "val_DiceMetric": [0.7, 0.75, 0.8, ...],
            "val_DiceMetric_Anterior": [0.65, 0.7, 0.75, ...],  # per-class
            ...
        }
        Returns empty dict if no validation files found.
    """
    import glob

    results_path = Path(results_dir) / "history"
    val_pattern = str(results_path / "validation_epoch_*.json")
    val_files = sorted(glob.glob(val_pattern))

    if not val_files:
        return {}

    # Collect data from all validation files
    val_data: dict[str, list] = {"val_epochs": []}

    for val_file in val_files:
        with open(val_file) as f:
            val_history = json.load(f)

        epoch = val_history.get("epoch")
        if epoch is None:
            continue

        val_data["val_epochs"].append(epoch)

        # Extract summary metrics (mean values across all samples)
        summary = val_history.get("summary", {})

        for metric_name, metric_stats in summary.items():
            # Add main metric (mean across all classes and samples)
            metric_key = f"val_{metric_name}"
            if metric_key not in val_data:
                val_data[metric_key] = []
            val_data[metric_key].append(metric_stats["mean"])

            # Add per-class metrics if available
            per_class = metric_stats.get("per_class", {})
            for class_name, class_stats in per_class.items():
                class_key = f"val_{metric_name}_{class_name}"
                if class_key not in val_data:
                    val_data[class_key] = []
                class_key_data = class_stats["mean"]
                val_data[class_key].append(class_key_data)

    return val_data


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
