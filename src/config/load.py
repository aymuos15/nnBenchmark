"""
Configuration loading via MONAI ConfigParser.

Provides helpers for loading YAML configs with _target_ instantiation,
config inheritance, training history, and CV splits.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from monai.bundle import ConfigParser


def load_config(config_path: str) -> ConfigParser:
    """
    Load YAML configuration file via MONAI ConfigParser.

    Supports config inheritance via base_config + overrides:
        base_config: path/to/base_fold_0.yaml
        overrides:
          training:
            epochs: 400

    Args:
        config_path: Path to YAML config file

    Returns:
        ConfigParser instance with parsed config
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "base_config" not in raw:
        parser = ConfigParser(config=raw)
        return parser

    # Resolve base_config path relative to current config file
    base_config_path = raw["base_config"]
    config_dir = Path(config_path).parent

    if not Path(base_config_path).is_absolute():
        base_config_path = str(config_dir / base_config_path)

    # Load base config, then merge overrides on top
    with open(base_config_path, "r") as f:
        base = yaml.safe_load(f)

    overrides = raw.get("overrides", {})
    merged = _deep_merge(base, overrides)

    parser = ConfigParser(config=merged)
    return parser


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def instantiate(parser: ConfigParser, key: str) -> Any:
    """
    Instantiate a component from config via _target_.

    Args:
        parser: ConfigParser instance
        key: Config key to instantiate (e.g., "loss", "model")

    Returns:
        Instantiated object
    """
    return parser.get_parsed_content(key)


def instantiate_list(parser: ConfigParser, key: str) -> list[Any]:
    """
    Instantiate a list of components from config.

    Each item in the list should have a _target_ key.

    Args:
        parser: ConfigParser instance
        key: Config key pointing to a list (e.g., "validation_metrics")

    Returns:
        List of instantiated objects
    """
    items = parser.get(key, [])
    result = []
    for i in range(len(items)):
        result.append(parser.get_parsed_content(f"{key}::{i}"))
    return result


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

    Args:
        results_dir: Directory containing history/validation_history_epoch_*.json files

    Returns:
        Dictionary with aggregated validation data.
        Returns empty dict if no validation files found.
    """
    results_path = Path(results_dir) / "history"
    val_files = sorted(results_path.glob("validation_epoch_*.json"))

    if not val_files:
        return {}

    val_data: dict[str, list] = {"val_epochs": []}

    for val_file in val_files:
        with open(val_file) as f:
            val_history = json.load(f)

        epoch = val_history.get("epoch")
        if epoch is None:
            continue

        val_data["val_epochs"].append(epoch)

        summary = val_history.get("summary", {})

        for metric_name, metric_stats in summary.items():
            metric_key = f"val_{metric_name}"
            if metric_key not in val_data:
                val_data[metric_key] = []
            val_data[metric_key].append(metric_stats["mean"])

            per_class = metric_stats.get("per_class", {})
            for class_name, class_stats in per_class.items():
                class_key = f"val_{metric_name}_{class_name}"
                if class_key not in val_data:
                    val_data[class_key] = []
                val_data[class_key].append(class_stats["mean"])

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
