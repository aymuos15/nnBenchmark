"""
Common runner utilities for training and testing scripts.
Provides helpers for experiment setup to reduce code duplication.
"""

from pathlib import Path
from typing import Any

import torch

from src.config import get_datasets_root, get_results_root
from src.config.load import load_config
from src.utils.files import ensure_directory


def setup_device(verbose: bool = True) -> torch.device:
    """
    Setup and return the appropriate device (CUDA or CPU).

    Args:
        verbose: If True, print device information

    Returns:
        torch.device for model placement

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    return device


def get_config_name(config_path: str) -> str:
    """
    Extract configuration name from config file path.

    Args:
        config_path: Path to config file (e.g., "configs/dataset001_hippo.yaml")

    Returns:
        Config name without extension (e.g., "dataset001_hippo")

    """
    config_file = Path(config_path)
    return config_file.stem


def setup_results_dir(config_name: str, dataset_name: str, create: bool = True) -> str:
    """
    Setup results directory path from config name and dataset name.

    Args:
        config_name: Name of the config (from get_config_name)
        dataset_name: Name of the dataset
        create: If True, create the directory if it doesn't exist

    Returns:
        Path to results directory

    """
    results_dir = str(get_results_root() / dataset_name / config_name)
    if create:
        return ensure_directory(results_dir)
    return results_dir


def setup_experiment(
    config_path: str, create_results_dir: bool = True
) -> tuple[dict[str, Any], torch.device, str, str, str]:
    """
    One-stop setup for training/testing experiments.

    Performs common setup steps:
    1. Load configuration
    2. Setup device
    3. Extract config name
    4. Setup results directory
    5. Derive data directory path from dataset name

    Args:
        config_path: Path to YAML config file
        create_results_dir: If True, create results directory

    Returns:
        Tuple of (cfg, device, data_dir, results_dir, config_name)
    """
    # Load config
    cfg = load_config(config_path)

    # Setup device
    device = setup_device(verbose=False)

    # Get paths
    dataset_name = cfg["dataset"]["name"]
    data_dir = str(get_datasets_root() / dataset_name)
    config_name = get_config_name(config_path)
    results_dir = setup_results_dir(
        config_name, dataset_name, create=create_results_dir
    )

    return cfg, device, data_dir, results_dir, config_name
