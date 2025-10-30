"""
Dataset and results paths configuration.

This module uses environment variables to locate datasets, preprocessed data,
and results, following nnUNet's convention. Set these environment variables:

    export nnUNet_raw="/path/to/nnUNet_raw"
    export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
    export nnUNet_results="/path/to/nnUNet_results"

Add these to your ~/.bashrc or ~/.zshrc for persistence.
"""

from __future__ import annotations

import os
from pathlib import Path


def _get_env_path(env_var: str, description: str) -> Path:
    """
    Get a required path from environment variable.

    Args:
        env_var: Name of environment variable
        description: Human-readable description for error messages

    Returns:
        Path from environment variable

    Raises:
        RuntimeError: If environment variable is not set or path parent doesn't exist
    """
    value = os.environ.get(env_var)
    if value is None:
        raise RuntimeError(
            f"Environment variable '{env_var}' is not set.\n"
            f"This should point to your {description}.\n"
            f"Set it with: export {env_var}='/path/to/{description}'\n"
            f"Or add it to your ~/.bashrc or ~/.zshrc for persistence."
        )

    path = Path(value).expanduser().resolve()

    # Only validate that parent exists (so users can point to new directories)
    if not path.parent.exists():
        raise RuntimeError(
            f"Parent directory of {env_var}={path} does not exist.\n"
            f"Please ensure the path is correct."
        )

    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    return path


def get_datasets_root() -> Path:
    """
    Get the datasets root directory from nnUNet_raw environment variable.

    Returns:
        Path object pointing to the datasets root directory

    Raises:
        RuntimeError: If nnUNet_raw environment variable is not set
    """
    return _get_env_path("nnUNet_raw", "nnUNet_raw")


def get_preprocessed_root() -> Path:
    """
    Get the preprocessed data root directory from nnUNet_preprocessed.

    Returns:
        Path object pointing to the preprocessed root directory

    Raises:
        RuntimeError: If nnUNet_preprocessed environment variable is not set
    """
    return _get_env_path("nnUNet_preprocessed", "nnUNet_preprocessed")


def get_results_root() -> Path:
    """
    Get the results root directory from nnUNet_results environment variable.

    Returns:
        Path object pointing to the results root directory

    Raises:
        RuntimeError: If nnUNet_results environment variable is not set
    """
    return _get_env_path("nnUNet_results", "nnUNet_results")


def get_dataset_path(dataset_name: str) -> Path:
    """
    Get the full path to a specific dataset.

    Args:
        dataset_name: Dataset directory name (e.g., 'Dataset001_Hippo')

    Returns:
        Full path to the dataset directory
    """
    return get_datasets_root() / dataset_name
