"""Configuration module for nnBenchmark."""

from __future__ import annotations

from src.config.paths import (
    get_dataset_path,
    get_datasets_root,
    get_preprocessed_root,
    get_results_root,
)
from src.config.resolution import resolve_config_path

__all__ = [
    "get_datasets_root",
    "get_preprocessed_root",
    "get_results_root",
    "get_dataset_path",
    "resolve_config_path",
]
