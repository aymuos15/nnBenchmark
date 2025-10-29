"""Configuration module for nnBenchmark."""

from src.config.paths import (
    CONFIGS_ROOT,
    DATASETS_ROOT,
    RESULTS_ROOT,
    get_configs_root,
    get_dataset_path,
    get_datasets_root,
    get_results_root,
)

__all__ = [
    "DATASETS_ROOT",
    "RESULTS_ROOT",
    "CONFIGS_ROOT",
    "get_datasets_root",
    "get_results_root",
    "get_configs_root",
    "get_dataset_path",
]

