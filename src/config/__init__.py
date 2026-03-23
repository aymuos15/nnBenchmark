"""Configuration module for nnBenchmark."""


from src.config.paths import (
    get_datasets_root,
    get_preprocessed_root,
    get_results_root,
)
from src.config.resolution import resolve_config_path

__all__ = [
    "get_datasets_root",
    "get_preprocessed_root",
    "get_results_root",
    "resolve_config_path",
]
