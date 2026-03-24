"""Configuration module for nnBenchmark."""


from src.config.load import instantiate, instantiate_list, load_config
from src.config.paths import (
    get_datasets_root,
    get_preprocessed_root,
    get_results_root,
)

__all__ = [
    "get_datasets_root",
    "get_preprocessed_root",
    "get_results_root",
    "instantiate",
    "instantiate_list",
    "load_config",
]
