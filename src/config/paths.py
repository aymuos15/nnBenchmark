"""
Dataset and results paths configuration.

EDIT THIS FILE: Paste your dataset root directory path below.
This is where nnBenchmark will look for datasets and store results.
"""

from pathlib import Path

# ============================================================================
# USER CONFIGURATION - EDIT THIS
# ============================================================================
# Paste your dataset root directory path here.
# Examples:
#   DATASETS_ROOT = Path("/mnt/data/datasets")
#   DATASETS_ROOT = Path("/home/user/medical_data/datasets")
#   DATASETS_ROOT = Path("./datasets")  # Relative to working directory
#
# This directory should contain subdirectories like:
#   <DATASETS_ROOT>/Dataset001_Hippo/
#   <DATASETS_ROOT>/Dataset002_Kits/
#   etc.

DATASETS_ROOT = Path("/home/localssk23/CAI4Soumya/nnUNet_raw")  # ← EDIT THIS PATH

# Results directory (where training outputs are saved)
RESULTS_ROOT = Path("results")

# Configs directory (where auto-generated configs are saved)
CONFIGS_ROOT = Path("configs")

# ============================================================================
# END OF USER CONFIGURATION
# ============================================================================


def get_datasets_root() -> Path:
    """
    Get the datasets root directory.

    Returns:
        Path object pointing to the datasets root directory
    """
    return DATASETS_ROOT.expanduser().resolve()


def get_results_root() -> Path:
    """
    Get the results root directory.

    Returns:
        Path object pointing to the results root directory
    """
    return RESULTS_ROOT.expanduser().resolve()


def get_configs_root() -> Path:
    """
    Get the configs root directory.

    Returns:
        Path object pointing to the configs root directory
    """
    return CONFIGS_ROOT.expanduser().resolve()


def get_dataset_path(dataset_name: str) -> Path:
    """
    Get the full path to a specific dataset.

    Args:
        dataset_name: Dataset directory name (e.g., 'Dataset001_Hippo')

    Returns:
        Full path to the dataset directory
    """
    return get_datasets_root() / dataset_name
