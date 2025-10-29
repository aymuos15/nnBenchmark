"""
nnBenchmark utilities package.

This package provides utilities for data loading, component building,
and file operations.

Modules:
    data: Data dictionary creation and dataset preparation (get_data_dicts, get_test_data_dicts)
    builders: Factory functions for MONAI components (build_model, build_loss, build_optimizer, etc.)
    files: File system utilities (extract_case_id, ensure_directory, etc.)
    runner: Common experiment setup utilities (setup_experiment, setup_device, etc.)
    logging: Logging utilities for training and testing
"""

__version__ = "0.1.0"
