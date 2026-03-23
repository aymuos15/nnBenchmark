"""
Dataset and results paths configuration.

This module uses environment variables to locate datasets, preprocessed data,
and results. Set these environment variables:

    export nnBench_raw="/path/to/nnBench_raw"
    export nnBench_preprocessed="/path/to/nnBench_preprocessed"
    export nnBench_results="/path/to/nnBench_results"

For backward compatibility, the old nnUNet_* variables are still supported
but deprecated. If both are set, nnBench_* takes precedence.

Add these to your ~/.bashrc or ~/.zshrc for persistence.
"""


import os
import warnings
from pathlib import Path


def _get_env_path_with_fallback(new_var: str, old_var: str, description: str) -> Path:
    """
    Get a required path from environment variable with backward compatibility.

    Checks for new_var first, then falls back to old_var with deprecation warning.

    Args:
        new_var: Name of new environment variable (e.g., 'nnBench_raw')
        old_var: Name of old environment variable (e.g., 'nnUNet_raw')
        description: Human-readable description for error messages

    Returns:
        Path from environment variable

    Raises:
        RuntimeError: If neither environment variable is set or path parent doesn't exist
    """
    new_value = os.environ.get(new_var)
    old_value = os.environ.get(old_var)

    if new_value is not None:
        # New variable is set, use it
        env_var = new_var
        value = new_value
    elif old_value is not None:
        # Fall back to old variable with deprecation warning
        warnings.warn(
            f"Environment variable '{old_var}' is deprecated and will be removed in a future version.\n"
            f"Please use '{new_var}' instead:\n"
            f"  export {new_var}='{old_value}'\n"
            f"Update your ~/.bashrc or ~/.zshrc accordingly.",
            DeprecationWarning,
            stacklevel=3,
        )
        env_var = old_var
        value = old_value
    else:
        # Neither variable is set
        raise RuntimeError(
            f"Environment variable '{new_var}' (or '{old_var}') is not set.\n"
            f"This should point to your {description}.\n"
            f"Set it with: export {new_var}='/path/to/{description}'\n"
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
    Get the datasets root directory from nnBench_raw environment variable.

    Falls back to nnUNet_raw for backward compatibility (with deprecation warning).

    Returns:
        Path object pointing to the datasets root directory

    Raises:
        RuntimeError: If neither nnBench_raw nor nnUNet_raw is set
    """
    return _get_env_path_with_fallback("nnBench_raw", "nnUNet_raw", "nnBench_raw")


def get_preprocessed_root() -> Path:
    """
    Get the preprocessed data root directory from nnBench_preprocessed.

    Falls back to nnUNet_preprocessed for backward compatibility (with deprecation warning).

    Returns:
        Path object pointing to the preprocessed root directory

    Raises:
        RuntimeError: If neither nnBench_preprocessed nor nnUNet_preprocessed is set
    """
    return _get_env_path_with_fallback(
        "nnBench_preprocessed", "nnUNet_preprocessed", "nnBench_preprocessed"
    )


def get_results_root() -> Path:
    """
    Get the results root directory from nnBench_results environment variable.

    Falls back to nnUNet_results for backward compatibility (with deprecation warning).

    Returns:
        Path object pointing to the results root directory

    Raises:
        RuntimeError: If neither nnBench_results nor nnUNet_results is set
    """
    return _get_env_path_with_fallback(
        "nnBench_results", "nnUNet_results", "nnBench_results"
    )


