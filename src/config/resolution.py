"""Config file path resolution logic."""

from __future__ import annotations

from pathlib import Path

from src.config.paths import get_results_root


def resolve_config_path(config: str, dataset: str | None = None) -> Path:
    """
    Resolve a config file path, supporting both absolute and relative paths.

    If config is an absolute path, returns it as-is. If config is relative and
    a dataset is provided, searches for the config in:
        nnBench_results/<dataset>/<config_name>/

    Args:
        config: Config file path (absolute or relative, e.g., "fold_0.yaml")
        dataset: Dataset name to search in (required if config is relative)

    Returns:
        Resolved Path object pointing to the config file

    Raises:
        ValueError: If config is relative but dataset is not provided
        FileNotFoundError: If resolved config file does not exist
    """
    config_path = Path(config)

    # If absolute path, use it directly
    if config_path.is_absolute():
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    # If relative path, search in results directory with dataset name
    if dataset is None:
        raise ValueError(
            f"Relative config path '{config}' requires --dataset argument."
        )

    # Config should be in: nnBench_results/<dataset>/<config_name>/<config>
    # where config_name is the stem of the config file
    config_name = config_path.stem
    resolved_path = get_results_root() / dataset / config_name / config

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved_path}\n"
            f"Please ensure you have run 'nnBench.plan --dataset {dataset}' first."
        )

    return resolved_path
