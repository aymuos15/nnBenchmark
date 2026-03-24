"""CLI entry point for training."""

import argparse
from pathlib import Path

import yaml

from src.config.load import load_config
from src.config.paths import get_results_root
from src.engines.train.run import run_training


def _resolve_config_path(config: str, dataset: str | None = None) -> Path:
    """Resolve config path (absolute or relative to results dir)."""
    config_path = Path(config)
    if config_path.is_absolute():
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    if dataset is None:
        raise ValueError(f"Relative config path '{config}' requires --dataset argument.")

    resolved = get_results_root() / dataset / "configs" / config
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}\n"
            f"Please ensure you have run 'nnBench.plan --dataset {dataset}' first."
        )
    return resolved


def _apply_overrides(config_path: str, overrides: list[str]) -> str:
    """Apply CLI overrides to config and return path to temp config.

    If no overrides, returns the original path unchanged.
    If overrides exist, loads config, applies them, writes to temp file.
    """
    if not overrides:
        return config_path

    parser = load_config(config_path)
    for override in overrides:
        key, value = override.split("=", 1)
        # Convert :: separator to nested dict access
        keys = key.split("::")
        parsed_value = yaml.safe_load(value)

        if len(keys) == 1:
            parser[keys[0]] = parsed_value
        else:
            # Navigate to parent, set leaf
            current = parser.config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = parsed_value

    # Write modified config to temp file next to original
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=str(Path(config_path).parent),
    )
    yaml.dump(dict(parser.config), tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return tmp.name


def main() -> None:
    """Entry point for nnBench.train CLI command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (required for relative config paths, e.g., Dataset001_Hippo)",
    )
    parser.add_argument(
        "--fresh",
        dest="force_fresh",
        action="store_true",
        help="Force fresh start, delete existing checkpoints and restart training",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides: training::epochs=500 training::batch_size=4",
    )
    args = parser.parse_args()

    resolved = str(_resolve_config_path(args.config, args.dataset))
    config_path = _apply_overrides(resolved, args.override)

    try:
        run_training(config_path, force_fresh=args.force_fresh)
    finally:
        # Clean up temp config if we created one
        if config_path != resolved:
            Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
