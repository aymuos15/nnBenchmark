"""CLI entry point for validation functionality."""

import argparse
from pathlib import Path

from src.config.paths import get_results_root
from src.engines.validate.run import run_validation


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


def main() -> None:
    """Entry point for nnBench.validate CLI command."""
    parser = argparse.ArgumentParser(
        description="Run validation on trained model checkpoints. "
        "Validates all epoch checkpoints for the given config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (e.g., fold_0.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (required for relative config paths, e.g., Dataset001_Cellpose)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file to validate (optional). If not provided, validates all epoch checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for validation (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loader workers (overrides config)",
    )
    args = parser.parse_args()

    resolved = str(_resolve_config_path(args.config, args.dataset))

    run_validation(
        config_path=resolved,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
