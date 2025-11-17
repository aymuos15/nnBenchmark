"""CLI entry point for validation functionality."""

import argparse
from pathlib import Path

from src.engines.validate.run import run_validation


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

    run_validation(
        config_path=args.config,
        dataset=args.dataset,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
