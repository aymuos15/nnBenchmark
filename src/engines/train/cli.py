"""CLI entry point for training."""

import argparse

from src.engines.train.run import run_training


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
        "--continue",
        "-c",
        dest="resume",
        action="store_true",
        help="Resume training from last checkpoint (deprecated - now automatic)",
    )
    parser.add_argument(
        "--fresh",
        "--no-resume",
        dest="force_fresh",
        action="store_true",
        help="Force fresh start, delete existing checkpoints",
    )
    args = parser.parse_args()
    run_training(
        args.config, dataset=args.dataset, resume=args.resume, force_fresh=args.force_fresh
    )


if __name__ == "__main__":
    main()
