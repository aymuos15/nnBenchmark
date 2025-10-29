"""
CLI entry point for inference/testing functionality.
"""

import argparse

from src.inference.run import run_testing


def main() -> None:
    """Entry point for nnBench.test CLI command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument(
        "--use-val-split",
        action="store_true",
        help="Use validation split instead of dedicated test set (imagesTs/labelsTs)",
    )
    args = parser.parse_args()
    # Default is to use test set (use_test_set=True), unless --use-val-split is specified
    use_test_set = not args.use_val_split
    run_testing(args.config, args.model, use_test_set)


if __name__ == "__main__":
    main()
