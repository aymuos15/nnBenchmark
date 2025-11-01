"""
CLI entry point for inference functionality.
"""

import argparse

from src.inference.run import run_inference


def main() -> None:
    """Entry point for nnBench.inference CLI command."""
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
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument(
        "--use-val-split",
        action="store_true",
        help="Use validation split instead of dedicated test set (imagesTs/labelsTs)",
    )
    args = parser.parse_args()
    # Default is to use test set (use_test_set=True), unless --use-val-split is specified
    use_test_set = not args.use_val_split
    run_inference(args.config, args.model, use_test_set, dataset=args.dataset)


if __name__ == "__main__":
    main()
