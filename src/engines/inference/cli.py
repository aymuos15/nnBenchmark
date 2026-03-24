"""
CLI entry point for inference functionality.
"""

import argparse
from pathlib import Path

from src.config.paths import get_results_root
from src.engines.inference.run import run_inference


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
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input image folder"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output prediction folder"
    )
    args = parser.parse_args()

    resolved = str(_resolve_config_path(args.config, args.dataset))
    use_test_set = not args.use_val_split

    run_inference(
        resolved, args.model, use_test_set,
        input_folder=args.input, output_folder=args.output,
    )


if __name__ == "__main__":
    main()
