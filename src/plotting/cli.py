"""CLI entry point for plot generation."""

import argparse
from pathlib import Path

from src.plotting.generate import generate_plots
from src.utils.runner import get_config_name, setup_results_dir


def main() -> None:
    """Main function for plot generation."""
    parser = argparse.ArgumentParser(
        description="Generate all plots from training and test results using SciencePlots",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )

    args = parser.parse_args()

    # Determine results directory from config name
    config_name = get_config_name(args.config)
    results_dir = setup_results_dir(config_name, create=False)

    # Validate results directory exists
    if not Path(results_dir).exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Make sure you've trained the model first with:\n"
            f"  python -m src.train --config {args.config}"
        )

    # Generate all plots
    generate_plots(results_dir)


if __name__ == "__main__":
    main()
