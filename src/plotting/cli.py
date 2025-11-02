"""CLI entry point for plot generation."""

import argparse
from pathlib import Path

from src.config import resolve_config_path
from src.engines.common import get_config_name, setup_results_dir
from src.plotting.generate import generate_plots


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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (required for relative config paths, e.g., Dataset001_Hippo)",
    )

    args = parser.parse_args()

    # Resolve config path (handles both absolute and relative paths)
    resolved_config_path = str(resolve_config_path(args.config, args.dataset))

    # Determine results directory from config name
    config_name = get_config_name(resolved_config_path)
    dataset_name = Path(resolved_config_path).parent.parent.name
    results_dir = setup_results_dir(config_name, dataset_name, create=False)

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
