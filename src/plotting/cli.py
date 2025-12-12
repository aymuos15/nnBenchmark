"""CLI entry point for plot generation."""

import argparse
from pathlib import Path

from src.config import resolve_config_path
from src.engines.setup import get_config_name, setup_results_dir
from src.plotting.inference import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import plot_validation_metric, save_validation_visualizations


def main() -> None:
    """Main function for plot generation."""
    parser = argparse.ArgumentParser(
        description="Generate plots from training and validation results using SciencePlots",
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

    # Generate plots from training history
    try:
        plot_training_loss(str(results_dir / "history" / "training_history.json"), str(results_dir / "plots"))
    except Exception as e:
        from loguru import logger
        logger.warning(f"Could not generate training loss plot: {e}")

    # Generate plots from validation history
    try:
        plot_validation_metric(str(results_dir / "history" / "validation_history.json"), str(results_dir / "plots"))
    except Exception as e:
        from loguru import logger
        logger.warning(f"Could not generate validation metric plot: {e}")

    # Generate classwise score plots if test results exist
    test_json = results_dir / "history" / "test.json"
    if test_json.exists():
        try:
            plot_classwise_scores(str(test_json), str(results_dir / "plots"))
        except Exception as e:
            from loguru import logger
            logger.warning(f"Could not generate classwise scores plot: {e}")


if __name__ == "__main__":
    main()
