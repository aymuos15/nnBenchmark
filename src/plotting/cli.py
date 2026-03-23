"""CLI entry point for plot generation."""

import argparse
from pathlib import Path

from src.config import resolve_config_path
from src.config.load import load_training_history, load_validation_histories
from src.engines.setup import get_config_name, setup_results_dir
from src.plotting.inference import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import plot_validation_metric
from src.utils.files import ensure_directory


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
    results_dir_str = setup_results_dir(config_name, dataset_name, create=False)
    results_dir = Path(results_dir_str)

    # Validate results directory exists
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Make sure you've trained the model first with:\n"
            f"  python -m src.train --config {args.config}"
        )

    # Ensure plots directory exists
    plots_dir = ensure_directory(str(results_dir / "plots"))

    # Generate plots from training history
    try:
        training_data = load_training_history(results_dir_str)
        epochs = training_data.get("epochs", [])
        train_loss = training_data.get("train_loss", [])
        if epochs and train_loss:
            save_path = str(Path(plots_dir) / "training_loss.png")
            plot_training_loss(epochs, train_loss, save_path)
    except FileNotFoundError:
        pass  # No training history yet
    except Exception as e:
        from loguru import logger

        logger.warning(f"Could not generate training loss plot: {e}")

    # Generate plots from validation history (aggregate from epoch files)
    try:
        val_data = load_validation_histories(results_dir_str)
        if val_data:
            val_epochs = val_data.get("val_epochs", [])
            # Find the first metric key (skip "val_epochs" and per-class keys)
            metric_name = "Dice"
            metric_values: list[float] = []
            for key in val_data:
                if key == "val_epochs" or key.count("_") > 1:
                    continue
                metric_name = key.removeprefix("val_")
                metric_values = val_data[key]
                break

            if val_epochs and metric_values:
                save_path = str(
                    Path(plots_dir) / f"validation_{metric_name.lower()}.png"
                )
                plot_validation_metric(val_epochs, metric_values, metric_name, save_path)
    except Exception as e:
        from loguru import logger

        logger.warning(f"Could not generate validation metric plot: {e}")

    # Generate classwise score plots if test results exist
    test_json = results_dir / "history" / "test.json"
    if test_json.exists():
        try:
            plot_classwise_scores(str(test_json), plots_dir)
        except Exception as e:
            from loguru import logger

            logger.warning(f"Could not generate classwise scores plot: {e}")


if __name__ == "__main__":
    main()
