"""CLI entry point for plot generation."""

import argparse
import json
from pathlib import Path

from src.config import resolve_config_path
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
    training_json = results_dir / "history" / "training.json"
    try:
        if training_json.exists():
            with open(training_json) as f:
                training_data = json.load(f)
            epochs = training_data.get("epochs", [])
            train_loss = training_data.get("train_loss", [])
            if epochs and train_loss:
                save_path = str(Path(plots_dir) / "training_loss.png")
                plot_training_loss(epochs, train_loss, save_path)
    except Exception as e:
        from loguru import logger

        logger.warning(f"Could not generate training loss plot: {e}")

    # Generate plots from validation history (aggregate from epoch files)
    history_dir = results_dir / "history"
    try:
        val_files = sorted(history_dir.glob("validation_epoch_*.json"))
        if val_files:
            epochs: list[int] = []
            metric_values: list[float] = []
            metric_name = "Dice"  # Default metric name

            for val_file in val_files:
                with open(val_file) as f:
                    val_data = json.load(f)
                epoch = val_data.get("epoch")
                summary = val_data.get("summary", {})
                # Get first metric's mean value
                if summary and epoch is not None:
                    for name, stats in summary.items():
                        metric_name = name
                        if isinstance(stats, dict) and "mean" in stats:
                            epochs.append(epoch)
                            metric_values.append(stats["mean"])
                        break

            if epochs and metric_values:
                save_path = str(
                    Path(plots_dir) / f"validation_{metric_name.lower()}.png"
                )
                plot_validation_metric(epochs, metric_values, metric_name, save_path)
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
