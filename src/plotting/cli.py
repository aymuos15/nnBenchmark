"""CLI entry point for plot generation."""

import argparse
from pathlib import Path

from src.config.load import load_training_history, load_validation_histories
from src.config.paths import get_results_root
from src.engines.setup import get_config_name, setup_results_dir
from src.plotting.inference import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import plot_validation_metric
from src.utils.files import ensure_directory


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

    resolved_config_path = str(_resolve_config_path(args.config, args.dataset))

    config_name = get_config_name(resolved_config_path)
    dataset_name = Path(resolved_config_path).parent.parent.name
    results_dir_str = setup_results_dir(config_name, dataset_name, create=False)
    results_dir = Path(results_dir_str)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Make sure you've trained the model first with:\n"
            f"  python -m src.train --config {args.config}"
        )

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
        pass
    except Exception as e:
        from loguru import logger

        logger.warning(f"Could not generate training loss plot: {e}")

    # Generate plots from validation history
    try:
        val_data = load_validation_histories(results_dir_str)
        if val_data:
            val_epochs = val_data.get("val_epochs", [])
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
