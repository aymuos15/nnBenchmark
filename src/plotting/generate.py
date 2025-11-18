"""
Generate training plots from saved training history.
Uses SciencePlots for publication-ready figures.
"""

import json
from pathlib import Path

from src.config.load import load_training_history, load_validation_histories
from src.plotting.inference import (
    plot_classwise_bar,
    plot_sample_mean_distribution,
)
from src.plotting.training import plot_training_loss
from src.plotting.validation import plot_validation_metric


def generate_plots(results_dir: str) -> None:
    """
    Generate all plots from training, validation, and test results.

    Creates:
    1. Training loss plot (training_loss.png)
    2. Validation metric plots (val_{metric}.png for each metric, if validation data exists)
    3. Classwise test scores plot (test_cls_wise_{metric}_scores.png, if test_history.json exists)

    Args:
        results_dir: Directory containing training_history.json, validation_history_epoch_*.json,
                     and optionally test_history.json
    """
    print(f"Loading results from: {results_dir}")

    # Create plots subdirectory
    plots_dir = str(Path(results_dir) / "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Load training history
    history = load_training_history(results_dir)

    # Validate required keys
    if "epochs" not in history or "train_loss" not in history:
        raise ValueError("Training history must contain 'epochs' and 'train_loss' keys")

    epochs: list[int] = [int(e) for e in history["epochs"]]
    train_loss: list[float] = history["train_loss"]

    if len(epochs) == 0:
        print("Warning: No training data found in history")
        return

    # Load validation histories (if available)
    val_data = load_validation_histories(results_dir)
    if val_data:
        print(f"  Found validation data for {len(val_data['val_epochs'])} epochs")
        # Merge validation data into history for plotting
        history.update(val_data)
    else:
        print("  No validation history files found")

    # 1. Plot training loss
    loss_plot_path = str(Path(plots_dir) / "training_loss.png")
    plot_training_loss(epochs, train_loss, loss_plot_path)

    # 2. Plot validation metrics (each metric gets its own plot)
    # Collect all validation metrics (treat all as main metrics unless we implement per-class detection)
    main_val_metrics = {}

    for k, v in history.items():
        if k.startswith("val_") and k != "val_epochs":
            # Extract metric name (everything after "val_")
            # This handles metric names with underscores like "CCMetric_dice"
            metric_name = k.replace("val_", "")
            main_val_metrics[metric_name] = v

    if not main_val_metrics:
        print("  No validation metrics found in history")
    else:
        # Get validation epochs
        val_epochs: list[int] = [int(e) for e in history.get("val_epochs", [])]

        for metric_name, metric_values in main_val_metrics.items():
            if len(metric_values) > 0:
                # Validate val_epochs length matches metric_values
                if len(val_epochs) != len(metric_values):
                    raise ValueError(
                        f"val_epochs length ({len(val_epochs)}) does not match "
                        f"{metric_name} values length ({len(metric_values)})"
                    )
                epochs_for_plot = val_epochs

                metric_plot_path = str(Path(plots_dir) / f"val_{metric_name}.png")
                plot_validation_metric(
                    epochs_for_plot,
                    metric_values,
                    metric_name,
                    metric_plot_path,
                    per_class_values=None,  # Not supported yet for scalar metrics
                )

    # 3. Plot test results if available
    test_history_path = str(Path(results_dir) / "history" / "test.json")
    if Path(test_history_path).exists():
        # Load test history to check for metrics
        with open(test_history_path) as f:
            test_history = json.load(f)

        # Check if this is the multi-metric format
        if "metrics" in test_history and isinstance(test_history["metrics"], list):
            print(f"  Found test data with {len(test_history['metrics'])} metrics")

            # Plot each metric: create 2 plots per metric
            for metric_name in test_history["metrics"]:
                if metric_name not in test_history["summary"]:
                    continue

                # 1. Violin plot: Distribution of sample mean scores
                try:
                    sample_mean_path = str(
                        Path(plots_dir) / f"test_sample_mean_{metric_name}.png"
                    )
                    plot_sample_mean_distribution(
                        test_history_path=test_history_path,
                        metric_name=metric_name,
                        save_path=sample_mean_path,
                        figsize=(6, 4),
                        show_points=True,
                    )
                except Exception as e:
                    print(
                        f"  Warning: Could not create sample mean plot for {metric_name}: {e}"
                    )

                # 2. Bar plot: Classwise mean scores
                try:
                    bar_path = str(
                        Path(plots_dir) / f"test_classwise_bar_{metric_name}.png"
                    )
                    plot_classwise_bar(
                        test_history_path=test_history_path,
                        metric_name=metric_name,
                        save_path=bar_path,
                        figsize=(6, 4),
                    )
                except Exception as e:
                    print(
                        f"  Warning: Could not create classwise bar plot for {metric_name}: {e}"
                    )
        else:
            print(
                "  Note: Test history not in multi-metric format, skipping test plots"
            )

    print(f"\nAll plots saved to: {plots_dir}")
