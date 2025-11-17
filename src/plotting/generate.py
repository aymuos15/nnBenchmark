"""
Generate training plots from saved training history.
Uses SciencePlots for publication-ready figures.
"""

import json
from pathlib import Path

from src.config.load import load_training_history
from src.plotting.inference import plot_classwise_scores
from src.plotting.training import plot_training_loss
from src.plotting.validation import plot_validation_metric


def generate_plots(results_dir: str) -> None:
    """
    Generate all plots from training and test results.

    Creates:
    1. Training loss plot (training_loss.png)
    2. Validation metric plots (val_{metric}.png for each metric)
    3. Classwise test scores plot (classwise_scores.png, if test_history.json exists)

    Args:
        results_dir: Directory containing training_history.json and optionally test_history.json
    """
    print(f"Loading training history from: {results_dir}")

    # Create plots subdirectory
    plots_dir = str(Path(results_dir) / "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Load history
    history = load_training_history(results_dir)

    # Validate required keys
    if "epochs" not in history or "train_loss" not in history:
        raise ValueError("Training history must contain 'epochs' and 'train_loss' keys")

    epochs: list[int] = [int(e) for e in history["epochs"]]
    train_loss: list[float] = history["train_loss"]

    if len(epochs) == 0:
        print("Warning: No training data found in history")
        return

    # 1. Plot training loss
    loss_plot_path = str(Path(plots_dir) / "training_loss.png")
    plot_training_loss(epochs, train_loss, loss_plot_path)

    # 2. Plot validation metrics (each metric gets its own plot)
    # Separate main metrics from per-class metrics
    main_val_metrics = {}
    per_class_metrics = {}

    for k, v in history.items():
        if k.startswith("val_") and k != "val_epochs":
            # Check if this is a per-class metric (contains underscore after metric name)
            # e.g., "val_Dice_Anterior" vs "val_Dice"
            parts = k.replace("val_", "").split("_", 1)
            if len(parts) == 1:
                # Main metric (e.g., "val_Dice")
                metric_name = parts[0]
                main_val_metrics[metric_name] = v
            else:
                # Per-class metric (e.g., "val_Dice_Anterior")
                metric_name, class_name = parts
                if metric_name not in per_class_metrics:
                    per_class_metrics[metric_name] = {}
                per_class_metrics[metric_name][class_name] = v

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

                # Get per-class values for this metric if available
                per_class_vals = per_class_metrics.get(metric_name, None)

                metric_plot_path = str(Path(plots_dir) / f"val_{metric_name}.png")
                plot_validation_metric(
                    epochs_for_plot,
                    metric_values,
                    metric_name,
                    metric_plot_path,
                    per_class_values=per_class_vals,
                )

    # 3. Plot test results if available (classwise violin plot for each metric)
    test_history_path = str(Path(results_dir) / "test_history.json")
    if Path(test_history_path).exists():
        try:
            # Load test history to check for multiple metrics
            with open(test_history_path) as f:
                test_history = json.load(f)

            # Check if this is the new multi-metric format
            if "metrics" in test_history and isinstance(test_history["metrics"], list):
                # Plot each metric separately
                for metric_name in test_history["metrics"]:
                    if (
                        metric_name in test_history["summary"]
                        and "per_class" in test_history["summary"][metric_name]
                    ):
                        save_path = str(
                            Path(plots_dir) / f"test_cls_wise_{metric_name}_scores.png"
                        )
                        classwise_plot_path = plot_classwise_scores(
                            test_history_path=test_history_path,
                            save_path=save_path,
                            figsize=(6, 4),
                            show_points=True,
                            metric_name=metric_name,
                        )
                        print(f"Saved: {classwise_plot_path}")
        except ValueError as e:
            # Per-class data not available, skip this plot
            print(f"  Note: Skipping per-class plots (data not available: {e})")

    print(f"\nAll plots saved to: {plots_dir}")
