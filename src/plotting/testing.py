"""
Test result plot generation.
Contains functions for plotting test scores, including per-class violin plots.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib styles on import
from src.plotting.styles import mpl  # noqa: F401


def plot_classwise_scores(
    test_history_path: str,
    save_path: str | None = None,
    figsize: tuple[int, int] = (6, 4),
    show_points: bool = True,
    metric_name: str | None = None,
) -> str:
    """
    Create violin plot with box overlay showing per-class test scores using scienceplots.

    Args:
        test_history_path: Path to test_history.json file
        save_path: Optional custom save path. If None, saves next to test_history.json
        figsize: Figure size as (width, height)
        show_points: Whether to overlay individual data points on violin plot
        metric_name: Optional specific metric to plot. If None, uses first metric with per_class data

    Returns:
        Path to saved plot

    Raises:
        FileNotFoundError: If test_history.json doesn't exist
        ValueError: If no per_class data found in test_history
    """
    # Load test history
    if not Path(test_history_path).exists():
        raise FileNotFoundError(f"Test history not found: {test_history_path}")

    with open(test_history_path) as f:
        test_history: dict[str, Any] = json.load(f)

    # Extract per-class scores from multi-metric format
    if "summary" not in test_history:
        raise ValueError("No summary data found in test_history.")

    if "metrics" not in test_history or not isinstance(test_history["metrics"], list):
        raise ValueError(
            "Invalid test_history format. Expected multi-metric format with 'metrics' list. "
            "Please regenerate test results with the current version."
        )

    summary = test_history["summary"]
    per_class_data = None
    metric_name_found = None
    num_samples: str | int = "Unknown"

    if metric_name is not None:
        # Use specified metric
        if metric_name in summary and "per_class" in summary[metric_name]:
            per_class_data = summary[metric_name]["per_class"]
            metric_name_found = metric_name
            num_samples = summary[metric_name].get("num_samples", "Unknown")
        else:
            raise ValueError(
                f"Metric '{metric_name}' not found or has no per_class data in test_history."
            )
    else:
        # Use the first metric with per_class data
        for m_name in test_history["metrics"]:
            if m_name in summary and "per_class" in summary[m_name]:
                per_class_data = summary[m_name]["per_class"]
                metric_name_found = m_name
                num_samples = summary[m_name].get("num_samples", "Unknown")
                break

        if per_class_data is None:
            raise ValueError(
                "No per_class data found in test_history for any metric. "
                "Make sure your metrics have include_background=False and return per-class scores."
            )

    metric_name = metric_name_found

    dataset_name = test_history.get("dataset_name", "Unknown")
    fold = test_history.get("fold")

    # Prepare data for plotting
    class_names = list(per_class_data.keys())
    class_scores = [per_class_data[name]["all_scores"] for name in class_names]

    # Create figure using matplotlib
    fig, ax = plt.subplots(figsize=figsize)

    # Create violin plot
    parts = ax.violinplot(
        class_scores,
        positions=range(len(class_names)),
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    # Customize violin colors
    # Type ignore because matplotlib's violinplot returns a dict with Collection objects
    for pc in parts["bodies"]:  # type: ignore[attr-defined]
        pc.set_facecolor("#8dd3c7")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(1.0)

    # Overlay box plot for quartiles
    ax.boxplot(
        class_scores,
        positions=range(len(class_names)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="red", linewidth=1.5),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
    )

    # Overlay individual points if requested
    if show_points:
        for i, scores in enumerate(class_scores):
            # Add jitter to x-coordinates for better visibility
            x = np.random.normal(i, 0.04, size=len(scores))
            ax.plot(
                x,
                scores,
                "o",
                alpha=0.3,
                markersize=2,
                color="darkblue",
                markeredgewidth=0,
            )

    # Customize plot
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_ylabel(f"{metric_name} Score")
    ax.set_xlabel("Class")

    # Title with dataset info
    title = f"Per-Class {metric_name} Scores - {dataset_name}"
    if fold is not None:
        title += f" (Fold {fold})"
    title += f"\nn = {num_samples} samples"
    ax.set_title(title)

    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add mean scores as text annotations
    for i, name in enumerate(class_names):
        mean_score = per_class_data[name]["mean"]
        std_score = per_class_data[name]["std"]
        # Position text above the plot
        y_max = max(class_scores[i])
        y_offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(
            i,
            y_max + y_offset,
            f"{mean_score:.3f} ± {std_score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Determine save path
    if save_path is None:
        results_dir = str(Path(test_history_path).parent)
        save_path = str(Path(results_dir) / "test_cls_wise_scores.png")

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path
