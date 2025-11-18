"""
Test result plot generation.
Contains functions for plotting test scores, including per-class violin plots.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib styles on import (side-effect import - must run before plotting)
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
    num_cases: str | int = "Unknown"

    if metric_name is not None:
        # Use specified metric
        if metric_name in summary and "per_class" in summary[metric_name]:
            per_class_data = summary[metric_name]["per_class"]
            metric_name_found = metric_name
            num_cases = summary[metric_name].get("num_cases", "Unknown")
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
                num_cases = summary[m_name].get("num_cases", "Unknown")
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
    title += f"\nn = {num_cases} cases"
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


def plot_sample_mean_distribution(
    test_history_path: str,
    metric_name: str,
    save_path: str,
    figsize: tuple[int, int] = (6, 4),
    show_points: bool = True,
) -> str:
    """
    Create violin plot showing distribution of per-sample mean scores.

    For each test sample, computes the mean score across all classes,
    then displays the distribution of these sample means as a violin plot.

    Args:
        test_history_path: Path to test_history.json file
        metric_name: Name of metric to plot (e.g., "DiceMetric")
        save_path: Path where plot will be saved
        figsize: Figure size as (width, height)
        show_points: Whether to overlay individual data points on violin plot

    Returns:
        Path to saved plot

    Raises:
        FileNotFoundError: If test_history.json doesn't exist
        ValueError: If metric not found in test_history
    """
    # Load test history
    if not Path(test_history_path).exists():
        raise FileNotFoundError(f"Test history not found: {test_history_path}")

    with open(test_history_path) as f:
        test_history: dict[str, Any] = json.load(f)

    # Validate metric exists
    if metric_name not in test_history.get("per_sample_scores", {}):
        raise ValueError(f"Metric '{metric_name}' not found in test_history")

    # Extract per-sample scores
    per_sample_scores = test_history["per_sample_scores"][metric_name]

    # Compute mean across classes for each sample
    if isinstance(per_sample_scores[0], list):
        # Per-class scores: compute mean across classes
        sample_means = [np.mean(sample) for sample in per_sample_scores]
    else:
        # Scalar scores: already per-sample means
        sample_means = per_sample_scores

    # Get metadata
    dataset_name = test_history.get("dataset_name", "Unknown")
    fold = test_history.get("fold")
    num_cases = len(sample_means)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create violin plot
    parts = ax.violinplot(
        [sample_means],
        positions=[0],
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    # Customize violin colors
    for pc in parts["bodies"]:  # type: ignore[attr-defined]
        pc.set_facecolor("#8dd3c7")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(1.0)

    # Overlay box plot for quartiles
    ax.boxplot(
        [sample_means],
        positions=[0],
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
        x = np.random.normal(0, 0.04, size=len(sample_means))
        ax.plot(
            x,
            sample_means,
            "o",
            alpha=0.3,
            markersize=2,
            color="darkblue",
            markeredgewidth=0,
        )

    # Customize plot
    ax.set_xticks([0])
    ax.set_xticklabels([metric_name])
    ax.set_ylabel(f"{metric_name} Score")
    ax.set_xlabel("Metric")

    # Title with dataset info
    title = f"Sample Mean {metric_name} Scores - {dataset_name}"
    if fold is not None:
        title += f" (Fold {fold})"
    title += f"\nn = {num_cases} cases"
    ax.set_title(title)

    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add mean ± std annotation
    mean_score = np.mean(sample_means)
    std_score = np.std(sample_means)
    y_max = max(sample_means)
    y_offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(
        0,
        y_max + y_offset,
        f"{mean_score:.3f} ± {std_score:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")
    return save_path


def plot_classwise_bar(
    test_history_path: str,
    metric_name: str,
    save_path: str,
    figsize: tuple[int, int] = (6, 4),
) -> str:
    """
    Create bar plot showing mean score per class with error bars.

    If per_class data is not available (scalar metrics), creates a single bar
    representing the overall metric with a generic class label.

    Args:
        test_history_path: Path to test_history.json file
        metric_name: Name of metric to plot (e.g., "DiceMetric")
        save_path: Path where plot will be saved
        figsize: Figure size as (width, height)

    Returns:
        Path to saved plot

    Raises:
        FileNotFoundError: If test_history.json doesn't exist
        ValueError: If metric not found
    """
    # Load test history
    if not Path(test_history_path).exists():
        raise FileNotFoundError(f"Test history not found: {test_history_path}")

    with open(test_history_path) as f:
        test_history: dict[str, Any] = json.load(f)

    # Validate metric exists
    if metric_name not in test_history.get("summary", {}):
        raise ValueError(f"Metric '{metric_name}' not found in test_history")

    summary = test_history["summary"][metric_name]

    # Extract per-class statistics or create from scalar
    if "per_class" in summary:
        # Per-class data available
        per_class = summary["per_class"]
        class_names = list(per_class.keys())
        means = [per_class[name]["mean"] for name in class_names]
        stds = [per_class[name]["std"] for name in class_names]
    else:
        # Scalar data - create single bar with overall statistics
        class_names = ["Overall"]
        means = [summary["mean"]]
        stds = [summary["std"]]

    # Get metadata
    dataset_name = test_history.get("dataset_name", "Unknown")
    fold = test_history.get("fold")
    num_cases = summary.get("num_cases", "Unknown")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot with error bars
    x_pos = np.arange(len(class_names))
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        color="#8dd3c7",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.7,
        error_kw=dict(linewidth=1.5, ecolor="black"),
    )

    # Add value labels on top of bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i,
            mean + std + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names)
    ax.set_ylabel(f"{metric_name} Score")
    ax.set_xlabel("Class")

    # Title with dataset info
    title = f"Classwise Mean {metric_name} Scores - {dataset_name}"
    if fold is not None:
        title += f" (Fold {fold})"
    title += f"\nn = {num_cases} cases"
    ax.set_title(title)

    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")
    return save_path
