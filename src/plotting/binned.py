"""Plotting functions for binned statistics visualization.

This module provides visualization functions for metrics that compute
instance-size-based binned statistics (e.g., CCMetric).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.files import load_json


def plot_binned_bar_chart(
    history_path: str | Path,
    metric_name: str,
    output_path: str | Path,
) -> None:
    """Generate bar chart showing metric performance by instance size bins.

    Creates a bar chart with error bars showing mean ± std for each instance
    size bin (0-2cc, 2-10cc, >10cc). Includes count annotations above bars.

    Args:
        history_path: Path to test or validation history JSON file
        metric_name: Name of the metric to plot (e.g., "CCMetric")
        output_path: Path where the plot PNG will be saved

    Example:
        >>> plot_binned_bar_chart(
        ...     "results/history/test.json",
        ...     "CCMetric",
        ...     "results/plots/test_binned_CCMetric.png"
        ... )
    """
    # Load history JSON
    history = load_json(history_path)

    # Check if metric exists and has binned statistics
    if "summary" not in history or metric_name not in history["summary"]:
        print(f"Warning: Metric '{metric_name}' not found in {history_path}")
        return

    metric_summary = history["summary"][metric_name]
    if "bins" not in metric_summary:
        print(f"Warning: No binned statistics found for '{metric_name}'")
        return

    binned_stats = metric_summary["bins"]

    # Define bin order (exclude "all" as it's redundant with overall metric)
    bin_names = ["0-2cc", "2-10cc", ">10cc"]
    bin_labels = ["0-2 px", "2-10 px", ">10 px"]

    # Extract data for plotting
    means = []
    stds = []
    counts = []

    for bin_name in bin_names:
        if bin_name in binned_stats:
            bin_data = binned_stats[bin_name]
            means.append(bin_data["mean"])
            stds.append(bin_data["std"])
            counts.append(bin_data["count"])
        else:
            # Handle missing bins
            means.append(0.0)
            stds.append(0.0)
            counts.append(0)

    # Create figure
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar positions
    x_pos = np.arange(len(bin_labels))
    bar_width = 0.6

    # Create bars with error bars
    bars = ax.bar(
        x_pos,
        means,
        bar_width,
        yerr=stds,
        capsize=5,
        color="#4A90E2",
        edgecolor="#2E5C8A",
        linewidth=1.5,
        alpha=0.8,
        error_kw={"linewidth": 2, "ecolor": "#2E5C8A"},
    )

    # Add mean value and count annotations above bars
    for _i, (bar, count, mean, std) in enumerate(zip(bars, counts, means, stds)):
        height = mean + std
        # Add mean value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        # Add count below the mean value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.06,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="normal",
            color="gray",
        )

    # Customize plot
    ax.set_xlabel("Instance Size Bin", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{metric_name} Score", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{metric_name} Performance by Instance Size",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.set_ylim(0, 1.0)  # Assuming metrics are normalized to [0, 1]
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add horizontal line at y=0.5 for reference
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved binned bar chart: {output_path}")


def plot_binned_bar_chart_multi_metric(
    history_path: str | Path,
    metric_names: list[str],
    output_path: str | Path,
) -> None:
    """Generate grouped bar chart comparing multiple metrics across bins.

    Creates a grouped bar chart showing multiple metrics side-by-side for
    each instance size bin.

    Args:
        history_path: Path to test or validation history JSON file
        metric_names: List of metric names to compare
        output_path: Path where the plot PNG will be saved

    Example:
        >>> plot_binned_bar_chart_multi_metric(
        ...     "results/history/test.json",
        ...     ["CCMetric", "DiceMetric"],
        ...     "results/plots/test_binned_comparison.png"
        ... )
    """
    # Load history JSON
    history = load_json(history_path)

    # Define bin order
    bin_names = ["0-2cc", "2-10cc", ">10cc"]
    bin_labels = ["0-2 px", "2-10 px", ">10 px"]

    # Collect data for all metrics
    all_means: dict[str, list[float]] = {}
    all_stds: dict[str, list[float]] = {}

    for metric_name in metric_names:
        if metric_name not in history.get("summary", {}):
            continue

        metric_summary = history["summary"][metric_name]
        if "bins" not in metric_summary:
            continue

        binned_stats = metric_summary["bins"]

        means = []
        stds = []

        for bin_name in bin_names:
            if bin_name in binned_stats:
                means.append(binned_stats[bin_name]["mean"])
                stds.append(binned_stats[bin_name]["std"])
            else:
                means.append(0.0)
                stds.append(0.0)

        all_means[metric_name] = means
        all_stds[metric_name] = stds

    if not all_means:
        print(f"Warning: No binned statistics found for any metric in {history_path}")
        return

    # Create figure
    _fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar positions
    x_pos = np.arange(len(bin_labels))
    bar_width = 0.8 / len(all_means)  # Divide space by number of metrics
    colors = ["#4A90E2", "#E94B3C", "#50C878", "#F39C12", "#9B59B6"]

    # Plot bars for each metric
    for i, (metric_name, means) in enumerate(all_means.items()):
        stds = all_stds[metric_name]
        offset = (i - len(all_means) / 2 + 0.5) * bar_width

        ax.bar(
            x_pos + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=3,
            label=metric_name,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    # Customize plot
    ax.set_xlabel("Instance Size Bin", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Metric Comparison by Instance Size",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=11)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved multi-metric binned bar chart: {output_path}")


def plot_fp_tp_fn_bar_chart(
    history_path: str | Path,
    metric_name: str,
    output_path: str | Path,
) -> None:
    """Generate grouped bar chart showing TP/FN/FP counts by instance size bins.

    Creates a grouped bar chart with three groups (one per size bin), where each
    group has three bars: TP (green), FN (red), FP (orange).

    Args:
        history_path: Path to test or validation history JSON file
        metric_name: Name of the metric to plot (e.g., "CCMetric")
        output_path: Path where the plot PNG will be saved

    Example:
        >>> plot_fp_tp_fn_bar_chart(
        ...     "results/history/test.json",
        ...     "CCMetric",
        ...     "results/plots/test_fp_tp_fn_CCMetric.png"
        ... )
    """
    # Load history JSON
    history = load_json(history_path)

    # Check if metric exists and has FP/TP/FN statistics
    if "summary" not in history or metric_name not in history["summary"]:
        print(f"Warning: Metric '{metric_name}' not found in {history_path}")
        return

    metric_summary = history["summary"][metric_name]
    if "fp_tp_fn" not in metric_summary:
        print(f"Warning: No FP/TP/FN statistics found for '{metric_name}'")
        return

    fp_tp_fn_stats = metric_summary["fp_tp_fn"]

    # Define bin order (exclude "all" for cleaner visualization)
    bin_names = ["0-2cc", "2-10cc", ">10cc"]
    bin_labels = ["0-2 px", "2-10 px", ">10 px"]

    # Extract data for plotting
    tp_counts = []
    fn_counts = []
    fp_counts = []

    for bin_name in bin_names:
        if bin_name in fp_tp_fn_stats:
            bin_data = fp_tp_fn_stats[bin_name]
            tp_counts.append(bin_data.get("TP", 0))
            fn_counts.append(bin_data.get("FN", 0))
            fp_counts.append(bin_data.get("FP", 0))
        else:
            # Handle missing bins
            tp_counts.append(0)
            fn_counts.append(0)
            fp_counts.append(0)

    # Create figure
    _fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar positions
    x_pos = np.arange(len(bin_labels))
    bar_width = 0.25

    # Create grouped bars
    bars_tp = ax.bar(
        x_pos - bar_width,
        tp_counts,
        bar_width,
        label="True Positive (TP)",
        color="#50C878",  # Green
        edgecolor="#2E7D4E",
        linewidth=1.5,
        alpha=0.8,
    )

    bars_fn = ax.bar(
        x_pos,
        fn_counts,
        bar_width,
        label="False Negative (FN)",
        color="#E94B3C",  # Red
        edgecolor="#A63228",
        linewidth=1.5,
        alpha=0.8,
    )

    bars_fp = ax.bar(
        x_pos + bar_width,
        fp_counts,
        bar_width,
        label="False Positive (FP)",
        color="#F39C12",  # Orange
        edgecolor="#B7740E",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add count labels on top of bars
    # Calculate max height for label positioning
    all_counts = tp_counts + fn_counts + fp_counts
    max_count = max(all_counts) if all_counts else 1

    for bars, counts in [
        (bars_tp, tp_counts),
        (bars_fn, fn_counts),
        (bars_fp, fp_counts),
    ]:
        for bar, count in zip(bars, counts):
            if count > 0:  # Only show label if count is non-zero
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_count * 0.02,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    # Customize plot
    ax.set_xlabel("Instance Size Bin", fontsize=14, fontweight="bold")
    ax.set_ylabel("Instance Count", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{metric_name} - TP/FN/FP Counts by Instance Size",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved FP/TP/FN bar chart: {output_path}")


def plot_per_sample_fp_tp_fn(
    history_path: str | Path,
    metric_name: str,
    output_path: str | Path,
    sample_names: list[str] | None = None,
) -> None:
    """Generate line plot showing TP/FN/FP counts across samples.

    Creates a line plot with three lines (TP, FN, FP) showing how counts
    vary across samples. Useful for identifying problematic samples.

    Args:
        history_path: Path to test or validation history JSON file
        metric_name: Name of the metric to plot (e.g., "CCMetric")
        output_path: Path where the plot PNG will be saved
        sample_names: Optional list of sample names for x-axis labels

    Example:
        >>> plot_per_sample_fp_tp_fn(
        ...     "results/history/test.json",
        ...     "CCMetric",
        ...     "results/plots/test_per_sample_fp_tp_fn_CCMetric.png"
        ... )
    """
    # Load history JSON
    history = load_json(history_path)

    # Check if metric exists and has per-sample FP/TP/FN statistics
    if "summary" not in history or metric_name not in history["summary"]:
        print(f"Warning: Metric '{metric_name}' not found in {history_path}")
        return

    metric_summary = history["summary"][metric_name]
    if "per_sample_fp_tp_fn" not in metric_summary:
        print(f"Warning: No per-sample FP/TP/FN statistics found for '{metric_name}'")
        return

    per_sample_stats = metric_summary["per_sample_fp_tp_fn"]

    # Extract TP/FN/FP counts for "all" bin across samples
    tp_counts = [sample["all"]["TP"] for sample in per_sample_stats]
    fn_counts = [sample["all"]["FN"] for sample in per_sample_stats]
    fp_counts = [sample["all"]["FP"] for sample in per_sample_stats]

    # Create figure
    _fig, ax = plt.subplots(figsize=(14, 6))

    # Sample indices for x-axis
    sample_indices = np.arange(len(tp_counts))

    # Plot lines
    ax.plot(
        sample_indices,
        tp_counts,
        marker="o",
        linewidth=2,
        markersize=4,
        label="True Positive (TP)",
        color="#50C878",
        alpha=0.8,
    )

    ax.plot(
        sample_indices,
        fn_counts,
        marker="s",
        linewidth=2,
        markersize=4,
        label="False Negative (FN)",
        color="#E94B3C",
        alpha=0.8,
    )

    ax.plot(
        sample_indices,
        fp_counts,
        marker="^",
        linewidth=2,
        markersize=4,
        label="False Positive (FP)",
        color="#F39C12",
        alpha=0.8,
    )

    # Customize plot
    ax.set_xlabel("Sample Index", fontsize=14, fontweight="bold")
    ax.set_ylabel("Instance Count", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{metric_name} - TP/FN/FP Counts Per Sample",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(axis="both", alpha=0.3, linestyle="--")

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved per-sample FP/TP/FN plot: {output_path}")


__all__ = [
    "plot_binned_bar_chart",
    "plot_binned_bar_chart_multi_metric",
    "plot_fp_tp_fn_bar_chart",
    "plot_per_sample_fp_tp_fn",
]
