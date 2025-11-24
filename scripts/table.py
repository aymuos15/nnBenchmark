#!/usr/bin/env python3
"""Display metrics from validation/test JSON in a formatted rich table.

Usage:
    python table.py path/to/test.json
    python table.py path/to/validation_history.json
"""

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def load_json(filepath: str) -> dict[str, Any]:
    """Load JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        console.print(f"[red]Error:[/red] Invalid JSON file: {filepath}")
        sys.exit(1)


def create_metrics_table(data: dict[str, Any]) -> Table:
    """Create main metrics table."""
    table = Table(title="📊 Metrics Summary", show_header=True, header_style="bold cyan")

    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right", style="yellow")
    table.add_column("Max", justify="right", style="yellow")
    table.add_column("Cases", justify="right", style="magenta")

    summary = data.get("summary", {})

    for metric_name, metric_data in summary.items():
        mean = metric_data.get("mean", 0)
        std = metric_data.get("std", 0)
        min_val = metric_data.get("min", 0)
        max_val = metric_data.get("max", 0)
        num_cases = metric_data.get("num_cases", 0)

        table.add_row(
            metric_name,
            f"{mean:.4f}",
            f"{std:.4f}",
            f"{min_val:.4f}",
            f"{max_val:.4f}",
            str(num_cases),
        )

    return table


def create_binned_table(data: dict[str, Any]) -> Table:
    """Create binned metrics table."""
    table = Table(
        title="📦 Binned Metrics by Instance Size",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Bin", style="magenta", width=10)
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right", style="yellow")
    table.add_column("Max", justify="right", style="yellow")
    table.add_column("Count", justify="right")

    summary = data.get("summary", {})

    for metric_name, metric_data in summary.items():
        bins = metric_data.get("bins", {})

        if not bins:
            continue

        first_bin = True
        for bin_name in ["all", "0-2cc", "2-10cc", ">10cc"]:
            if bin_name not in bins:
                continue

            bin_data = bins[bin_name]
            mean = bin_data.get("mean", 0)
            std = bin_data.get("std", 0)
            min_val = bin_data.get("min", 0)
            max_val = bin_data.get("max", 0)
            count = bin_data.get("count", 0)

            # Only show metric name for first bin
            display_metric = metric_name if first_bin else ""

            # Color code bins
            bin_style = "yellow"
            if bin_name == "all":
                bin_style = "cyan"
            elif bin_name == "0-2cc":
                bin_style = "red"
            elif bin_name == "2-10cc":
                bin_style = "blue"

            table.add_row(
                display_metric,
                f"[{bin_style}]{bin_name}[/{bin_style}]",
                f"{mean:.4f}" if count > 0 else "—",
                f"{std:.4f}" if count > 0 else "—",
                f"{min_val:.4f}" if count > 0 else "—",
                f"{max_val:.4f}" if count > 0 else "—",
                str(count) if count > 0 else "0",
            )

            first_bin = False

    return table


def create_per_sample_summary(data: dict[str, Any]) -> Table:
    """Create per-sample summary table."""
    table = Table(
        title="📋 Per-Sample Binned Summary (First 10 Samples)",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Sample", style="cyan", width=20)
    table.add_column("Metric", style="magenta", width=20)
    table.add_column("Bin", width=10)
    table.add_column("Count", justify="right")
    table.add_column("Mean", justify="right", style="green")

    summary = data.get("summary", {})
    sample_names = data.get("sample_names", [])

    for metric_name, metric_data in summary.items():
        per_sample_bins = metric_data.get("per_sample_bins", [])

        if not per_sample_bins:
            continue

        # Limit to first 10 samples
        for sample_idx, sample_data in enumerate(per_sample_bins[:10]):
            for bin_name in ["all", "0-2cc", "2-10cc", ">10cc"]:
                if bin_name not in sample_data:
                    continue

                bin_data = sample_data[bin_name]
                count = bin_data.get("count", 0)
                mean = bin_data.get("mean", 0)

                # Only show sample/metric names for "all" bin
                if bin_name == "all":
                    sample_name = sample_names[sample_idx] if sample_idx < len(
                        sample_names
                    ) else f"Sample {sample_idx}"
                    display_sample = sample_name
                    display_metric = metric_name
                else:
                    display_sample = ""
                    display_metric = ""

                # Color code bins
                bin_style = "yellow"
                if bin_name == "all":
                    bin_style = "cyan"
                elif bin_name == "0-2cc":
                    bin_style = "red"
                elif bin_name == "2-10cc":
                    bin_style = "blue"

                table.add_row(
                    display_sample,
                    display_metric,
                    f"[{bin_style}]{bin_name}[/{bin_style}]",
                    str(count) if count > 0 else "0",
                    f"{mean:.4f}" if count > 0 else "—",
                )

    return table


def create_config_table(data: dict[str, Any]) -> Table:
    """Create configuration info table."""
    table = Table(
        title="⚙️  Configuration",
        show_header=False,
        show_footer=False,
        border_style="dim",
    )

    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="green")

    config_keys = ["config_name", "dataset_name", "fold", "model_path"]

    for key in config_keys:
        value = data.get(key, "N/A")
        if value is not None:
            if isinstance(value, str) and len(value) > 80:
                value = value[:77] + "..."
            table.add_row(key.replace("_", " ").title(), str(value))

    return table


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        console.print(
            "[red]Usage:[/red] python table.py [bold]<path/to/json>[/bold]"
        )
        console.print("\nExamples:")
        console.print("  python table.py results/history/test.json")
        console.print("  python table.py results/history/validation_epoch_010.json")
        sys.exit(1)

    json_path = sys.argv[1]

    # Verify file exists
    if not Path(json_path).exists():
        console.print(f"[red]Error:[/red] File not found: {json_path}")
        sys.exit(1)

    # Load JSON
    data = load_json(json_path)

    # Display configuration
    console.print()
    console.print(create_config_table(data))

    # Display main metrics
    console.print()
    console.print(create_metrics_table(data))

    # Display binned metrics if available
    summary = data.get("summary", {})
    has_bins = any("bins" in metric_data for metric_data in summary.values())

    if has_bins:
        console.print()
        console.print(create_binned_table(data))

    # Display per-sample summary if available
    has_per_sample = any(
        "per_sample_bins" in metric_data for metric_data in summary.values()
    )

    if has_per_sample:
        console.print()
        console.print(create_per_sample_summary(data))

    console.print()


if __name__ == "__main__":
    console = Console()
    main()
