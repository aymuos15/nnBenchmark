#!/usr/bin/env python3
"""
Script to read test_history.json files from nnUNet results and create a summary table.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    from rich.table import Table
    from rich.console import Console
except ImportError:
    print("Error: rich package not found. Install it with: pip install rich")
    sys.exit(1)


def collect_test_results(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Collect all test results from test_history.json files in subdirectories.

    Args:
        dataset_path: Path to the dataset directory in nnUNet_results

    Returns:
        List of dictionaries containing experiment results with summary stats
    """
    results = []

    # Find all test_history.json files
    test_history_files = list(dataset_path.glob("*/test_history.json"))

    if not test_history_files:
        print(f"No test_history.json files found in {dataset_path}")
        return results

    for test_file in sorted(test_history_files):
        experiment_name = test_file.parent.name

        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)

            # Extract summary statistics
            result_row = {"Experiment": experiment_name}

            # Look for summary statistics in the data
            if isinstance(test_data, dict):
                # Check if there's a 'summary' key with metric statistics
                if 'summary' in test_data:
                    summary = test_data['summary']
                    for metric_name, stats in summary.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            result_row[f"{metric_name}_mean"] = stats['mean']
                            result_row[f"{metric_name}_std"] = stats.get('std', 0)
                else:
                    # If no summary key, extract available top-level metrics
                    for key, value in test_data.items():
                        if isinstance(value, (int, float)):
                            result_row[key] = value

            results.append(result_row)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {test_file}: {e}", file=sys.stderr)

    return results


def format_results_table(results: List[Dict[str, Any]]) -> None:
    """
    Display results as a rich table with 4 columns: Experiment, Dice, CCMetric, SurfaceDice.
    Each metric shows mean ± std.

    Args:
        results: List of result dictionaries
    """
    if not results:
        print("No results found.")
        return

    # Define the 4 metrics we want to display
    # Each tuple is (display_name, [possible_mean_keys], [possible_std_keys])
    metrics = [
        ("Dice", ["DiceMetric_mean"], ["DiceMetric_std"]),
        ("CC-Dice", ["CCMetric_dice_mean", "CCMetric_mean"], ["CCMetric_dice_std", "CCMetric_std"]),
        ("NSD", ["SurfaceDiceMetric_mean"], ["SurfaceDiceMetric_std"]),
        ("CC-NSD", ["CCMetric_surface_dice_mean"], ["CCMetric_surface_dice_std"]),
    ]

    # Create rich table
    table = Table(title="Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Experiment", style="cyan")

    # Add metric columns
    for display_name, _, _ in metrics:
        table.add_column(display_name, style="cyan")

    # Add rows
    for result in results:
        row = [result.get("Experiment", "")]

        for display_name, mean_keys, std_keys in metrics:
            # Try to find the metric in the result using possible key variations
            mean_val = None
            std_val = None

            for mean_key in mean_keys:
                if mean_key in result:
                    mean_val = result[mean_key]
                    break

            for std_key in std_keys:
                if std_key in result:
                    std_val = result[std_key]
                    break

            # Format as mean ± std
            if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                formatted = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                formatted = ""

            row.append(formatted)

        table.add_row(*row)

    # Display table
    console = Console()
    console.print(table)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python results.py <dataset_name>")
        print("\nExample: python results.py Dataset001_Cellpose")
        print("\nAvailable datasets in ../SegData/nnUNet_results:")

        results_dir = Path("../SegData/nnUNet_results")
        if results_dir.exists():
            datasets = [d.name for d in results_dir.iterdir() if d.is_dir()]
            for dataset in sorted(datasets):
                print(f"  - {dataset}")
        else:
            print("  (nnUNet_results directory not found)")

        sys.exit(1)

    dataset_name = sys.argv[1]
    dataset_path = Path("../SegData/nnUNet_results") / dataset_name

    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Collect and display results
    results = collect_test_results(dataset_path)

    if results:
        console = Console()
        console.print(f"\n[bold]Test Results for {dataset_name}[/bold]")
        format_results_table(results)
    else:
        print(f"No test results found for {dataset_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
