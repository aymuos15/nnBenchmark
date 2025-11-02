"""CLI entry point for automatic experiment planning."""

import argparse

from src.planning.run import run_planning


def main() -> None:
    """Entry point for nnBench.plan CLI command."""
    parser = argparse.ArgumentParser(
        description="Automatically generate optimal training configuration using nnU-Net heuristics",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or path (e.g., Dataset002_Kits or datasets/Dataset002_Kits)",
    )

    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=None,
        help="Target GPU memory in GB (default: auto-detect from system, fallback to 8.0)",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number to use for training (default: 0)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for generated YAML config (default: configs/<dataset_name>.yaml)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for dataset fingerprinting (default: auto-detect, max 8). Set to 1 to disable parallelization",
    )

    args = parser.parse_args()
    run_planning(
        dataset=args.dataset,
        gpu_memory_gb=args.gpu_memory_gb,
        fold=args.fold,
        output=args.output,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
