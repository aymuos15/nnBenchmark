"""
Standalone planning execution function.
"""


import sys
from pathlib import Path

from loguru import logger

from src.config import get_datasets_root, get_preprocessed_root, get_results_root
from src.logging import setup_dual_logging
from src.planning.constants import PLANNING_CONSTANTS
from src.planning.fingerprinting.fingerprint import fingerprint_dataset
from src.planning.fingerprinting.prepare_dataset import prepare_dataset
from src.planning.fingerprinting.resources import (
    get_gpu_memory_for_planning,
    get_system_resources,
)
from src.planning.planner.create import create_experiment_plan
from src.planning.splits import (
    create_splits,
    extract_case_identifiers,
    load_dataset_json,
    save_splits,
)
from src.planning.yaml_generator import generate_config_yaml


def run_planning(
    dataset: str,
    gpu_memory_gb: float | None = None,
    fold: int = 0,
    output: str | None = None,
    num_workers: int | None = None,
) -> None:
    """
    Run automatic experiment planning for a dataset.

    Args:
        dataset: Dataset name or path (e.g., Dataset002_Kits or datasets/Dataset002_Kits)
        gpu_memory_gb: Target GPU memory in GB (auto-detect if None)
        fold: Fold number to use for training (default: 0)
        output: Output path for generated YAML config (default: configs/<dataset_name>.yaml)
        num_workers: Number of parallel workers for fingerprinting (default: auto-detect)

    Raises:
        FileNotFoundError: If dataset directory or dataset.json not found
        Exception: If planning fails
    """

    # Determine dataset directory
    dataset_path = Path(dataset)
    if dataset_path.is_dir():
        dataset_dir = str(dataset_path)
    else:
        # Look in configured datasets root directory
        dataset_dir = str(get_datasets_root() / dataset)

    # Verify dataset directory exists
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Please provide a valid dataset name or path"
        )

    # Determine output path
    if output is None:
        # Auto-generate output path: nnBench_results/<dataset_name>/configs/fold_0.yaml
        dataset_name = dataset_path.name
        config_name = "fold_0"
        output_path = str(
            get_results_root() / dataset_name / "configs" / f"{config_name}.yaml"
        )
    else:
        output_path = output

    # Create results directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Setup dual logging: verbose to file, minimal to console
    dataset_name = dataset_path.name
    preprocessed_dir = get_preprocessed_root() / dataset_name
    log_file = str(preprocessed_dir / "planning.log")
    setup_dual_logging(log_file)

    # Detect GPU memory if not specified
    if gpu_memory_gb is None:
        gpu_memory_gb = get_gpu_memory_for_planning()
        gpu_memory_source = "auto-detected"
    else:
        gpu_memory_source = "user-specified"

    # Detect system resources (before any heavy operations)
    # Using conservative strategy to avoid CUDA initialization errors with too many workers
    system_resources = get_system_resources(
        gpu_memory_gb_override=gpu_memory_gb if gpu_memory_gb else None,
        num_workers_strategy="conservative",
        dataset_size_mb=0.0,  # Will be estimated after fingerprinting
    )

    # Banner
    print("=" * 80)
    print("nnBenchmark Automatic Experiment Planning")
    print("Following nnU-Net heuristics for optimal configuration")
    print("=" * 80)

    print(f"Dataset: {dataset_dir}")
    print(f"GPU memory target: {gpu_memory_gb} GB ({gpu_memory_source})")
    print(f"Output: {output_path}")
    print()

    # Log detected system resources (debug level, goes to file only)
    logger.debug("Detected system resources:")
    logger.debug(
        f"  CPU cores: {system_resources.cpu_count} physical, {system_resources.cpu_logical_count} logical"
    )
    logger.debug(
        f"  RAM: {system_resources.total_ram_gb} GB total, {system_resources.available_ram_gb} GB available"
    )
    if system_resources.gpu_available:
        logger.debug(
            f"  GPU: {system_resources.gpu_name} ({system_resources.gpu_memory_gb} GB usable)"
        )
    else:
        logger.debug("  GPU: Not available")

    try:
        # Step 0: Preprocessing (ALWAYS run, independent of dataset.json/splits.json)
        dataset_json_path = dataset_path / "dataset.json"
        dataset_name = dataset_path.name
        preprocessed_dir = get_preprocessed_root() / dataset_name
        splits_json_path = preprocessed_dir / "splits.json"
        images_dir = preprocessed_dir / "imagesTr"

        # Check if preprocessing has already been done
        if not images_dir.exists():
            from src.planning.fingerprinting.prepare_dataset import (
                preprocess_and_crop_dataset,
            )

            preprocess_and_crop_dataset(
                dataset_path=str(dataset_path),
                output_dir=str(preprocessed_dir),
                force=False,
            )
            logger.debug("Preprocessing complete")
        else:
            logger.debug("Dataset already preprocessed")

        # Step 0b: Prepare dataset metadata if needed
        if not dataset_json_path.exists() or not splits_json_path.exists():
            prepare_dataset(
                dataset_path=str(dataset_path),
                dataset_name=None,  # Use directory name as default
                channel="Unknown",
                num_classes=2,
                description="",
                force=False,
                preprocess=False,  # Skip preprocessing (already done above)
                preprocessed_dir=str(
                    preprocessed_dir
                ),  # Save splits to preprocessed folder
            )
            logger.debug("Dataset metadata generated")

        # Verify dataset.json exists after preparation
        if not dataset_json_path.is_file():
            raise FileNotFoundError(
                f"dataset.json not found in {dataset_dir}\n"
                "Dataset must contain a dataset.json file or have imagesTr/labelsTr directories"
            )

        # Step 1: Fingerprint dataset (using preprocessed/cropped images)
        fingerprint = fingerprint_dataset(
            str(preprocessed_dir), num_workers=num_workers
        )

        # Step 2: Create experiment plan
        plan = create_experiment_plan(fingerprint, gpu_memory_gb=gpu_memory_gb)

        # Step 3: Generate YAML config with resource-optimized settings

        # Update resource detection with actual dataset size for cache optimization
        # Estimate dataset size from preprocessed training files
        images_dir_preprocessed = preprocessed_dir / "imagesTr"
        dataset_size_bytes = sum(
            f.stat().st_size
            for f in images_dir_preprocessed.glob("**/*")
            if f.is_file() and not f.name.startswith("._")
        )
        dataset_size_mb = dataset_size_bytes / (1024 * 1024)

        # Re-calculate resources with actual dataset size for intelligent caching
        # Using conservative strategy to avoid CUDA initialization errors with too many workers
        system_resources = get_system_resources(
            gpu_memory_gb_override=gpu_memory_gb if gpu_memory_gb else None,
            num_workers_strategy="conservative",
            dataset_size_mb=dataset_size_mb,
        )

        # Generate config with optimized resource parameters
        generate_config_yaml(
            plan,
            dataset_dir,
            output_path,
            fold=fold,
            num_workers=system_resources.num_workers,
            cache_enabled=system_resources.cache_enabled,
            cache_rate=system_resources.cache_rate,
        )

        # Log resource optimization decisions (debug level, goes to file only)
        logger.debug("Resource Optimization Applied:")
        logger.debug(
            f"  num_workers: {system_resources.num_workers} (based on {system_resources.cpu_logical_count} CPU cores)"
        )
        logger.debug(f"  cache_enabled: {system_resources.cache_enabled}")
        if system_resources.cache_enabled:
            logger.debug(
                f"  cache_rate: {system_resources.cache_rate} (dataset {dataset_size_mb:.0f}MB)"
            )

        # Step 4: Generate cross-validation splits
        dataset_json = load_dataset_json(dataset_dir)
        case_identifiers = extract_case_identifiers(
            dataset_json, dataset_path=dataset_dir
        )
        splits = create_splits(
            case_identifiers,
            n_folds=PLANNING_CONSTANTS.N_FOLDS,
            stratified=False,
            seed=PLANNING_CONSTANTS.RANDOM_SEED,
        )
        # Save splits to preprocessed directory instead of raw dataset directory
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        splits_output_path = str(preprocessed_dir / "splits.json")
        save_splits(splits, splits_output_path)
        print()

        # Success summary
        print()
        print("Next steps:")
        dataset_name = dataset_path.name
        config_name = Path(output_path).name
        print(f"  1. Review the generated config: {output_path}")
        print(
            f"  2. Train the model: nnBench.train --config {config_name} --dataset {dataset_name}"
        )
        print()
        print(f"Detailed logs saved to: {log_file}")

    except Exception as e:
        print(f"\nError: Failed to generate configuration: {e}", file=sys.stderr)
        logger.exception(e)
        raise
