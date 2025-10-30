"""
Testing orchestration module for running complete inference workflows.
"""

import warnings
from pathlib import Path

import numpy as np
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.config import resolve_config_path
from src.config.validation import validate_sliding_window_config
from src.inference.evaluate import evaluate
from src.lightning import SegmentationModule
from src.logging import (
    log_and_print,
    log_header,
    log_separator,
    log_system_info,
    setup_test_logger,
)
from src.utils.builders import build_metrics, build_transforms
from src.utils.data import get_test_data_dicts
from src.utils.files import ensure_directory, save_json
from src.utils.runner import setup_experiment
from src.utils.seeding import (
    enable_cuda_determinism,
    get_seed_from_config,
    set_random_seeds,
)

# Suppress MONAI deprecation warnings for get_mask_edges (used internally by SurfaceDiceMetric)
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


def print_test_results(results: dict, metric_name: str) -> None:
    """
    Print test results to console in a formatted way.

    Args:
        results: Results dictionary with 'mean', 'std', 'min', 'max' keys and optional 'per_class'
        metric_name: Name of the metric (e.g., "Dice")
    """
    print("\n" + "=" * 50)
    print(f"{metric_name} TEST RESULTS")
    print("=" * 50)
    print(f"Mean {metric_name} Score: {results['mean']:.4f} ± {results['std']:.4f}")

    # Print per-class results if available
    if "per_class" in results:
        per_class = results["per_class"]
        if isinstance(per_class, dict):
            for class_name, class_stats in per_class.items():
                print(
                    f"{class_name}: {class_stats['mean']:.4f} ± {class_stats['std']:.4f}"
                )

    print(f"\nMin {metric_name} Score: {results['min']:.4f}")
    print(f"Max {metric_name} Score: {results['max']:.4f}")
    print("=" * 50)


def run_testing(
    config_path: str,
    model_path: str | None = None,
    use_test_set: bool = False,
    dataset: str | None = None,
) -> None:
    # Resolve config path (handles both absolute and relative paths)
    resolved_config_path = str(resolve_config_path(config_path, dataset))

    # Setup experiment (load config, setup device, paths)
    cfg, device, data_dir, results_dir, config_name = setup_experiment(
        resolved_config_path, create_results_dir=False
    )

    # Setup logger for testing
    log = setup_test_logger(results_dir)
    log_header(log, f"Testing started for config: {config_name}")

    seed: int = get_seed_from_config(cfg)
    set_random_seeds(seed)
    enable_cuda_determinism(deterministic=False)
    log.info(f"Random seed: {seed}")

    # Log system information
    log_system_info(log, device)

    if model_path is None:
        model_path = str(Path(results_dir) / "best_model.ckpt")

    # Get fold number (required unless using dedicated test set)
    fold: int | None
    if not use_test_set:
        if "fold" not in cfg["dataset"]:
            raise ValueError(
                "'fold' parameter is required in dataset config. Please specify which fold to use (e.g., fold: 0)"
            )
        fold = cfg["dataset"]["fold"]
        log.info("CONFIG INFO:")
        log.info(f"Fold: {fold}")
    else:
        fold = None  # Not needed for dedicated test set
        log.info("Using dedicated test set")

    # Check if GPU memory logging is enabled (default: True)
    log_gpu_mem: bool = cfg.get("log_gpu_memory", True)
    if log_gpu_mem:
        log.info("GPU memory logging is enabled")

    # Check if mixed precision is enabled (default: False)
    use_amp: bool = cfg.get("training", {}).get("mixed_precision", False)
    if use_amp and device.type == "cuda":
        log.info("Mixed precision (FP16) inference enabled")
    elif use_amp:
        use_amp = False
        log.warning(
            "Mixed precision requested but CUDA not available, falling back to FP32"
        )
    else:
        log.info("Mixed precision (FP16) inference disabled")

    # Data
    test_data = get_test_data_dicts(data_dir, fold, use_test_set)
    log.info(f"Test samples: {len(test_data)}")
    log.info(
        f"Test mode: {'Dedicated test set' if use_test_set else 'Validation split'}"
    )

    # Transforms from config
    test_transforms = build_transforms(cfg, mode="test")

    # Dataset and loader
    test_batch_size: int = cfg.get("testing", {}).get("batch_size", 1)
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader: DataLoader = DataLoader(
        test_ds, batch_size=test_batch_size, num_workers=cfg["training"]["num_workers"]
    )

    # Load Lightning checkpoint
    log.info(f"Model: {cfg['model']['type']}")
    if Path(model_path).exists():
        log.info(f"Loading checkpoint from: {model_path}")
        lit_module = SegmentationModule.load_from_checkpoint(
            model_path,
            cfg=cfg,
            device=device,
            map_location=device,
        )
        # Extract model for testing
        model = lit_module.model
        model.eval()
        log.info(f"Loaded model from: {model_path}")
    else:
        log_and_print(log, f"Model not found: {model_path}", level="ERROR")
        return

    # Validate sliding window configuration (if present)
    try:
        validate_sliding_window_config(cfg)
        # Log sliding window status
        testing_cfg = cfg.get("testing", {})
        sliding_window_cfg = testing_cfg.get("sliding_window", {})
        if sliding_window_cfg.get("enabled", False):
            roi_size = sliding_window_cfg.get(
                "roi_size", cfg.get("dataset", {}).get("spatial_size")
            )
            overlap = sliding_window_cfg.get("overlap", 0.5)
            mode = sliding_window_cfg.get("mode", "gaussian")
            log.info("Sliding window inference enabled")
            log.info(f"  ROI size: {roi_size}")
            log.info(f"  Overlap: {overlap}")
            log.info(f"  Blending mode: {mode}")
        else:
            log.info("Using full-volume inference")
    except ValueError as e:
        log.error(f"Invalid sliding window configuration: {e}")
        raise

    # Metrics from config
    metric_fns = build_metrics(cfg)
    metric_names = list(metric_fns.keys())
    log.info(f"Metrics: {', '.join(metric_names)}")

    # Run evaluation
    log_header(log, "Running inference on test set...")
    all_results = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        cfg=cfg,
        verbose=True,
        data_dicts=test_data,
        logger=log,
        log_gpu_mem=log_gpu_mem,
        metric_fns=metric_fns,
        data_dir=data_dir,
        use_amp=use_amp,
    )

    # Print results for all metrics
    for metric_name, results in all_results.items():
        print_test_results(results, metric_name)

    # Log summary statistics for all metrics (log file only, not console)
    log_header(log, "TEST RESULTS SUMMARY", print_too=False)
    for metric_name, results in all_results.items():
        log.info(f"\n{metric_name}:")
        log.info(f"  Mean: {results['mean']:.4f} ± {results['std']:.4f}")

        # Log per-class results if available
        if "per_class" in results:
            log.info("  Per-Class Results:")
            for class_name, class_stats in results["per_class"].items():
                log.info(
                    f"    {class_name}: {class_stats['mean']:.4f} ± {class_stats['std']:.4f}"
                )

        log.info(f"  Min: {results['min']:.4f}")
        log.info(f"  Max: {results['max']:.4f}")
    log.info(f"\nNumber of samples: {len(all_results[metric_names[0]]['all_scores'])}")

    # Save results
    ensure_directory(results_dir)

    # Prepare summary statistics for all metrics
    summary = {}
    per_sample_scores = {}

    for metric_name, results in all_results.items():
        metric_summary = {
            "mean": results["mean"],
            "std": results["std"],
            "min": results["min"],
            "max": results["max"],
            "num_samples": len(results["all_scores"]),
        }

        # Add per-class statistics to summary if available
        if "per_class" in results:
            metric_summary["per_class"] = results["per_class"]

        summary[metric_name] = metric_summary

        # Convert per_sample_scores to JSON-serializable format
        # Handle both scalar and per-class (numpy array) cases
        metric_per_sample_scores = results["all_scores"]
        if len(metric_per_sample_scores) > 0 and isinstance(
            metric_per_sample_scores[0], np.ndarray
        ):
            # Per-class scores - convert numpy arrays to lists
            metric_per_sample_scores = [
                score.tolist() for score in metric_per_sample_scores
            ]

        per_sample_scores[metric_name] = metric_per_sample_scores

    # Create test history JSON
    test_history = {
        "config_name": config_name,
        "dataset_name": cfg["dataset"]["name"],
        "fold": fold,
        "use_test_set": use_test_set,
        "model_path": model_path,
        "metrics": metric_names,  # List of all metrics
        "summary": summary,  # Summary per metric
        "per_sample_scores": per_sample_scores,  # Scores per metric
        "sample_names": [Path(d.get("image", "unknown")).name for d in test_data],
    }

    # Save test history JSON
    test_history_path = str(Path(results_dir) / "test_history.json")
    save_json(test_history, test_history_path)

    # Log to file with separators (not to console to avoid clutter)
    log_separator(log, print_too=False)
    log.info(f"Results saved to: {results_dir}")
    log.info(f"Test history file: {test_history_path}")
    log.info("Testing completed successfully!")
    log_separator(log, print_too=False)

    # Print clean summary to console
    log_and_print(log, f"\nResults saved to: {results_dir}")
    log_and_print(log, f"Test history: {test_history_path}")
