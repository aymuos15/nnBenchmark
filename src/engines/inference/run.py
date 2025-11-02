"""
Inference orchestration module for running complete inference workflows.

Uses Ignite-based InferenceEngine for event-driven inference.
"""

import warnings
from pathlib import Path

import torch
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.config import resolve_config_path
from src.config.validation import validate_sliding_window_config
from src.engines.common import setup_experiment
from src.engines.inference.engine import InferenceEngine
from src.engines.inference.handlers import (
    InferenceMetricsHandler,
    InferenceProgressHandler,
    InferenceResultsHandler,
)
from src.factory import metric_registry, model_registry, transform_registry
from src.logging import (
    log_and_print,
    log_header,
    log_separator,
    log_system_info,
    setup_test_logger,
)
from src.utils.data import get_test_data_dicts
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


def run_inference(
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

    # Setup logger for inference
    log = setup_test_logger(results_dir)
    log_header(log, f"Inference started for config: {config_name}")

    seed: int = get_seed_from_config(cfg)
    set_random_seeds(seed)
    enable_cuda_determinism(deterministic=False)
    log.info(f"Random seed: {seed}")

    # Log system information
    log_system_info(log, device)

    if model_path is None:
        # Try to find the best model checkpoint (MONAI format)
        import glob

        # Look for best model checkpoint (MONAI naming: best_model_model_key_metric=*.pt)
        best_model_pattern = str(Path(results_dir) / "best_model*key_metric*.pt")
        checkpoints = glob.glob(best_model_pattern)

        if checkpoints:
            # Sort by modification time, use most recent
            model_path = max(checkpoints, key=lambda p: Path(p).stat().st_mtime)
        else:
            # Fall back to final checkpoint (MONAI naming: best_model_model_final_iteration=*.pt)
            final_pattern = str(Path(results_dir) / "best_model*final*.pt")
            checkpoints = glob.glob(final_pattern)
            if checkpoints:
                model_path = max(checkpoints, key=lambda p: Path(p).stat().st_mtime)
            else:
                # Last resort: look for any checkpoint file
                model_path = str(Path(results_dir) / "checkpoint_final_checkpoint.pt")

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
    log.info(f"Test cases: {len(test_data)}")
    log.info(
        f"Test mode: {'Dedicated test set' if use_test_set else 'Validation split'}"
    )

    # Transforms from config
    test_transforms = transform_registry.build(cfg, mode="test")

    # Dataset and loader (batch_size=1 for inference)
    test_batch_size: int = cfg.get("inference", {}).get("batch_size", 1)
    test_ds = Dataset(data=test_data, transform=test_transforms)
    test_loader: DataLoader = DataLoader(
        test_ds, batch_size=test_batch_size, num_workers=cfg["training"]["num_workers"]
    )

    # Load model checkpoint (MONAI format)
    log.info(f"Model: {cfg['model']['type']}")
    if Path(model_path).exists():
        log.info(f"Loading checkpoint from: {model_path}")

        # Build model from config
        model = model_registry.build(cfg["model"], device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Extract model state dict (MONAI CheckpointSaver format)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)

        model.eval()
        log.info(f"Loaded model from: {model_path}")
    else:
        log_and_print(log, f"Model not found: {model_path}", level="ERROR")
        return

    # Validate sliding window configuration (if present)
    try:
        validate_sliding_window_config(cfg)
        # Log sliding window status
        inference_cfg = cfg.get("inference", {})
        sliding_window_cfg = inference_cfg.get("sliding_window", {})
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
    metric_fns = metric_registry.build(cfg)
    metric_names = list(metric_fns.keys())
    log.info(f"Metrics: {', '.join(metric_names)}")

    # Get include_background from first metric config
    include_background = False
    if "metrics" in cfg and len(cfg["metrics"]) > 0:
        include_background = cfg["metrics"][0].get("include_background", False)

    # Create InferenceEngine
    log_header(log, "Running inference on test set...")
    inference_engine = InferenceEngine(
        model=model,
        device=device,
        cfg=cfg,
        metric_fns=metric_fns,
        data_dir=data_dir,
    )

    # Attach handlers
    metrics_handler = InferenceMetricsHandler(
        metric_fns=metric_fns,
        logger=log,
        data_dir=data_dir,
        include_background=include_background,
        verbose=True,
        device=device,
        data_dicts=test_data,
    )
    metrics_handler.attach(inference_engine.engine)

    progress_handler = InferenceProgressHandler(
        logger=log,
        total_samples=len(test_data),
        data_dicts=test_data,
    )
    progress_handler.attach(inference_engine.engine)

    results_handler = InferenceResultsHandler(
        results_dir=results_dir,
        config_name=config_name,
        cfg=cfg,
        fold=fold,
        use_test_set=use_test_set,
        model_path=model_path,
        data_dicts=test_data,
    )
    results_handler.attach(inference_engine.engine)

    # Run inference
    inference_engine.run(test_loader)

    # Get results from engine state (set by InferenceMetricsHandler)
    all_results = inference_engine.engine.state.metrics

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
    log.info(f"\nNumber of cases: {len(all_results[metric_names[0]]['all_scores'])}")

    # Log completion
    log_separator(log, print_too=False)
    log.info(f"Results saved to: {results_dir}")
    log.info(f"Test history file: {Path(results_dir) / 'test_history.json'}")
    log.info("Inference completed successfully!")
    log_separator(log, print_too=False)
