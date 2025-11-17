"""Validation orchestration module for post-training validation workflows.

Uses Ignite-based ValidationEngine for event-driven validation.
"""

import warnings
from pathlib import Path

import torch
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.config import resolve_config_path
from src.config.validation import validate_sliding_window_config
from src.engines.validate.engine import ValidationEngine
from src.engines.validate.handlers import (
    ValidationMetricsHandler,
    ValidationProgressHandler,
    ValidationResultsHandler,
    ValidationVisualizationHandler,
)
from src.factory import metric_registry, model_registry, transform_registry
from src.logging import (
    log_header,
    log_separator,
    log_system_info,
    setup_val_logger,
)
from src.utils.data import get_data_dicts
from src.utils.seeding import (
    enable_cuda_determinism,
    get_seed_from_config,
    set_random_seeds,
)

# Suppress MONAI deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


def print_validation_results(results: dict, metric_name: str) -> None:
    """Print validation results to console in a formatted way.

    Args:
        results: Results dictionary with 'mean', 'std', 'min', 'max' keys
        metric_name: Name of the metric (e.g., "Dice")
    """
    print("\n" + "=" * 50)
    print(f"{metric_name} VALIDATION RESULTS")
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


def run_validation(
    config_path: str,
    dataset: str | None = None,
    checkpoint_path: Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Run validation on trained model checkpoint(s).

    Args:
        config_path: Path to config YAML file
        dataset: Dataset name for resolving relative paths
        checkpoint_path: Specific checkpoint to validate (optional, validates all if None)
        batch_size: Batch size for validation (overrides config)
        num_workers: Number of data loader workers (overrides config)
    """
    from src.engines.common import setup_experiment

    # Resolve config path (handles both absolute and relative paths)
    resolved_config_path = str(resolve_config_path(config_path, dataset))

    # Setup experiment (load config, setup device, paths)
    cfg, device, data_dir, results_dir, config_name = setup_experiment(
        resolved_config_path, create_results_dir=False
    )

    # Determine which checkpoints to validate
    results_path = Path(results_dir)
    checkpoints_to_validate = []

    if checkpoint_path is not None:
        # Single checkpoint mode
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoints_to_validate = [checkpoint_path]
    else:
        # Multi-checkpoint mode: find all epoch checkpoints
        import glob

        checkpoint_pattern = str(results_path / "checkpoint_epoch_*.pt")
        checkpoint_files = sorted(glob.glob(checkpoint_pattern))

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No epoch checkpoints found in {results_path}. "
                f"Looking for pattern: checkpoint_epoch_*.pt"
            )

        checkpoints_to_validate = [Path(f) for f in checkpoint_files]
        print(f"Found {len(checkpoints_to_validate)} epoch checkpoints to validate")

    # Validate each checkpoint
    for idx, ckpt_path in enumerate(checkpoints_to_validate, 1):
        if len(checkpoints_to_validate) > 1:
            print(f"\n{'=' * 70}")
            print(
                f"Validating checkpoint {idx}/{len(checkpoints_to_validate)}: {ckpt_path.name}"
            )
            print(f"{'=' * 70}\n")

        _validate_single_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=cfg,
            device=device,
            data_dir=data_dir,
            results_dir=results_dir,
            config_name=config_name,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def _validate_single_checkpoint(
    checkpoint_path: Path,
    cfg: dict,
    device: torch.device,
    data_dir: str,
    results_dir: str,
    config_name: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Run validation on a single checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint file
        cfg: Configuration dictionary
        device: Device to use
        data_dir: Dataset directory
        results_dir: Results directory
        config_name: Config file name
        batch_size: Batch size for validation (overrides config)
        num_workers: Number of data loader workers (overrides config)
    """
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract epoch from checkpoint
    epoch = checkpoint.get("epoch", None)

    # Setup logger (creates val.log)
    log = setup_val_logger(results_dir)
    log_header(log, f"Validation started for: {config_name} (Epoch {epoch})")

    # Seeding
    seed: int = get_seed_from_config(cfg)
    set_random_seeds(seed)
    enable_cuda_determinism(deterministic=False)
    log.info(f"Random seed: {seed}")

    # Log system information
    log_system_info(log, device)

    # Get fold from config
    fold = cfg["dataset"].get("fold")

    if fold is None:
        raise ValueError("Fold number not found in config")

    log.info("CONFIG INFO:")
    log.info(f"Dataset: {cfg['dataset']['name']}")
    log.info(f"Fold: {fold}")
    log.info(f"Epoch: {epoch}")
    log.info(f"Checkpoint: {checkpoint_path}")

    # Check mixed precision
    use_amp: bool = cfg.get("training", {}).get("mixed_precision", False)
    if use_amp and device.type == "cuda":
        log.info("Mixed precision (FP16) validation enabled")
    elif use_amp:
        use_amp = False
        log.warning(
            "Mixed precision requested but CUDA not available, falling back to FP32"
        )
    else:
        log.info("Mixed precision (FP16) validation disabled")

    # Get validation data
    _, val_data = get_data_dicts(data_dir, fold)
    log.info(f"Validation cases: {len(val_data)}")

    # Transforms
    val_transforms = transform_registry.build(cfg, mode="val")

    # Dataset and loader
    val_batch_size = (
        batch_size
        if batch_size is not None
        else cfg.get("inference", {}).get("batch_size", 1)
    )
    val_num_workers = (
        num_workers if num_workers is not None else cfg["training"]["num_workers"]
    )

    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader: DataLoader = DataLoader(
        val_ds, batch_size=val_batch_size, num_workers=val_num_workers
    )

    # Load model
    log.info(f"Model: {cfg['model']['type']}")
    model = model_registry.build(cfg["model"], device)

    # Load model weights from checkpoint
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("No 'model' key found in checkpoint")

    model.eval()
    log.info("Model loaded successfully")

    # Validate sliding window configuration
    try:
        validate_sliding_window_config(cfg)
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

    # Metrics
    metric_fns = metric_registry.build(cfg)
    metric_names = list(metric_fns.keys())
    log.info(f"Metrics: {', '.join(metric_names)}")

    # Get include_background from first metric config
    include_background = False
    if "metrics" in cfg and len(cfg["metrics"]) > 0:
        include_background = cfg["metrics"][0].get("include_background", False)

    # Create ValidationEngine
    log_header(log, "Running validation...")
    validation_engine = ValidationEngine(
        model=model,
        device=device,
        cfg=cfg,
        metric_fns=metric_fns,
        data_dir=data_dir,
    )

    # Attach handlers
    metrics_handler = ValidationMetricsHandler(
        metric_fns=metric_fns,
        logger=log,
        data_dir=data_dir,
        include_background=include_background,
        verbose=True,
        device=device,
        data_dicts=val_data,
    )
    metrics_handler.attach(validation_engine.engine)

    progress_handler = ValidationProgressHandler(
        logger=log,
        total_samples=len(val_data),
        data_dicts=val_data,
    )
    progress_handler.attach(validation_engine.engine)

    results_handler = ValidationResultsHandler(
        results_dir=str(results_dir),
        config_name=config_name,
        cfg=cfg,
        fold=fold,
        checkpoint_path=str(checkpoint_path),
        epoch=epoch,
        data_dicts=val_data,
    )
    results_handler.attach(validation_engine.engine)

    viz_handler = ValidationVisualizationHandler(
        results_dir=str(results_dir),
        spatial_dims=cfg["model"].get("spatial_dims", 3),
        epoch=epoch if epoch is not None else 0,
        save_first_n_batches=1,
    )
    viz_handler.attach(validation_engine.engine)

    # Run validation
    validation_engine.run(val_loader)

    # Get results from engine state
    all_results = validation_engine.engine.state.metrics

    # Print results for all metrics
    for metric_name, results in all_results.items():
        print_validation_results(results, metric_name)

    # Log summary statistics (log file only, not console)
    log_header(log, "VALIDATION RESULTS SUMMARY", print_too=False)
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
    val_history_file = (
        f"validation_history_epoch_{epoch:03d}.json"
        if epoch
        else "validation_history.json"
    )
    log.info(f"Validation history: {Path(results_dir) / val_history_file}")
    log.info("Validation completed successfully!")
    log_separator(log, print_too=False)
