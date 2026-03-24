"""Validation orchestration module for post-training validation workflows.

Uses Ignite-based ValidationEngine for event-driven validation.
"""

from pathlib import Path

import torch
from monai.bundle import ConfigParser
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.engines.inference.engine import EvaluationEngine
from src.engines.setup import (
    build_metrics,
    build_model,
    build_transforms,
    log_metrics_summary,
    print_results,
    setup_experiment,
)
from src.engines.validate.handlers import (
    ValidationMetricsHandler,
    ValidationProgressHandler,
    ValidationResultsHandler,
    ValidationVisualizationHandler,
)
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


def run_validation(
    config_path: str,
    checkpoint_path: Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Run validation on trained model checkpoint(s).

    Args:
        config_path: Resolved path to config YAML file
        checkpoint_path: Specific checkpoint to validate (optional, validates all if None)
        batch_size: Batch size for validation (overrides config)
        num_workers: Number of data loader workers (overrides config)
    """
    cfg, device, data_dir, results_dir, config_name = setup_experiment(
        config_path, create_results_dir=False
    )

    # Determine which checkpoints to validate
    results_path = Path(results_dir)
    checkpoints_to_validate = []

    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoints_to_validate = [checkpoint_path]
    else:
        import glob

        checkpoints_dir = results_path / "checkpoints"
        checkpoint_pattern = str(checkpoints_dir / "epoch_*.pt")
        checkpoint_files = sorted(glob.glob(checkpoint_pattern))

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No epoch checkpoints found in {checkpoints_dir}. "
                f"Looking for pattern: epoch_*.pt"
            )

        checkpoints_to_validate = [Path(f) for f in checkpoint_files]
        print(f"Found {len(checkpoints_to_validate)} epoch checkpoints to validate")

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
    cfg: ConfigParser,
    device: torch.device,
    data_dir: str,
    results_dir: str,
    config_name: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> None:
    """Run validation on a single checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    epoch = checkpoint.get("epoch", None)

    log = setup_val_logger(results_dir)
    log_header(log, f"Validation started for: {config_name} (Epoch {epoch})")

    seed: int = get_seed_from_config(cfg)
    set_random_seeds(seed)
    enable_cuda_determinism(deterministic=False)
    log.info(f"Random seed: {seed}")

    log_system_info(log, device)

    fold = cfg["dataset"].get("fold")

    if fold is None:
        raise ValueError("Fold number not found in config")

    log.info("CONFIG INFO:")
    log.info(f"Dataset: {cfg['dataset']['name']}")
    log.info(f"Fold: {fold}")
    log.info(f"Epoch: {epoch}")
    log.info(f"Checkpoint: {checkpoint_path}")

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

    _, val_data = get_data_dicts(data_dir, fold)
    log.info(f"Validation cases: {len(val_data)}")

    val_transforms = build_transforms(cfg, mode="val")

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

    log.info(f"Model: {cfg['model']['_target_']}")
    model = build_model(cfg, device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("No 'model' key found in checkpoint")

    model.eval()
    log.info("Model loaded successfully")

    # Log sliding window status
    inference_cfg = cfg.get("inference", {})
    sliding_window_cfg = inference_cfg.get("sliding_window", {}) if inference_cfg else {}
    if sliding_window_cfg and sliding_window_cfg.get("enabled", False):
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

    # Metrics
    if "validation_metrics" in cfg:
        metric_fns = build_metrics(cfg, section="validation_metrics")
        log.info("Using validation-specific metrics")
    else:
        metric_fns = build_metrics(cfg, section="metrics")
        log.info("Using default metrics (validation_metrics not specified)")

    metric_names = list(metric_fns.keys())
    log.info(f"Metrics: {', '.join(metric_names)}")

    # Get include_background from first metric config
    include_background = False
    metrics_list = cfg.get("validation_metrics") or cfg.get("metrics", [])
    if metrics_list and len(metrics_list) > 0:
        include_background = metrics_list[0].get("include_background", False)

    # Create EvaluationEngine
    log_header(log, "Running validation...")
    validation_engine = EvaluationEngine(
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

    validation_engine.run(val_loader)

    all_results = validation_engine.engine.state.metrics

    for metric_name, results in all_results.items():
        print_results(results, metric_name, context="VALIDATION")

    log_metrics_summary(log, all_results, context="VALIDATION RESULTS")

    log_separator(log, print_too=False)
    log.info(f"Results saved to: {results_dir}")
    val_history_file = (
        f"validation_epoch_{epoch:03d}.json" if epoch is not None else "validation.json"
    )
    log.info(f"Validation history: {Path(results_dir) / 'history' / val_history_file}")
    log.info("Validation completed successfully!")
    log_separator(log, print_too=False)
