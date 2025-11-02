"""
Standalone training execution function.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset, Dataset

from src.config import resolve_config_path
from src.config.validation import (
    validate_deep_supervision_config,
    validate_metrics_config,
    validate_required_field,
)
from src.engines.common import setup_experiment
from src.engines.ignite_utils import create_trainer
from src.factory import metric_registry, transform_registry
from src.logging import log_and_print, log_header, log_system_info, setup_train_logger
from src.utils.data import get_data_dicts
from src.utils.seeding import (
    enable_cuda_determinism,
    get_seed_from_config,
    set_random_seeds,
)

# Suppress MONAI deprecation warnings for get_mask_edges (used internally by SurfaceDiceMetric)
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


def run_training(
    config_path: str, dataset: str | None = None, resume: bool = False
) -> None:
    """
    Run training using MONAI SupervisedTrainer.

    Args:
        config_path: Path to YAML config file or relative path (e.g., fold_0.yaml)
        dataset: Dataset name (required if config_path is relative)
        resume: Whether to resume from last checkpoint
    """
    # Resolve config path (handles both absolute and relative paths)
    resolved_config_path = str(resolve_config_path(config_path, dataset))

    # Setup experiment (load config, setup device, paths)
    cfg, device, data_dir, results_dir, config_name = setup_experiment(
        resolved_config_path
    )

    seed: int = get_seed_from_config(cfg)
    set_random_seeds(seed)
    enable_cuda_determinism(deterministic=False)

    # Setup logger for training (append to log if resuming)
    log = setup_train_logger(results_dir, resume=resume)
    log_header(log, f"Training started for config: {config_name}")
    log.info(f"Random seed: {seed}")

    # Log system information
    log_system_info(log, device)

    # Get fold number (required)
    validate_required_field(cfg, ["dataset", "fold"], "fold", "fold: 0")
    fold: int = cfg["dataset"]["fold"]

    # Check if we're training on all data (fold=-1, no validation)
    training_all_data: bool = fold == -1

    # Check if mixed precision training is enabled (default: False)
    use_amp: bool = cfg.get("training", {}).get("mixed_precision", False)

    # Build metrics for config validation (still needed for loss computation during training)
    metric_fns = metric_registry.build(cfg)

    if not training_all_data:
        checkpoint_metric, plot_metrics = validate_metrics_config(cfg, metric_fns)
    else:
        checkpoint_metric = None
        plot_metrics = []

    # Validate deep supervision configuration
    validate_deep_supervision_config(cfg)

    # Log training configuration
    log.info(f"Dataset: {cfg['dataset']['name']}")
    log.info(f"Fold: {fold}")
    log.info(f"Epochs: {cfg['training']['epochs']}")
    log.info(f"Batch size: {cfg['training']['batch_size']}")
    if not training_all_data:
        log.info(f"Validation interval: {cfg['training']['val_interval']}")
        log.info(f"Checkpoint metric: {checkpoint_metric}")
        log.info(f"Plot metrics: {plot_metrics}")
    else:
        log.info("Training on all data (no validation split)")
    log.info(f"Mixed precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # Log deep supervision configuration
    model_cfg = cfg.get("model", {})
    if model_cfg.get("deep_supervision", False):
        ds_weights = model_cfg.get("ds_weights", [])
        log.info("Deep supervision: Enabled")
        log.info(f"Deep supervision weights: {ds_weights}")

    # Configure TF32 precision (new PyTorch API)
    # Use 'high' for balanced precision/performance on Tensor Core GPUs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Please use the new API settings to control TF32 behavior.*",
        )
        torch.set_float32_matmul_precision("high")

    # Create data loaders
    log.info("Creating data loaders...")

    # Get data dicts
    train_data, val_data = get_data_dicts(data_dir, fold)

    # Build transforms
    train_transforms = transform_registry.build(cfg, mode="train")
    val_transforms = transform_registry.build(cfg, mode="val")

    # Check if caching is enabled
    cache_config = cfg.get("dataset", {}).get("cache", {})
    use_cache = cache_config.get("enabled", False)
    cache_rate = cache_config.get("cache_rate", 1.0)
    num_workers = cfg["training"]["num_workers"]

    if use_cache:
        # Use CacheDataset for faster training
        log_and_print(log, f"Caching {int(cache_rate * 100)}% of training data...")
        train_ds = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        if val_data:
            log_and_print(
                log, f"Caching {int(cache_rate * 100)}% of validation data..."
            )
            val_ds = CacheDataset(
                data=val_data,
                transform=val_transforms,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            val_ds = None
        persistent_workers = False  # CacheDataset requires persistent_workers=False
        log_and_print(log, "Data caching completed!")
    else:
        # Use basic Dataset (no caching)
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms) if val_data else None
        persistent_workers = num_workers > 0

    # Create data loaders
    # pin_memory=False to reduce GPU memory pressure and avoid CUDA transfer issues
    # This is particularly important for small GPUs (e.g., 4GB RTX A1000)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=False,
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=1,  # Validation uses batch_size=1 for full volumes
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )
        if val_ds is not None
        else None
    )

    log.info(f"Training samples: {len(train_ds)}")
    if val_loader is not None:
        log.info(f"Validation samples: {len(val_ds)}")  # type: ignore[arg-type]

    # Clean up existing checkpoints if not resuming
    results_path = Path(results_dir)
    if not resume and results_path.exists():
        checkpoint_files = [
            f.name
            for f in results_path.iterdir()
            if f.name.endswith(".pt") or f.name.endswith(".pth")
        ]
        if checkpoint_files:
            log.info(
                f"Removing {len(checkpoint_files)} existing checkpoint file(s) "
                f"(not resuming, starting fresh)"
            )
            for ckpt_file in checkpoint_files:
                ckpt_path = results_path / ckpt_file
                ckpt_path.unlink()

    # Create MONAI trainer and evaluator
    log.info("Creating MONAI SupervisedTrainer...")
    log.info("Using GPU device: 0")

    trainer, evaluator = create_trainer(
        cfg=cfg,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        results_dir=results_dir,
        logger=log,
        resume=resume,
    )

    # Handle checkpoint resumption
    if resume:
        checkpoint_path = str(Path(results_dir) / "checkpoint_final_checkpoint.pt")
        if not Path(checkpoint_path).exists():
            # Try alternative checkpoint name
            checkpoint_path = str(Path(results_dir) / "best_model_key_metric*.pt")
            import glob

            checkpoints = glob.glob(checkpoint_path)
            if not checkpoints:
                log_and_print(
                    log,
                    f"ERROR: Resume requested but checkpoint not found in: {results_dir}",
                    level="ERROR",
                )
                sys.exit(1)
            checkpoint_path = checkpoints[0]

        log_and_print(log, f"Resuming training from: {checkpoint_path}")
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        trainer.network.load_state_dict(checkpoint["model"])
        log.info("Checkpoint loaded successfully")

    # Train
    log_header(log, "STARTING TRAINING")
    log_and_print(log, f"Training for {cfg['training']['epochs']} epochs...")

    trainer.run()

    # Log completion
    log_header(log, "TRAINING COMPLETED")

    log.info(f"Total epochs trained: {cfg['training']['epochs']}")

    # Log output locations
    history_path = str(Path(results_dir) / "training_history.json")
    log.info(f"Training history saved to: {history_path}")

    if training_all_data:
        log.info("Training on all data completed (no validation metrics)")
        log_and_print(log, "\nTraining completed on all data!")
    else:
        # Get best metric value from evaluator if available
        if evaluator and hasattr(evaluator.state, "metrics"):
            best_metric_val = evaluator.state.metrics.get(
                f"val_{checkpoint_metric}", -1.0
            )
            if isinstance(best_metric_val, torch.Tensor):
                best_metric_val = best_metric_val.item()
            log.info(f"Best {checkpoint_metric}: {best_metric_val:.4f}")
            log_and_print(
                log,
                f"\nTraining completed! Best {checkpoint_metric}: {best_metric_val:.4f}",
            )
        else:
            log_and_print(log, "\nTraining completed!")

    log_and_print(log, f"Training history saved to: {history_path}")
