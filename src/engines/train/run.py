"""
Standalone training execution function.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset, Dataset

from src.config import resolve_config_path
from src.config.validation import (
    validate_deep_supervision_config,
    validate_required_field,
)
from src.engines.ignite_utils import create_trainer
from src.engines.setup import build_transforms, setup_experiment
from src.logging import log_and_print, log_header, log_system_info, setup_train_logger
from src.utils.data import get_data_dicts
from src.utils.seeding import (
    enable_cuda_determinism,
    get_seed_from_config,
    set_random_seeds,
)


def find_latest_checkpoint(results_dir: str) -> str | None:
    """
    Find the most recent checkpoint in results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Path to latest checkpoint file, or None if no checkpoint found
    """
    # Look in checkpoints subdirectory
    checkpoints_path = Path(results_dir) / "checkpoints"
    if not checkpoints_path.exists():
        return None

    # Look for checkpoint files (in priority order)
    checkpoint_patterns = [
        "final.pt",
        "best_loss*.pt",
    ]

    for pattern in checkpoint_patterns:
        if "*" in pattern:
            checkpoints = list(checkpoints_path.glob(pattern))
        else:
            checkpoint_file = checkpoints_path / pattern
            checkpoints = [checkpoint_file] if checkpoint_file.exists() else []

        if checkpoints:
            # Return most recent by modification time
            return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    return None


def validate_checkpoint_config(checkpoint: dict, cfg: dict) -> list[str]:
    """
    Validate that checkpoint config matches current training config.

    Args:
        checkpoint: Loaded checkpoint dictionary
        cfg: Current training configuration

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings_list = []

    # Check if checkpoint has config metadata
    if "config_metadata" not in checkpoint:
        warnings_list.append(
            "Checkpoint missing config metadata (old checkpoint format)"
        )
        return warnings_list

    metadata = checkpoint["config_metadata"]

    # Validate dataset name
    current_dataset = cfg.get("dataset", {}).get("name", "unknown")
    checkpoint_dataset = metadata.get("dataset_name", "unknown")
    if current_dataset != checkpoint_dataset:
        warnings_list.append(
            f"Dataset mismatch: checkpoint={checkpoint_dataset}, current={current_dataset}"
        )

    # Validate fold number
    current_fold = cfg.get("dataset", {}).get("fold", -999)
    checkpoint_fold = metadata.get("fold", -999)
    if current_fold != checkpoint_fold:
        warnings_list.append(
            f"Fold mismatch: checkpoint={checkpoint_fold}, current={current_fold}"
        )

    # Validate model type
    current_model = cfg.get("model", {}).get("type", "unknown")
    checkpoint_model = metadata.get("model_type", "unknown")
    if current_model != checkpoint_model:
        warnings_list.append(
            f"Model type mismatch: checkpoint={checkpoint_model}, current={current_model}"
        )

    return warnings_list


def is_training_complete(checkpoint: dict, cfg: dict) -> bool:
    """
    Check if training is already complete based on checkpoint epoch.

    Args:
        checkpoint: Loaded checkpoint dictionary
        cfg: Current training configuration

    Returns:
        True if training is complete, False otherwise
    """
    # Check if checkpoint has epoch information
    if "epoch" not in checkpoint:
        return False

    checkpoint_epoch = checkpoint["epoch"]
    total_epochs = cfg.get("training", {}).get("epochs", 0)

    # Training is complete if we've reached or exceeded the configured epochs
    return checkpoint_epoch >= total_epochs


def run_training(
    config_path: str,
    dataset: str | None = None,
    force_fresh: bool = False,
) -> None:
    """
    Run training using MONAI SupervisedTrainer.
    Logs always append to existing files. Checkpoints are automatically resumed.

    Args:
        config_path: Path to YAML config file or relative path (e.g., fold_0.yaml)
        dataset: Dataset name (required if config_path is relative)
        force_fresh: Force fresh start, delete existing checkpoints
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

    # Setup logger for training (always appends to existing logs)
    log = setup_train_logger(results_dir)
    log_header(log, f"Training started for config: {config_name}")
    log.info(f"Random seed: {seed}")

    # Log system information
    log_system_info(log, device)

    # Get fold number (required)
    validate_required_field(cfg, ["dataset", "fold"], "fold", "fold: 0")
    fold: int = cfg["dataset"]["fold"]

    # Check if we're training on all data (fold=-1)
    # When fold=-1, we use the training set as validation set
    training_all_data: bool = fold == -1

    # Check if mixed precision training is enabled (default: False)
    use_amp: bool = cfg.get("training", {}).get("mixed_precision", False)

    # Validate deep supervision configuration
    validate_deep_supervision_config(cfg)

    # Log training configuration
    log.info(f"Dataset: {cfg['dataset']['name']}")
    log.info(f"Fold: {fold}")
    log.info(f"Epochs: {cfg['training']['epochs']}")
    log.info(f"Batch size: {cfg['training']['batch_size']}")
    if training_all_data:
        log.info("Training on all data (fold=-1): training without validation split")
    log.info("Checkpoint: Best model based on training loss")
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

    # If training on all data (fold=-1), use training set as validation set
    if training_all_data:
        val_data = train_data

    # Build transforms
    train_transforms = build_transforms(cfg, mode="train")
    val_transforms = build_transforms(cfg, mode="val")

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
            _ = CacheDataset(
                data=val_data,
                transform=val_transforms,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )  # Cache validation data for consistency (not used in training)
        persistent_workers = False  # CacheDataset requires persistent_workers=False
        log_and_print(log, "Data caching completed!")
    else:
        # Use basic Dataset (no caching)
        train_ds = Dataset(data=train_data, transform=train_transforms)
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

    log.info(f"Training samples: {len(train_ds)}")

    # Handle force_fresh: delete all checkpoints if requested
    results_path = Path(results_dir)
    if force_fresh and results_path.exists():
        checkpoint_files = [
            f.name
            for f in results_path.iterdir()
            if f.name.endswith(".pt") or f.name.endswith(".pth")
        ]
        if checkpoint_files:
            log_and_print(
                log,
                f"Force fresh start: Removing {len(checkpoint_files)} existing checkpoint(s)",
            )
            for ckpt_file in checkpoint_files:
                ckpt_path = results_path / ckpt_file
                ckpt_path.unlink()

    # Create MONAI trainer
    log.info("Creating MONAI SupervisedTrainer...")
    log.info("Using GPU device: 0")

    trainer, optimizer, lr_scheduler, scaler = create_trainer(
        cfg=cfg,
        device=device,
        train_loader=train_loader,
        results_dir=results_dir,
        logger=log,
    )

    # Automatic checkpoint detection and loading (unless force_fresh)
    checkpoint_path = find_latest_checkpoint(results_dir) if not force_fresh else None

    if checkpoint_path:
        log_and_print(log, f"Checkpoint detected: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if training is already complete
        if is_training_complete(checkpoint, cfg):
            checkpoint_epoch = checkpoint.get("epoch", "?")
            total_epochs = cfg.get("training", {}).get("epochs", 0)
            log_and_print(
                log,
                f"\nTraining already complete! Checkpoint at epoch {checkpoint_epoch}/{total_epochs}",
            )
            log_and_print(log, "No further training needed. Exiting.")
            sys.exit(0)

        # Validate checkpoint config
        config_warnings = validate_checkpoint_config(checkpoint, cfg)
        if config_warnings:
            log_and_print(
                log,
                "⚠️  WARNING: Checkpoint config validation issues detected:",
                level="WARNING",
            )
            for warning in config_warnings:
                log_and_print(log, f"  - {warning}", level="WARNING")
            log_and_print(
                log, "Continuing with checkpoint load (proceed with caution)..."
            )

        # Load all training state
        log_and_print(log, "Loading training state from checkpoint...")

        # Load model weights
        trainer.network.load_state_dict(checkpoint["model"])
        log.info("✓ Model weights loaded")

        # Load optimizer state
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info("✓ Optimizer state loaded")
        else:
            log_and_print(
                log,
                "⚠️  WARNING: Optimizer state not found in checkpoint (old format)",
                level="WARNING",
            )

        # Load learning rate scheduler state
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            log.info("✓ LR scheduler state loaded")
        else:
            log_and_print(
                log,
                "⚠️  WARNING: LR scheduler state not found in checkpoint (old format)",
                level="WARNING",
            )

        # Load GradScaler state (for AMP)
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
            log.info("✓ GradScaler state loaded")
        else:
            log_and_print(
                log,
                "⚠️  WARNING: GradScaler state not found in checkpoint (old format)",
                level="WARNING",
            )

        # Restore epoch number
        if "epoch" in checkpoint:
            starting_epoch = checkpoint["epoch"]
            # Set trainer state to resume from next epoch
            trainer.state.epoch = starting_epoch
            trainer.state.iteration = starting_epoch * len(train_loader)
            log_and_print(
                log,
                f"✓ Resuming from epoch {starting_epoch}/{cfg['training']['epochs']}",
            )
        else:
            log_and_print(
                log,
                "⚠️  WARNING: Epoch number not found in checkpoint (old format)",
                level="WARNING",
            )

        log_and_print(log, "Checkpoint loaded successfully - resuming training")
    else:
        # No checkpoint found, starting fresh
        log.info("No checkpoint found - starting fresh training")

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

    # Report training completion
    log_and_print(log, "\nTraining completed!")
    log_and_print(log, f"Training history saved to: {history_path}")

    # Report best checkpoint if loss-based tracking was used
    import glob

    best_checkpoint_pattern = str(Path(results_dir) / "checkpoints" / "best_loss*.pt")
    best_checkpoints = glob.glob(best_checkpoint_pattern)
    if best_checkpoints:
        best_checkpoint = best_checkpoints[0]
        # Extract loss value from filename
        match = re.search(r"loss=([\d.]+)\.pt", best_checkpoint)
        if match:
            best_loss = float(match.group(1))
            log_and_print(log, f"Best training loss: {best_loss:.4f}")
            log_and_print(log, f"Best checkpoint saved to: {best_checkpoint}")
