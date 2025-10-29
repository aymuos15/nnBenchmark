"""
Standalone training execution function.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.config.validation import (
    validate_deep_supervision_config,
    validate_metrics_config,
    validate_required_field,
)
from src.lightning import (
    GPUMemoryCallback,
    SegmentationDataModule,
    SegmentationModule,
    TrainingHistoryCallback,
    TrainingStepLogger,
    ValidationVisualizationCallback,
)
from src.logging import log_and_print, log_header, log_system_info, setup_train_logger
from src.utils.builders import build_metrics
from src.utils.runner import setup_experiment
from src.utils.seeding import (
    enable_cuda_determinism,
    get_seed_from_config,
    set_random_seeds,
)

# Suppress MONAI deprecation warnings for get_mask_edges (used internally by SurfaceDiceMetric)
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")

# Suppress PyTorch Lightning checkpoint directory not empty warning
warnings.filterwarnings(
    "ignore",
    message=".*Checkpoint directory.*exists and is not empty.*",
    category=UserWarning,
)


def run_training(config_path: str, resume: bool = False) -> None:
    """
    Run training using PyTorch Lightning.

    Args:
        config_path: Path to YAML config file
        resume: Whether to resume from last checkpoint
    """
    # Setup experiment (load config, setup device, paths)
    cfg, device, data_dir, results_dir, config_name = setup_experiment(config_path)

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

    # Check if GPU memory logging is enabled (default: True)
    log_gpu_mem: bool = cfg.get("log_gpu_memory", True)

    # Check if mixed precision training is enabled (default: False)
    use_amp: bool = cfg.get("training", {}).get("mixed_precision", False)

    # Build metrics for config validation (still needed for loss computation during training)
    metric_fns = build_metrics(cfg)

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
    log.info(f"GPU memory logging: {log_gpu_mem}")
    log.info(f"Mixed precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # Log deep supervision configuration
    model_cfg = cfg.get("model", {})
    if model_cfg.get("deep_supervision", False):
        ds_weights = model_cfg.get("ds_weights", [])
        log.info("Deep supervision: Enabled")
        log.info(f"Deep supervision weights: {ds_weights}")

    # Configure TF32 precision (new PyTorch API)
    # Use 'high' for balanced precision/performance on Tensor Core GPUs
    # Note: PyTorch 2.9.0 has a known issue where it internally triggers a deprecation
    # warning even when using the correct new API. This is a PyTorch internal bug,
    # not an issue with our code.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Please use the new API settings to control TF32 behavior.*",
        )
        torch.set_float32_matmul_precision("high")

    # Create Lightning module
    log.info("Creating Lightning module...")
    lit_module = SegmentationModule(cfg=cfg, device=device)

    # Create data module
    log.info("Creating data module...")
    datamodule = SegmentationDataModule(cfg, data_dir, fold)

    # Clean up existing checkpoints if not resuming (prevents "directory not empty" warning)
    results_path = Path(results_dir)
    if not resume and results_path.exists():
        checkpoint_files = [
            f.name
            for f in results_path.iterdir()
            if f.name.endswith(".ckpt") or f.name.endswith(".pt")
        ]
        if checkpoint_files:
            log.info(
                f"Removing {len(checkpoint_files)} existing checkpoint file(s) "
                f"(not resuming, starting fresh)"
            )
            for ckpt_file in checkpoint_files:
                ckpt_path = results_path / ckpt_file
                ckpt_path.unlink()

    # Setup callbacks
    callbacks = []

    if training_all_data:
        # For training on all data, use epoch-based checkpointing (no validation metric)
        callbacks.append(
            ModelCheckpoint(
                dirpath=results_dir,
                filename="best_model",
                save_top_k=1,
                save_last=True,  # Creates last.ckpt
                verbose=False,
                enable_version_counter=False,  # Overwrite checkpoints instead of versioning
                every_n_epochs=1,  # Save every epoch
            )
        )
    else:
        # For cross-validation, use metric-based checkpointing
        callbacks.append(
            ModelCheckpoint(
                dirpath=results_dir,
                filename="best_model",
                monitor=f"val_{checkpoint_metric}",
                mode="max",
                save_top_k=1,
                save_last=True,  # Creates last.ckpt
                verbose=False,
                enable_version_counter=False,  # Overwrite checkpoints instead of versioning
            )
        )

    callbacks.extend(
        [
            TrainingHistoryCallback(results_dir, training_all_data=training_all_data),
            ValidationVisualizationCallback(
                results_dir, cfg, skip_if_no_validation=training_all_data
            ),
        ]
    )

    # Add step logging callback (always enabled)
    callbacks.append(TrainingStepLogger(log, log_gpu_mem=log_gpu_mem))

    # Add GPU memory callback if enabled
    if log_gpu_mem:
        callbacks.append(GPUMemoryCallback(log))

    # Create Trainer
    log.info("Creating Lightning Trainer...")
    log.info("Using GPU device: 0")

    # Configure validation settings based on whether we have validation data
    if training_all_data:
        # Skip validation when training on all data
        val_interval = 1  # Irrelevant, but set a value
        sanity_val_steps = 0  # Skip sanity validation
    else:
        val_interval = cfg["training"]["val_interval"]
        sanity_val_steps = 2

    trainer = Trainer(
        max_epochs=cfg["training"]["epochs"],
        check_val_every_n_epoch=val_interval,
        num_sanity_val_steps=sanity_val_steps,
        precision="16-mixed" if use_amp else "32-true",
        accelerator="auto",
        devices=[0],  # Use GPU 0 by default
        strategy="auto",  # Auto-select strategy (single GPU mode)
        callbacks=callbacks,
        logger=False,  # Disable Lightning's default loggers, use loguru
        enable_progress_bar=True,
        enable_model_summary=False,
        sync_batchnorm=True,  # Important for DDP
        deterministic=False,  # Disable torch.use_deterministic_algorithms() as it conflicts with MONAI cross-entropy loss
    )

    # Determine checkpoint path for resumption
    ckpt_path = None
    if resume:
        ckpt_path = str(Path(results_dir) / "last.ckpt")
        if not Path(ckpt_path).exists():
            log_and_print(
                log,
                f"ERROR: Resume requested but checkpoint not found: {ckpt_path}",
                level="ERROR",
            )
            sys.exit(1)
        log_and_print(log, f"Resuming training from: {ckpt_path}")

    # Train
    log_header(log, "STARTING TRAINING")
    log_and_print(log, f"Training for {cfg['training']['epochs']} epochs...")

    trainer.fit(lit_module, datamodule=datamodule, ckpt_path=ckpt_path)

    # Log completion
    log_header(log, "TRAINING COMPLETED")

    log.info(f"Total epochs trained: {cfg['training']['epochs']}")

    # Log output locations
    history_path = str(Path(results_dir) / "training_history.json")
    log.info(f"Training history saved to: {history_path}")
    log.info(f"Best model checkpoint: {Path(results_dir) / 'best_model.ckpt'}")
    log.info(f"Last checkpoint: {Path(results_dir) / 'last.ckpt'}")

    if training_all_data:
        log.info("Training on all data completed (no validation metrics)")
        log_and_print(log, "\nTraining completed on all data!")
    else:
        # Get best metric value from trainer
        best_metric_val = trainer.callback_metrics.get(f"val_{checkpoint_metric}", -1.0)
        if isinstance(best_metric_val, torch.Tensor):
            best_metric_val = best_metric_val.item()
        log.info(f"Best {checkpoint_metric}: {best_metric_val:.4f}")
        log_and_print(
            log,
            f"\nTraining completed! Best {checkpoint_metric}: {best_metric_val:.4f}",
        )
    log_and_print(log, f"Training history saved to: {history_path}")
