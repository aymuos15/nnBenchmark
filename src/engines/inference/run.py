"""
Inference orchestration module for running complete inference workflows.

Uses Ignite-based EvaluationEngine for event-driven inference.
"""

from pathlib import Path

import torch
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from src.config import resolve_config_path
from src.config.validation import validate_sliding_window_config
from src.engines.inference.engine import EvaluationEngine
from src.engines.inference.handlers import (
    InferenceMetricsHandler,
    InferenceProgressHandler,
    InferenceResultsHandler,
)
from src.engines.setup import (
    build_metrics,
    build_model,
    build_transforms,
    log_metrics_summary,
    print_results,
    setup_experiment,
)
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


def _build_data_dicts_from_folder(input_folder: str, file_ending: str) -> list[dict[str, str]]:
    """Build data dicts from an input image folder, pairing with labels if available."""
    from src.utils.data import extract_base_name_for_label

    input_path = Path(input_folder)
    label_dir = input_path.parent / input_path.name.replace("images", "labels")

    data_dicts: list[dict[str, str]] = []
    for img in sorted(input_path.glob(f"*{file_ending}")):
        if img.name.startswith("._"):
            continue
        entry: dict[str, str] = {"image": str(img)}
        if label_dir.exists():
            base_name, label_ext = extract_base_name_for_label(img.name)
            label_path = label_dir / f"{base_name}{label_ext}"
            if label_path.exists():
                entry["label"] = str(label_path)
        data_dicts.append(entry)
    return data_dicts


def run_inference(
    config_path: str,
    model_path: str | None = None,
    use_test_set: bool = False,
    dataset: str | None = None,
    input_folder: str | None = None,
    output_folder: str | None = None,
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

    seed = get_seed_from_config(cfg)
    if seed is not None:
        set_random_seeds(seed)
    enable_cuda_determinism(deterministic=False)
    log.info(f"Random seed: {seed}")

    # Log system information
    log_system_info(log, device)

    if model_path is None:
        # Try to find the best model checkpoint in checkpoints/ subdirectory
        checkpoints_dir = Path(results_dir) / "checkpoints"

        # Look for best model checkpoint
        checkpoints = list(checkpoints_dir.glob("best_loss*.pt"))

        if checkpoints:
            # Sort by modification time, use most recent
            model_path = max(checkpoints, key=lambda p: Path(p).stat().st_mtime)
        else:
            # Fall back to final checkpoint
            model_path = str(checkpoints_dir / "final.pt")

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
    if input_folder is not None:
        import json as _json
        dataset_json_path = Path(data_dir) / "dataset.json"
        with open(dataset_json_path) as _f:
            file_ending = _json.load(_f).get("file_ending", ".nii.gz")
        test_data = _build_data_dicts_from_folder(input_folder, file_ending)
        log.info(f"Test cases: {len(test_data)} (from -i {input_folder})")
    else:
        test_data = get_test_data_dicts(data_dir, fold, use_test_set)
        log.info(f"Test cases: {len(test_data)}")
        log.info(
            f"Test mode: {'Dedicated test set' if use_test_set else 'Validation split'}"
        )

    # Transforms from config
    test_transforms = build_transforms(cfg, mode="test")

    # When loading raw data via -i, add EnsureChannelFirstd after LoadImaged
    if input_folder is not None:
        from monai.transforms import Compose, EnsureChannelFirstd

        transform_list = list(test_transforms.transforms)
        # Insert EnsureChannelFirstd after LoadImaged (index 1)
        for i, t in enumerate(transform_list):
            if type(t).__name__ == "LoadImaged":
                transform_list.insert(i + 1, EnsureChannelFirstd(keys=["image", "label"]))
                break
        test_transforms = Compose(transform_list)

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
        model = build_model(cfg, device)

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
    # Use inference_metrics if specified, otherwise fall back to metrics
    metrics_cfg = cfg.copy()
    if "inference_metrics" in cfg:
        metrics_cfg["metrics"] = cfg["inference_metrics"]
        log.info("Using inference-specific metrics")
    else:
        log.info("Using default metrics (inference_metrics not specified)")

    metric_fns = build_metrics(metrics_cfg)
    metric_names = list(metric_fns.keys())
    log.info(f"Metrics: {', '.join(metric_names)}")

    # Get include_background from first metric config
    include_background = False
    if "metrics" in metrics_cfg and len(metrics_cfg["metrics"]) > 0:
        include_background = metrics_cfg["metrics"][0].get("include_background", False)

    # Create EvaluationEngine
    log_header(log, "Running inference on test set...")
    inference_engine = EvaluationEngine(
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
        print_results(results, metric_name, context="TEST")

    # Log summary statistics for all metrics (log file only, not console)
    log_metrics_summary(log, all_results, context="TEST RESULTS")

    # Log completion
    log_separator(log, print_too=False)
    log.info(f"Results saved to: {results_dir}")
    log.info(f"Test history file: {Path(results_dir) / 'history' / 'test.json'}")
    log.info("Inference completed successfully!")
    log_separator(log, print_too=False)
