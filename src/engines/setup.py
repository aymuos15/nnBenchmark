"""
Common engine utilities for training and inference.
Provides helpers for experiment setup to reduce code duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from monai import metrics as monai_metrics
from monai import transforms
from monai.networks import nets as monai_nets

from src.config import get_datasets_root, get_results_root
from src.config.load import load_config
from src.utils.files import ensure_directory

if TYPE_CHECKING:
    from loguru._logger import Logger


def _safe_getattr(module: Any, name: str, module_name: str) -> type:
    """Safely get an attribute from a module with helpful error messages.

    Args:
        module: The module to get the attribute from
        name: The attribute name to retrieve
        module_name: Human-readable module name for error messages

    Returns:
        The requested attribute (class)

    Raises:
        ValueError: If attribute not found, with list of available options
    """
    try:
        return getattr(module, name)
    except AttributeError as e:
        # Get available public attributes
        available = sorted([n for n in dir(module) if not n.startswith('_')])
        # Show first 20 options
        options_str = ", ".join(available[:20])
        if len(available) > 20:
            options_str += f", ... and {len(available) - 20} more"
        raise ValueError(
            f"'{name}' not found in {module_name}. "
            f"Available options: {options_str}"
        ) from e


def setup_device(verbose: bool = True) -> torch.device:
    """
    Setup and return the appropriate device (CUDA or CPU).

    Args:
        verbose: If True, print device information

    Returns:
        torch.device for model placement

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
    return device


def get_config_name(config_path: str) -> str:
    """
    Extract configuration name from config file path.

    Args:
        config_path: Path to config file (e.g., "configs/dataset001_hippo.yaml")

    Returns:
        Config name without extension (e.g., "dataset001_hippo")

    """
    config_file = Path(config_path)
    return config_file.stem


def setup_results_dir(config_name: str, dataset_name: str, create: bool = True) -> str:
    """
    Setup results directory path from config name and dataset name.

    Args:
        config_name: Name of the config (from get_config_name)
        dataset_name: Name of the dataset
        create: If True, create the directory if it doesn't exist

    Returns:
        Path to results directory

    """
    results_dir = str(get_results_root() / dataset_name / config_name)
    if create:
        return ensure_directory(results_dir)
    return results_dir


def setup_experiment(
    config_path: str, create_results_dir: bool = True
) -> tuple[dict[str, Any], torch.device, str, str, str]:
    """
    One-stop setup for training/testing experiments.

    Performs common setup steps:
    1. Load configuration
    2. Setup device
    3. Extract config name
    4. Setup results directory
    5. Derive data directory path from dataset name

    Args:
        config_path: Path to YAML config file
        create_results_dir: If True, create results directory

    Returns:
        Tuple of (cfg, device, data_dir, results_dir, config_name)
    """
    # Load config
    cfg = load_config(config_path)

    # Setup device
    device = setup_device(verbose=False)

    # Get paths
    dataset_name = cfg["dataset"]["name"]
    data_dir = str(get_datasets_root() / dataset_name)
    config_name = get_config_name(config_path)
    results_dir = setup_results_dir(
        config_name, dataset_name, create=create_results_dir
    )

    return cfg, device, data_dir, results_dir, config_name


def build_transforms(config: dict, mode: str = "train") -> transforms.Compose:
    """Build transform pipeline from config using getattr for MONAI transforms.

    Args:
        config: Configuration dictionary with 'transforms' section
        mode: Transform mode ('train', 'val', or 'test')

    Returns:
        MONAI Compose object containing the transform pipeline
    """
    transform_list = []

    # Build common transforms
    for t_cfg in config["transforms"]["common"]:
        t_cfg = t_cfg.copy()
        t_type = t_cfg.pop("type")
        t_class = _safe_getattr(transforms, t_type, "monai.transforms")
        transform_list.append(t_class(**t_cfg))

    # Append mode-specific transforms
    for t_cfg in config["transforms"][mode]:
        t_cfg = t_cfg.copy()
        t_type = t_cfg.pop("type")
        t_class = _safe_getattr(transforms, t_type, "monai.transforms")
        transform_list.append(t_class(**t_cfg))

    return transforms.Compose(transform_list)


def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Build model from config using getattr for MONAI networks.

    Args:
        config: Configuration dictionary with 'model' section
        device: Device to place the model on

    Returns:
        PyTorch model instance
    """
    model_cfg = config["model"].copy()
    model_type = model_cfg.pop("type")
    # Remove training-only parameters that shouldn't be passed to model constructor
    model_cfg.pop("ds_weights", None)  # Used by DeepSupervisionLossWrapper, not model
    # deep_supervision only supported by DynUNet and BasicUNetPlusPlus
    if model_type not in ("DynUNet", "BasicUNetPlusPlus"):
        model_cfg.pop("deep_supervision", None)
        model_cfg.pop("deep_supr_num", None)
    # Remove other model type configs (e.g., UNet config when using DynUNet)
    model_cfg.pop("DynUNet", None)
    model_cfg.pop("UNet", None)
    # Merge model-specific parameters if present
    if model_type in config["model"] and isinstance(config["model"][model_type], dict):
        model_cfg.update(config["model"][model_type])
    model_class = _safe_getattr(monai_nets, model_type, "monai.networks.nets")
    return model_class(**model_cfg).to(device)


def build_metrics(config: dict) -> dict:
    """Build metrics from config using getattr for MONAI metrics.

    Args:
        config: Configuration dictionary with 'metrics' section

    Returns:
        Dictionary of metric functions {name: metric_fn}
    """
    metric_fns = {}
    for m_cfg in config["metrics"]:
        m_cfg = m_cfg.copy()
        m_type = m_cfg.pop("type")
        m_class = _safe_getattr(monai_metrics, m_type, "monai.metrics")
        metric_fns[m_type] = m_class(**m_cfg)
    return metric_fns


def print_results(results: dict, metric_name: str, context: str = "EVALUATION") -> None:
    """Print evaluation results to console in a formatted way.

    Args:
        results: Results dictionary with 'mean', 'std', 'min', 'max' keys
                and optional 'per_class'
        metric_name: Name of the metric (e.g., "Dice")
        context: Context string for header (e.g., "TEST", "VALIDATION")
    """
    logger.info("\\n" + "=" * 50)
    logger.info(f"{metric_name} {context} RESULTS")
    logger.info("=" * 50)
    logger.info(
        f"Mean {metric_name} Score: {results['mean']:.4f} ± {results['std']:.4f}"
    )

    # Log per-class results if available
    if "per_class" in results:
        per_class = results["per_class"]
        if isinstance(per_class, dict):
            for class_name, class_stats in per_class.items():
                logger.info(
                    f"{class_name}: {class_stats['mean']:.4f} ± {class_stats['std']:.4f}"
                )

    logger.info(f"\\nMin {metric_name} Score: {results['min']:.4f}")
    logger.info(f"Max {metric_name} Score: {results['max']:.4f}")
    logger.info("=" * 50)


def log_metrics_summary(
    log: Logger,
    all_results: dict[str, Any],
    context: str = "RESULTS",
) -> None:
    """Log metrics summary to log file (not console).

    Args:
        log: Logger instance
        all_results: Dictionary of all metric results
        context: Context string for header (e.g., "TEST RESULTS", "VALIDATION RESULTS")
    """
    from src.logging import log_header

    log_header(log, f"{context} SUMMARY", print_too=False)

    for metric_name, results in all_results.items():
        log.info(f"\\n{metric_name}:")
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

    # Log number of cases from first metric
    if all_results:
        first_metric_name = next(iter(all_results.keys()))
        num_cases = len(all_results[first_metric_name]["all_scores"])
        log.info(f"\\nNumber of cases: {num_cases}")
