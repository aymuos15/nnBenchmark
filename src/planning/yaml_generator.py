"""
YAML configuration file generator from experiment plans.
Converts ExperimentPlan dataclass to YAML config with _target_ keys
for MONAI ConfigParser instantiation.
"""

from pathlib import Path
from typing import Any

import yaml

from src.planning.constants import PLANNING_CONSTANTS
from src.planning.planner.create import ExperimentPlan


def _derive_unet_params_from_dynunet(plan: ExperimentPlan) -> dict[str, Any]:
    """Derive UNet parameters from DynUNet configuration.

    Args:
        plan: ExperimentPlan with DynUNet parameters

    Returns:
        Dictionary of UNet-specific parameters
    """
    channels = plan.filters
    strides = [s[0] for s in plan.strides[1:]]
    num_res_units = 2
    return {"channels": channels, "strides": strides, "num_res_units": num_res_units}


def _build_config(
    plan: ExperimentPlan,
    dataset_name: str,
    fold: int = 0,
    num_workers: int = 0,
    cache_enabled: bool = False,
    cache_rate: float = 0.0,
) -> dict[str, Any]:
    """Build full config dict from experiment plan.

    Args:
        plan: ExperimentPlan with optimized parameters
        dataset_name: Dataset identifier
        fold: Cross-validation fold number
        num_workers: DataLoader workers
        cache_enabled: Whether to enable dataset caching
        cache_rate: Fraction of dataset to cache

    Returns:
        Complete config dictionary with _target_ keys
    """
    spatial_dims = 2 if plan.is_2d else 3
    patch_size = list(plan.patch_size)

    config: dict[str, Any] = {}

    config["_comment"] = f"Auto-generated for {plan.dataset_name} using nnU-Net heuristics"

    # Dataset
    config["dataset"] = {
        "name": dataset_name,
        "median_shape": list(plan.median_shape),
        "median_spacing": list(plan.median_spacing),
        "foreground_intensity_mean": round(plan.foreground_intensity_mean, 2),
        "spatial_size": patch_size,
        "num_classes": plan.num_classes,
        "fold": fold,
        "cache": {
            "enabled": cache_enabled,
            "cache_rate": cache_rate,
        },
    }

    # Model (flat — only selected model's params)
    config["model"] = {
        "_target_": "src.models.dynunet.NativeDSDynUNet",
        "spatial_dims": spatial_dims,
        "in_channels": plan.num_input_channels,
        "out_channels": plan.num_classes,
        "deep_supervision": plan.deep_supervision,
        "filters": plan.filters,
        "kernel_size": [list(ks) for ks in plan.kernel_size],
        "strides": [list(s) for s in plan.strides],
        "upsample_kernel_size": [list(uks) for uks in plan.upsample_kernel_size],
        "norm_name": ["INSTANCE", {"affine": True}],
        "act_name": ["leakyrelu", {"inplace": True, "negative_slope": 0.01}],
        "res_block": False,
    }

    # Training
    config["training"] = {
        "epochs": PLANNING_CONSTANTS.EPOCHS,
        "num_iterations_per_epoch": PLANNING_CONSTANTS.NUM_ITERATIONS_PER_EPOCH,
        "batch_size": plan.batch_size,
        "learning_rate": PLANNING_CONSTANTS.LEARNING_RATE,
        "val_interval": PLANNING_CONSTANTS.VAL_INTERVAL,
        "num_workers": num_workers,
        "checkpoint_metric": "DiceMetric",
        "plot_metrics": ["DiceMetric", "SurfaceDiceMetric"],
        "mixed_precision": True,
        "ds_weights": plan.ds_weights,
        "deep_supr_num": plan.deep_supr_num,
    }

    # Optimizer
    config["optimizer"] = {
        "_target_": "torch.optim.SGD",
        "weight_decay": PLANNING_CONSTANTS.WEIGHT_DECAY,
        "momentum": PLANNING_CONSTANTS.MOMENTUM,
        "nesterov": PLANNING_CONSTANTS.NESTEROV,
    }

    # Loss
    config["loss"] = {
        "_target_": "monai.losses.DiceCELoss",
        "to_onehot_y": True,
        "softmax": True,
        "batch": plan.batch_dice,
    }

    # Validation metrics (fast — Dice only)
    config["validation_metrics"] = [
        {
            "_target_": "monai.metrics.DiceMetric",
            "include_background": False,
            "reduction": "mean_batch",
            "num_classes": plan.num_classes,
        },
    ]

    # Inference metrics (full — Dice + Surface Dice)
    config["inference_metrics"] = [
        {
            "_target_": "monai.metrics.DiceMetric",
            "include_background": False,
            "reduction": "mean_batch",
            "num_classes": plan.num_classes,
        },
        {
            "_target_": "monai.metrics.SurfaceDiceMetric",
            "include_background": False,
            "reduction": "mean_batch",
            "class_thresholds": [2.0] * (plan.num_classes - 1),
        },
    ]

    # Transforms
    config["transforms"] = _build_transforms(plan, patch_size, spatial_dims)

    # Inference
    config["inference"] = {"batch_size": 1}

    return config


def _build_transforms(
    plan: ExperimentPlan, patch_size: list[int], spatial_dims: int
) -> dict[str, list[dict[str, Any]]]:
    """Build transforms config with _target_ keys."""
    common: list[dict[str, Any]] = []

    # LoadImaged
    load_cfg: dict[str, Any] = {
        "_target_": "monai.transforms.LoadImaged",
        "keys": ["image", "label"],
    }
    if plan.is_2d:
        load_cfg["ensure_channel_first"] = True
    common.append(load_cfg)

    # CT intensity clipping
    if plan.normalization_scheme == "CTNormalization":
        common.append({
            "_target_": "monai.transforms.ScaleIntensityRanged",
            "keys": ["image"],
            "a_min": plan.intensity_clip_min,
            "a_max": plan.intensity_clip_max,
            "b_min": plan.intensity_clip_min,
            "b_max": plan.intensity_clip_max,
            "clip": True,
        })

    # Z-Score normalization
    common.append({
        "_target_": "monai.transforms.NormalizeIntensityd",
        "keys": ["image"],
        "nonzero": False,
        "channel_wise": False,
    })

    # Spatial padding
    common.append({
        "_target_": "monai.transforms.SpatialPadd",
        "keys": ["image", "label"],
        "spatial_size": patch_size,
        "mode": "constant",
    })

    # To tensor
    common.append({
        "_target_": "monai.transforms.ToTensord",
        "keys": ["image", "label"],
    })

    # Training transforms
    train: list[dict[str, Any]] = []

    # Random crop with foreground oversampling
    train.append({
        "_target_": "monai.transforms.RandCropByPosNegLabeld",
        "keys": ["image", "label"],
        "label_key": "label",
        "spatial_size": patch_size,
        "pos": 1,
        "neg": 2,
        "num_samples": 1,
    })

    # Rotation
    train.append({
        "_target_": "monai.transforms.RandRotated",
        "keys": ["image", "label"],
        "prob": 0.2,
        "range_x": 0.5236,
        "range_y": 0.5236,
        "range_z": 0.5236,
        "mode": ["bilinear", "nearest"],
        "padding_mode": "border",
    })

    # Scaling
    train.append({
        "_target_": "monai.transforms.RandZoomd",
        "keys": ["image", "label"],
        "prob": 0.2,
        "min_zoom": 0.7,
        "max_zoom": 1.4,
        "keep_size": True,
        "mode": ["trilinear", "nearest"],
    })

    # Mirroring (all axes)
    for axis in range(len(plan.patch_size)):
        train.append({
            "_target_": "monai.transforms.RandFlipd",
            "keys": ["image", "label"],
            "prob": 0.5,
            "spatial_axis": axis,
        })

    # Gaussian noise
    train.append({
        "_target_": "monai.transforms.RandGaussianNoised",
        "keys": ["image"],
        "prob": 0.1,
        "mean": 0.0,
        "std": 0.1,
    })

    # Gaussian blur
    train.append({
        "_target_": "monai.transforms.RandGaussianSmoothd",
        "keys": ["image"],
        "prob": 0.2,
        "sigma_x": [0.5, 1.0],
        "sigma_y": [0.5, 1.0],
        "sigma_z": [0.5, 1.0],
    })

    # Brightness
    train.append({
        "_target_": "monai.transforms.RandScaleIntensityd",
        "keys": ["image"],
        "factors": [0.75, 1.25],
        "prob": 0.15,
    })

    # Contrast (custom)
    train.append({
        "_target_": "src.transforms.contrast.RandContrastd",
        "keys": ["image"],
        "prob": 0.15,
        "contrast_range": [0.75, 1.25],
        "preserve_range": True,
    })

    # Low-resolution simulation
    train.append({
        "_target_": "monai.transforms.RandZoomd",
        "keys": ["image", "label"],
        "prob": 0.25,
        "min_zoom": 0.5,
        "max_zoom": 1.0,
        "mode": ["bilinear", "nearest"],
        "padding_mode": "edge",
    })

    # Gamma with inversion
    train.append({
        "_target_": "monai.transforms.RandAdjustContrastd",
        "keys": ["image"],
        "prob": 0.1,
        "gamma": [0.7, 1.5],
        "invert_image": True,
        "retain_stats": True,
    })

    # Gamma without inversion
    train.append({
        "_target_": "monai.transforms.RandAdjustContrastd",
        "keys": ["image"],
        "prob": 0.3,
        "gamma": [0.7, 1.5],
        "invert_image": False,
        "retain_stats": True,
    })

    # Ensure consistent patch size
    train.append({
        "_target_": "monai.transforms.ResizeWithPadOrCropd",
        "keys": ["image", "label"],
        "spatial_size": patch_size,
        "mode": ["constant", "constant"],
    })

    # Val/test transforms
    val_test = [{
        "_target_": "monai.transforms.CenterSpatialCropd",
        "keys": ["image", "label"],
        "roi_size": patch_size,
    }]

    return {
        "common": common,
        "train": train,
        "val": val_test,
        "test": list(val_test),  # copy
    }


def generate_config_yaml(
    plan: ExperimentPlan,
    dataset_dir: str,
    output_path: str,
    fold: int = 0,
    num_workers: int | None = None,
    cache_enabled: bool | None = None,
    cache_rate: float | None = None,
) -> None:
    """
    Generate YAML configuration file from experiment plan.

    Args:
        plan: ExperimentPlan with optimized parameters
        dataset_dir: Path to dataset directory (for extracting dataset name)
        output_path: Where to save the YAML config
        fold: Which fold to use for training (default: 0)
        num_workers: Number of DataLoader workers (default: 0)
        cache_enabled: Whether to enable caching (default: False)
        cache_rate: Cache rate fraction 0.0-1.0 (default: 0.0)
    """
    dataset_name = Path(dataset_dir).name

    config = _build_config(
        plan=plan,
        dataset_name=dataset_name,
        fold=fold,
        num_workers=num_workers if num_workers is not None else 0,
        cache_enabled=cache_enabled if cache_enabled is not None else False,
        cache_rate=cache_rate if cache_rate is not None else 0.0,
    )

    # Write header + YAML
    header = (
        f"# {'=' * 78}\n"
        f"# {plan.dataset_name} - Auto-Generated Configuration\n"
        f"# {'=' * 78}\n"
        f"# Patch Size: {list(plan.patch_size)}\n"
        f"# Batch Size: {plan.batch_size}\n"
        f"# Normalization: {plan.normalization_scheme}\n"
        f"# Spatial Dimensions: {'2D' if plan.is_2d else '3D'}\n"
        f"# Number of Classes: {plan.num_classes}\n"
        f"# {'=' * 78}\n\n"
    )

    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
