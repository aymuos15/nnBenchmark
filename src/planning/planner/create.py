"""
Experiment plan creation and configuration management.

This module contains:
- ExperimentPlan dataclass: Configuration plan generated from dataset fingerprint
- create_experiment_plan(): Main orchestrator using nnU-Net EXACT heuristics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.planning.constants import PLANNING_CONSTANTS
from src.planning.fingerprinting.fingerprint import DatasetFingerprint
from src.planning.planner.heuristics import (
    calculate_deep_supervision_weights,
    calculate_feature_channels,
    calculate_target_spacing,
)
from src.planning.planner.sizing import (
    calculate_batch_size,
    calculate_initial_patch_size,
)
from src.planning.planner.topology import get_pool_and_conv_props


@dataclass
class ExperimentPlan:
    """
    Configuration plan generated from dataset fingerprint.
    Contains all parameters needed to generate a DynUNet training config YAML.

    DynUNet is used to exactly match nnU-Net PlainConvUNet architecture.
    """

    # Dataset info
    dataset_name: str
    num_classes: int
    is_2d: bool

    # Dataset fingerprint properties (for tracking)
    median_shape: tuple[int, ...]  # Median image shape in voxels
    median_spacing: tuple[float, ...]  # Median voxel spacing
    foreground_intensity_mean: float  # Mean foreground intensity

    # Patch and batch configuration
    patch_size: tuple[int, ...]
    batch_size: int

    # DynUNet architecture (matches nnU-Net PlainConvUNet exactly)
    filters: list[int]  # Feature channels per stage (nnUNet: features_per_stage)
    kernel_size: list[tuple[int, ...]]  # Kernel size per stage
    strides: list[
        tuple[int, ...]
    ]  # Strides per stage (includes [1,1,1] at first level)
    upsample_kernel_size: list[tuple[int, ...]]  # Upsample kernel sizes for decoder

    # Deep supervision (nnU-Net style, always enabled)
    deep_supervision: bool  # Always True following nnU-Net approach
    ds_weights: list[float]  # Decreasing weights per decoder stage

    # Intensity normalization
    normalization_scheme: str
    intensity_clip_min: float
    intensity_clip_max: float

    # Spacing info (for transforms)
    target_spacing: tuple[float, ...]


# DOC: EXPERIMENT_PLAN_CREATION | Category: Constant+Adaptive | Documentation: docs/planning.md
# Description: 8-step process calculating all architecture parameters from fingerprint
# Function: create_experiment_plan | Documentation: docs/planning.md Step 2
def create_experiment_plan(
    fingerprint: DatasetFingerprint, gpu_memory_gb: float = 8.0
) -> ExperimentPlan:
    """
    Create experiment plan from dataset fingerprint using nnU-Net EXACT heuristics.

    This implementation exactly matches nnU-Net's get_plans_for_configuration method.

    Args:
        fingerprint: Dataset fingerprint with aggregated statistics
        gpu_memory_gb: Target GPU memory in GB (default: 8)

    Returns:
        ExperimentPlan with optimized configuration

    """
    logger.debug("Creating experiment plan using nnU-Net EXACT heuristics...")
    logger.debug(f"Target GPU memory: {gpu_memory_gb} GB")

    # Step 1: Calculate target spacing
    target_spacing = calculate_target_spacing(fingerprint)
    logger.debug(f"Target spacing: {target_spacing}")

    # Step 2: Calculate initial patch size (nnU-Net exact formula)
    ndim = 2 if fingerprint.is_2d else 3
    # Extract spatial dimensions only (skip channel dimension)
    # median_shape is in format [C, D, H, W] or [C, H, W]
    spatial_shape = (
        fingerprint.median_shape[1:]
        if len(fingerprint.median_shape) > ndim
        else fingerprint.median_shape
    )
    median_shape = spatial_shape[:ndim]  # Take first ndim spatial dimensions
    initial_patch = calculate_initial_patch_size(
        target_spacing, median_shape, fingerprint.is_2d
    )
    logger.debug(f"Initial patch size: {initial_patch}")

    # Step 3: Get model topology via get_pool_and_conv_props (nnU-Net exact)
    unet_featuremap_min_edge_length = (
        PLANNING_CONSTANTS.MIN_FEATURE_MAP_SIZE
    )  # nnU-Net constant
    (
        num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size,
        shape_must_be_divisible_by,
    ) = get_pool_and_conv_props(
        spacing=target_spacing[:ndim],
        patch_size=initial_patch,
        min_feature_map_size=unet_featuremap_min_edge_length,
        max_numpool=999999,
    )

    # Keep ALL strides including the first [1,1,1] for DynUNet (matches nnUNet exactly)
    strides = list(pool_op_kernel_sizes)
    num_stages = len(
        strides
    )  # Number of encoder stages (including first no-downsample stage)
    logger.debug(f"Model topology: {num_stages} stages")
    logger.debug(f"Strides (includes [1,1,1] at first level): {strides}")
    logger.debug(f"Conv kernel sizes: {conv_kernel_sizes}")
    logger.debug(f"Adjusted patch size: {patch_size}")
    logger.debug(f"Must be divisible by: {shape_must_be_divisible_by}")

    # Step 4: Calculate feature channels (nnU-Net exact)
    filters = calculate_feature_channels(num_stages, fingerprint.is_2d)
    logger.debug(f"Feature channels (filters): {filters}")

    # Step 4b: Generate kernel sizes (all 3x3x3 for nnUNet)
    kernel_sizes = [tuple([3] * ndim) for _ in range(num_stages)]
    logger.debug(f"Kernel sizes: {kernel_sizes}")

    # Step 4c: Calculate upsample kernel sizes (inverse of downsampling strides)
    # Skip first stride since there's no upsampling for the first level
    upsample_kernel_sizes = [stride for stride in strides[1:]]
    logger.debug(f"Upsample kernel sizes: {upsample_kernel_sizes}")

    # Step 5: Calculate deep supervision weights (nnU-Net style)
    # For DynUNet, deep_supr_num=1 means we get 2 outputs (final + 1 intermediate)
    # So we need ds_weights for 2 outputs, not all decoder stages
    deep_supr_num = 1  # DynUNet default for nnU-Net compatibility
    num_ds_outputs = deep_supr_num + 1  # final + intermediate outputs
    ds_weights = calculate_deep_supervision_weights(num_ds_outputs)
    logger.debug(
        f"Deep supervision enabled with {num_ds_outputs} outputs, weights: {ds_weights}"
    )

    # Step 6: Calculate approximate total voxels in dataset
    num_cases = fingerprint.num_training_cases
    approximate_n_voxels_dataset = float(num_cases * np.prod(median_shape))
    logger.debug(f"Approximate dataset voxels: {approximate_n_voxels_dataset:.0f}")

    # Step 7: Calculate batch size (nnU-Net exact formula)
    num_input_channels = 1  # Assume single channel for now
    batch_size = calculate_batch_size(
        patch_size=patch_size,
        num_pool_per_axis=num_pool_per_axis,
        num_input_channels=num_input_channels,
        num_classes=fingerprint.num_classes,
        is_2d=fingerprint.is_2d,
        approximate_n_voxels_dataset=approximate_n_voxels_dataset,
        gpu_memory_target_gb=gpu_memory_gb,
    )
    logger.debug(f"Batch size: {batch_size}")

    # Step 8: Determine intensity normalization ranges
    if fingerprint.normalization_scheme == "CTNormalization":
        # CT uses percentile clipping (0.5th and 99.5th) to handle outliers
        # This follows nnU-Net v2.4.1 for standardized CT Hounsfield unit handling
        clip_min = fingerprint.intensity_percentile_00_5
        clip_max = fingerprint.intensity_percentile_99_5
        logger.debug(
            f"CT Normalization: clipping to percentile range [{clip_min}, {clip_max}]"
        )
    else:
        # Other modalities (MRI, PET) use per-case z-score normalization without clipping
        # Store percentile values for reference but they won't be used in transforms
        clip_min = fingerprint.intensity_percentile_00_5
        clip_max = fingerprint.intensity_percentile_99_5
        logger.debug(
            f"Per-case Z-score Normalization: no clipping applied (intensity range [{clip_min}, {clip_max}] for reference)"
        )

    # Convert numpy types to plain Python types
    patch_size_py = tuple(int(x) for x in patch_size)
    strides_py = [tuple(int(x) for x in stride) for stride in strides]
    filters_py = [int(x) for x in filters]
    kernel_sizes_py = [tuple(int(x) for x in ks) for ks in kernel_sizes]
    upsample_kernel_sizes_py = [
        tuple(int(x) for x in uks) for uks in upsample_kernel_sizes
    ]

    # Verify constraints: For DynUNet, len(strides) == len(filters)
    assert len(strides_py) == len(filters_py), (
        f"Filter/stride mismatch: {len(filters_py)} filters requires "
        f"{len(filters_py)} strides, but got {len(strides_py)}"
    )

    # Convert fingerprint numpy arrays to Python tuples for serialization
    median_shape_py = tuple(int(x) for x in fingerprint.median_shape)
    median_spacing_py = tuple(float(x) for x in fingerprint.median_spacing)

    plan = ExperimentPlan(
        dataset_name=fingerprint.dataset_name,
        num_classes=fingerprint.num_classes,
        is_2d=fingerprint.is_2d,
        median_shape=median_shape_py,
        median_spacing=median_spacing_py,
        foreground_intensity_mean=float(fingerprint.intensity_mean),
        patch_size=patch_size_py,
        batch_size=batch_size,
        filters=filters_py,
        kernel_size=kernel_sizes_py,
        strides=strides_py,
        upsample_kernel_size=upsample_kernel_sizes_py,
        deep_supervision=True,  # Always enabled following nnU-Net approach
        ds_weights=ds_weights,  # Exponential decay weights
        normalization_scheme=fingerprint.normalization_scheme,
        intensity_clip_min=clip_min,
        intensity_clip_max=clip_max,
        target_spacing=target_spacing,
    )

    logger.debug("Experiment plan created successfully!")
    logger.debug(
        f"Final configuration matches nnU-Net: patch={patch_size}, batch={batch_size}, stages={num_stages}"
    )

    return plan
